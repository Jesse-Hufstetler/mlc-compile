"""Dory (looped normalized-GPT) for MLC-LLM.

Transcribed from llama.cpp src/models/dory.cpp. The shared recurrent block is unrolled
`n_recurrent_loops` times; weights are shared, every virtual attention layer has its own KV cache
("per_loop" mode).
"""

import dataclasses
import math
from typing import Any, Dict, List, Optional  # noqa: UP035

from tvm import relax, te, tirx
from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import Tensor, op

from mlc_llm import op as op_ext
from mlc_llm.model.model_utils import index_last_token
from mlc_llm.nn import PagedKVCache, RopeMode
from mlc_llm.support import logging
from mlc_llm.support.config import ConfigBase

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class DoryConfig(ConfigBase):
    """Configuration of the Dory model."""

    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    n_input_layers: int
    n_recurrent_layers: int
    n_output_layers: int
    n_recurrent_loops: int
    layer_types: List[str]  # noqa: UP006
    sliding_window: int
    rope_parameters: Dict[str, Any]  # noqa: UP006
    initializer_range: float
    ngpt_alpha_init: float
    attention_output_gate_logit_scale_init: Optional[float] = None
    max_position_embeddings: int = 262144
    context_window_size: int = 0
    prefill_chunk_size: int = 0
    tensor_parallel_shards: int = 1
    max_batch_size: int = 1
    sliding_window_size: int = -1
    kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)  # noqa: UP006

    def __post_init__(self):
        assert len(self.layer_types) == self.num_hidden_layers
        assert self.n_input_layers + self.n_recurrent_layers + self.n_output_layers == (
            self.num_hidden_layers
        )
        assert set(self.layer_types) <= {"sliding_attention", "full_attention", "mlp"}
        assert self.num_attention_heads % self.num_key_value_heads == 0
        assert self.tensor_parallel_shards == 1
        if self.attention_output_gate_logit_scale_init is None:
            self.attention_output_gate_logit_scale_init = math.sqrt(self.hidden_size)
        if self.context_window_size == 0:
            self.context_window_size = self.max_position_embeddings
        # The cache owns the per-layer sliding policy; keep the package-level setting off so
        # full-attention layers keep the whole context.
        self.sliding_window_size = -1
        if self.prefill_chunk_size <= 0 or self.prefill_chunk_size > self.sliding_window:
            self.prefill_chunk_size = min(self.sliding_window, self.context_window_size)

    def virtual_layers(self) -> List[int]:  # noqa: UP006
        """Physical layer index executed at each virtual step."""
        i, r, o = self.n_input_layers, self.n_recurrent_layers, self.n_output_layers
        return (
            list(range(i))
            + [i + k for _ in range(self.n_recurrent_loops) for k in range(r)]
            + [i + r + k for k in range(o)]
        )

    def attn_kinds(self) -> List[str]:  # noqa: UP006
        """KV-cache kind for each virtual attention layer, in execution order."""
        kinds = []
        for phys in self.virtual_layers():
            lt = self.layer_types[phys]
            if lt != "mlp":
                kinds.append("mha" if lt == "full_attention" else "mha_sliding")
        return kinds


def l2norm(x: Tensor) -> Tensor:
    """x / ||x||_2 over the last axis, accumulated in float32."""
    expr = x._expr  # pylint: disable=protected-access
    dtype = x.dtype
    if dtype != "float32":
        expr = relax.op.astype(expr, "float32")
    sq = relax.op.multiply(expr, expr)
    ss = relax.op.sum(sq, axis=[-1], keepdims=True)
    inv = relax.op.divide(
        relax.const(1.0, "float32"),
        relax.op.sqrt(relax.op.add(ss, relax.const(1e-6, "float32"))),
    )
    out = relax.op.multiply(expr, inv)
    if dtype != "float32":
        out = relax.op.astype(out, dtype)
    return op.wrap_nested(out, "dory_l2norm")


def apply_rope(values: Tensor, positions: Tensor, theta: float, name: str) -> Tensor:
    """NeoX-style (rotate-half) RoPE over the full head dim; values: (b, s, h, d)."""

    def _rope(values: te.Tensor, position_map: te.Tensor):
        batch, seq_len, _, head_dim = values.shape
        half_dim = head_dim // 2
        dtype = values.dtype

        def _value(b: tirx.Var, s: tirx.Var, h: tirx.Var, d: tirx.Var):
            frequency_index = d % half_dim
            angle = position_map[b * seq_len + s] / tirx.power(
                theta, (2 * frequency_index) / tirx.const(head_dim, "float32")
            )
            partner = tirx.if_then_else(d < half_dim, d + half_dim, d - half_dim)
            sign = tirx.if_then_else(d < half_dim, tirx.const(-1, dtype), tirx.const(1, dtype))
            value = values[b, s, h, d]
            rotated = values[b, s, h, partner] * sign
            return (value * tirx.cos(angle) + rotated * tirx.sin(angle)).astype(dtype)

        return te.compute(values.shape, _value, name="dory_rope")

    return op.tensor_expr_op(_rope, name, [values, positions])


class DoryLayer(nn.Module):
    """One sub-layer (attention or MLP) with nGPT residual mixing."""

    def __init__(self, config: DoryConfig, phys: int):
        self.kind = config.layer_types[phys]
        self.hidden_size = config.hidden_size
        self.alpha = nn.Parameter((config.hidden_size,))
        if self.kind == "mlp":
            self.ffn = config.intermediate_size
            self.gate_up_proj = nn.Linear(config.hidden_size, 2 * self.ffn, bias=False)
            self.down_proj = nn.Linear(self.ffn, config.hidden_size, bias=False)
            self.suv_gate = nn.Parameter((self.ffn,))
            self.suv_up = nn.Parameter((self.ffn,))
        else:
            self.nh = config.num_attention_heads
            self.nkv = config.num_key_value_heads
            self.hd = config.head_dim
            nq, nkv = self.nh * self.hd, self.nkv * self.hd
            self.nq, self.nkv_dim = nq, nkv
            self.theta = float(config.rope_parameters[self.kind]["rope_theta"])
            self.qkvg_proj = nn.Linear(config.hidden_size, 2 * nq + 2 * nkv, bias=False)
            self.o_proj = nn.Linear(nq, config.hidden_size, bias=False)
            self.sq = nn.Parameter((nq,))
            self.sgate = nn.Parameter((nq,))

    def _mlp(self, x: Tensor) -> Tensor:
        gate, up = op.split(self.gate_up_proj(x), 2, axis=-1)
        y = l2norm(op.silu(gate * self.suv_gate) * (up * self.suv_up))
        return self.down_proj(y)

    def _attn(
        self, x: Tensor, cache: PagedKVCache, positions: Tensor, kv_layer: int
    ) -> Tensor:
        b, s, _ = x.shape
        nq, nkv = self.nq, self.nkv_dim
        q, k, v, g = op.split(self.qkvg_proj(x), [nq, nq + nkv, nq + 2 * nkv], axis=-1)
        q = op.reshape(q, (b, s, self.nh, self.hd))
        k = op.reshape(k, (b, s, self.nkv, self.hd))
        v = op.reshape(v, (b, s, self.nkv, self.hd))
        q = apply_rope(l2norm(q), positions, self.theta, "dory_q_rope")
        q = q * op.reshape(self.sq, (1, 1, self.nh, self.hd))
        k = apply_rope(l2norm(k), positions, self.theta, "dory_k_rope")
        gate = op.sigmoid(g * self.sgate)
        qkv = op.concat([q, k, v], dim=2)
        out = cache.attention_with_fused_qkv(
            kv_layer, qkv, self.nh, sm_scale=1.0 / math.sqrt(self.hd)
        )
        out = op.reshape(out, (b, s, nq))
        return self.o_proj(l2norm(out * gate))

    def forward(
        self, x: Tensor, cache: PagedKVCache, positions: Tensor, kv_layer: int
    ) -> Tensor:
        if self.kind == "mlp":
            y = self._mlp(x)
        else:
            y = self._attn(x, cache, positions, kv_layer)
        a, c = l2norm(x), l2norm(y)
        return l2norm(a + (c - a) * self.alpha)


class DoryModel(nn.Module):
    def __init__(self, config: DoryConfig):
        self.config = config
        self.embed_tokens = nn.Embedding("vocab_size", config.hidden_size)
        self.layers = nn.ModuleList(
            [DoryLayer(config, p) for p in range(config.num_hidden_layers)]
        )

    def forward(self, input_embed: Tensor, cache: PagedKVCache) -> Tensor:
        cfg = self.config
        b, s, _ = input_embed.shape
        positions = cache.get_query_positions(b * s)
        h = input_embed
        kv = 0

        def run(h: Tensor, phys: int) -> Tensor:
            nonlocal kv
            layer = self.layers[phys]
            out = layer(h, cache, positions, kv)
            if layer.kind != "mlp":
                kv += 1
            return out

        for p in range(cfg.n_input_layers):
            h = run(h, p)
        input_h = h
        for _ in range(cfg.n_recurrent_loops):
            h = l2norm(h + input_h)
            for p in range(cfg.n_input_layers, cfg.n_input_layers + cfg.n_recurrent_layers):
                h = run(h, p)
            h = l2norm(h)
        for p in range(
            cfg.n_input_layers + cfg.n_recurrent_layers, cfg.num_hidden_layers
        ):
            h = run(h, p)
        return h


class DoryForCausalLM(nn.Module):
    def __init__(self, config: DoryConfig):
        self.config = config
        self.model = DoryModel(config)
        self.lm_head = nn.Linear(config.hidden_size, "vocab_size", bias=False)
        self.logit_scale = nn.Parameter(("vocab_size",))
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.dtype = "float32"

    def to(self, dtype: Optional[str] = None):
        super().to(dtype=dtype)
        if dtype is not None:
            self.dtype = dtype

    def get_logits(self, hidden_states: Tensor) -> Tensor:
        logits = self.lm_head(hidden_states)
        if logits.dtype != "float32":
            logits = logits.astype("float32")
        return logits * op.astype(self.logit_scale, "float32")

    def batch_forward(
        self,
        input_embeds: Tensor,
        paged_kv_cache: PagedKVCache,
        logit_positions: Optional[Tensor] = None,
    ):
        op_ext.configure()
        hidden_states = self.model(input_embeds, paged_kv_cache)
        if logit_positions is not None:
            hidden_states = op.take(hidden_states, logit_positions, axis=1)
        return self.get_logits(hidden_states)

    def embed(self, input_ids: Tensor):
        return self.model.embed_tokens(input_ids)

    def prefill(self, input_embed: Tensor, paged_kv_cache: PagedKVCache):
        op_ext.configure()
        hidden_states = self.model(input_embed, paged_kv_cache)
        hidden_states = index_last_token(hidden_states)
        return self.get_logits(hidden_states), paged_kv_cache

    def decode(self, input_embed: Tensor, paged_kv_cache: PagedKVCache):
        op_ext.configure()
        hidden_states = self.model(input_embed, paged_kv_cache)
        return self.get_logits(hidden_states), paged_kv_cache

    def batch_prefill(
        self, input_embeds: Tensor, logit_positions: Tensor, paged_kv_cache: PagedKVCache
    ):
        return self.batch_forward(input_embeds, paged_kv_cache, logit_positions), paged_kv_cache

    def batch_decode(self, input_embeds: Tensor, paged_kv_cache: PagedKVCache):
        return self.batch_forward(input_embeds, paged_kv_cache), paged_kv_cache

    def batch_verify(self, input_embeds: Tensor, paged_kv_cache: PagedKVCache):
        return self.batch_forward(input_embeds, paged_kv_cache), paged_kv_cache

    def create_paged_kv_cache(
        self,
        max_batch_size: tirx.Var,
        max_total_seq_len: tirx.Var,
        prefill_chunk_size: tirx.Var,
        page_size: tirx.Var,
        support_sliding_window: tirx.Var,
    ) -> PagedKVCache:
        cfg = self.config
        kinds = cfg.attn_kinds()
        return PagedKVCache.create_generic(
            attn_kind=kinds,
            max_batch_size=max_batch_size,
            max_total_seq_len=max_total_seq_len,
            prefill_chunk_size=prefill_chunk_size,
            page_size=page_size,
            support_sliding_window=support_sliding_window,
            num_hidden_layers=len(kinds),
            num_attention_heads=cfg.num_attention_heads,
            num_key_value_heads=cfg.num_key_value_heads,
            qk_head_dim=cfg.head_dim,
            v_head_dim=cfg.head_dim,
            rope_mode=RopeMode.NONE,
            rope_scale=1,
            rope_theta=10_000,
            dtype=self.dtype,
            layer_sliding_window_size=cfg.sliding_window,
        )

    def get_default_spec(self):
        packed = {"param_mode": "packed", "effect_mode": "none"}
        cache = nn.spec.Object(object_type=PagedKVCache)
        h = self.hidden_size
        mod_spec = {
            "embed": {"input_ids": nn.spec.Tensor(["seq_len"], "int32"), "$": packed},
            "prefill": {
                "input_embed": nn.spec.Tensor([1, "seq_len", h], self.dtype),
                "paged_kv_cache": cache,
                "$": packed,
            },
            "decode": {
                "input_embed": nn.spec.Tensor([1, 1, h], self.dtype),
                "paged_kv_cache": cache,
                "$": packed,
            },
            "batch_prefill": {
                "input_embeds": nn.spec.Tensor([1, "seq_len", h], self.dtype),
                "logit_positions": nn.spec.Tensor(["batch_size"], "int32"),
                "paged_kv_cache": cache,
                "$": packed,
            },
            "batch_decode": {
                "input_embeds": nn.spec.Tensor(["batch_size", 1, h], self.dtype),
                "paged_kv_cache": cache,
                "$": packed,
            },
            "batch_verify": {
                "input_embeds": nn.spec.Tensor([1, "seq_len", h], self.dtype),
                "paged_kv_cache": cache,
                "$": packed,
            },
            "create_paged_kv_cache": {
                "max_batch_size": int,
                "max_total_seq_len": int,
                "prefill_chunk_size": int,
                "page_size": int,
                "support_sliding_window": int,
                "$": {"param_mode": "none", "effect_mode": "none"},
            },
        }
        return nn.spec.ModuleSpec.from_raw(mod_spec, self)
