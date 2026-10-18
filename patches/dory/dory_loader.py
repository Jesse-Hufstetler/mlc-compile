"""HF safetensors -> MLC parameter mapping for Dory.

nGPT scale vectors are folded into their effective values here (as in llama.cpp dory.cpp) so the
compiled graph only multiplies by them.
"""

import functools
import math

import numpy as np

from mlc_llm.loader import ExternMapping
from mlc_llm.quantization import Quantization

from .dory_model import DoryConfig, DoryForCausalLM


def huggingface(model_config: DoryConfig, quantization: Quantization) -> ExternMapping:
    model = DoryForCausalLM(model_config)
    if quantization is not None:
        model.to(quantization.model_dtype)
    _, named_params, _ = model.export_tvm(spec=model.get_default_spec(), allow_extern=True)
    named = dict(named_params)

    cfg = model_config
    h, hd = cfg.hidden_size, cfg.head_dim
    std = cfg.initializer_range
    alpha_scale = cfg.ngpt_alpha_init / std
    sq_scale = math.sqrt(hd) / std
    gate_scale = math.sqrt(cfg.attention_output_gate_logit_scale_init) / std
    suv_scale = math.sqrt(h)

    mapping = ExternMapping()

    def add(mlc_name, hf_names, fn):
        dtype = named[mlc_name].dtype
        mapping.add_mapping(mlc_name, hf_names, functools.partial(fn, dtype=dtype))

    def f32(x):
        return np.asarray(x).astype("float32")

    def plain(x, dtype):
        return np.asarray(x).astype(dtype)

    add("model.embed_tokens.weight", ["backbone.embeddings.weight"], plain)
    add("lm_head.weight", ["lm_head.weight"], plain)
    add("logit_scale", ["lm_head.logit_scale"], lambda x, dtype: (f32(x) / std).astype(dtype))

    for p, kind in enumerate(cfg.layer_types):
        src = f"backbone.layers.{p}."
        dst = f"model.layers.{p}."
        if kind == "mlp":
            add(
                dst + "gate_up_proj.weight",
                [src + "mixer.gate_proj.weight", src + "mixer.up_proj.weight"],
                lambda g, u, dtype: np.concatenate([np.asarray(g), np.asarray(u)], 0).astype(dtype),
            )
            add(dst + "down_proj.weight", [src + "mixer.down_proj.weight"], plain)
            add(
                dst + "suv_gate",
                [src + "mixer.suv_gate"],
                lambda x, dtype: (f32(x) * suv_scale).astype(dtype),
            )
            add(
                dst + "suv_up",
                [src + "mixer.suv_up"],
                lambda x, dtype: (f32(x) * suv_scale).astype(dtype),
            )
            alpha_name = src + "mlp_alpha"
        else:
            add(
                dst + "qkvg_proj.weight",
                [
                    src + "mixer.q_proj.weight",
                    src + "mixer.k_proj.weight",
                    src + "mixer.v_proj.weight",
                    src + "mixer.g_proj.weight",
                ],
                lambda q, k, v, g, dtype: np.concatenate(
                    [np.asarray(t) for t in (q, k, v, g)], 0
                ).astype(dtype),
            )
            add(dst + "o_proj.weight", [src + "mixer.o_proj.weight"], plain)
            add(
                dst + "sq",
                [src + "mixer.sqk"],
                lambda x, dtype: ((f32(x) * sq_scale) ** 2).astype(dtype),
            )
            add(
                dst + "sgate",
                [src + "mixer.s_gate"],
                lambda x, dtype: ((f32(x) * gate_scale) ** 2).astype(dtype),
            )
            alpha_name = src + "attn_alpha"
        add(
            dst + "alpha",
            [alpha_name],
            lambda x, dtype: np.abs(f32(x) * alpha_scale).astype(dtype),
        )

    missing = [n for n in named if n not in mapping.param_map]
    assert not missing, f"unmapped Dory parameters: {missing[:5]}"
    return mapping
