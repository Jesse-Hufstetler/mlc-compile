"""Dory chat template (ChatML with an open <think> block, as in the model's own template)."""

from mlc_llm.protocol.conversation_protocol import Conversation, MessagePlaceholders

from .registry import ConvTemplateRegistry

ConvTemplateRegistry.register_conv_template(
    Conversation(
        name="dory",
        system_template=f"<|im_start|>system\n{MessagePlaceholders.SYSTEM.value}<|im_end|>\n",
        system_message="You are a helpful assistant.",
        roles={
            "user": "<|im_start|>user",
            "assistant": "<|im_start|>assistant\n<think>",
        },
        seps=["<|im_end|>\n"],
        role_content_sep="\n",
        role_empty_sep="\n",
        stop_str=["<|im_end|>"],
        stop_token_ids=[2, 11],
    )
)
