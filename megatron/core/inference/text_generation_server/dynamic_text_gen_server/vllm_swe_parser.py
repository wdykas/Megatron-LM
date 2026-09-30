# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Optional non-streaming Nano/Qwen3 parser used by SWE agents.

Requires a compatible vLLM installation (0.25.1 API) in the HTTP frontend
process. No vLLM engine is constructed; sampling and token metadata remain
owned by Megatron. The supplied plugin must register ``nano_v3``.
"""


def load_vllm_swe_parser(tokenizer, plugin_path):
    """Load the optional dependency once per frontend, keeping defaults untouched."""
    from transformers import PreTrainedTokenizerBase
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.parser import DelegatingParser
    from vllm.reasoning.abs_reasoning_parsers import ReasoningParserManager
    from vllm.tool_parsers.abstract_tool_parser import ToolParserManager

    ReasoningParserManager.import_reasoning_parser(plugin_path)
    reasoning_parser_cls = ReasoningParserManager.get_reasoning_parser("nano_v3")
    tool_parser_cls = ToolParserManager.get_tool_parser("qwen3_coder")

    # Megatron text tokenizers wrap their library, which wraps the HF tokenizer.
    if not isinstance(tokenizer, PreTrainedTokenizerBase):
        tokenizer = getattr(tokenizer, "_tokenizer", tokenizer)
        tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    if not isinstance(tokenizer, PreTrainedTokenizerBase):
        raise TypeError("The vLLM SWE parser requires a Hugging Face tokenizer")

    class SWEParser(DelegatingParser):
        """Compose the original Nano reasoning plugin with Qwen3 tool parsing."""

    SWEParser.reasoning_parser_cls = reasoning_parser_cls
    SWEParser.tool_parser_cls = tool_parser_cls

    def prepare_request(body, chat_template_kwargs):
        # Pass parser context only: Megatron remains responsible for generation
        # validation and sampling, including its tokenized/offloaded extensions.
        request = ChatCompletionRequest(
            model=body.get("model") or "megatron",
            messages=body["messages"],
            tools=body.get("tools"),
            tool_choice=body.get("tool_choice"),
            parallel_tool_calls=body.get("parallel_tool_calls", True),
            chat_template_kwargs=chat_template_kwargs,
        )

        def parse(text):
            # Parser state and tool-call counters must not cross requests/choices.
            parser = SWEParser(tokenizer, request.tools)
            reasoning, content, calls = parser.parse(text, request, enable_auto_tools=True)
            metadata = {}
            if reasoning is not None:
                metadata["reasoning"] = reasoning
            if calls:
                metadata["tool_calls"] = [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {"name": call.name, "arguments": call.arguments},
                    }
                    for call in calls
                ]
            return content, metadata

        return parse

    return prepare_request
