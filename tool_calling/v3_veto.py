from time import perf_counter

from tool_calling.baseline import generate_from_messages
from tool_calling.evaluation import parse_tool_calls


def _parse_calls(text: str) -> list[dict] | None:
    try:
        return parse_tool_calls(text)
    except ValueError:
        return None


def generate_with_v3_veto(
    model,
    tokenizer,
    messages: list[dict],
    tools: list[dict],
    max_new_tokens: int = 1024,
) -> dict:
    """Run one request on a PEFT model loaded with the v3 adapter."""
    start_time = perf_counter()
    with model.disable_adapter():
        baseline_text = generate_from_messages(
            model, tokenizer, messages, tools, max_new_tokens
        )
    baseline_latency = perf_counter() - start_time

    generated_text = baseline_text
    gate_checked = bool(_parse_calls(baseline_text))
    vetoed = False
    if gate_checked:
        v3_text = generate_from_messages(
            model, tokenizer, messages, tools, max_new_tokens
        )
        vetoed = bool(v3_text.strip()) and _parse_calls(v3_text) == []
        if vetoed:
            generated_text = v3_text

    return {
        "generated_text": generated_text,
        "baseline_generated_text": baseline_text,
        "gate_checked": gate_checked,
        "vetoed": vetoed,
        "baseline_latency_seconds": baseline_latency,
        "latency_seconds": perf_counter() - start_time,
    }
