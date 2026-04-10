"""
Cost estimation for LLM experiments.

Helps researchers know roughly what an experiment will cost before running it.
Pricing data is current as of 2026 — update as model prices change.
"""

# OpenRouter pricing (USD per million tokens)
# Format: (input_per_M, output_per_M)
MODEL_PRICING = {
    # StepFun
    "stepfun/step-3.5-flash": (0.10, 0.30),
    "stepfun/step-3.5-flash:free": (0.0, 0.0),

    # MiniMax
    "minimax/minimax-m2.7": (0.30, 1.20),
    "minimax/minimax-m2.5:free": (0.0, 0.0),

    # DeepSeek
    "deepseek/deepseek-v3.2": (0.26, 0.38),
    "deepseek/deepseek-chat-v3.1": (0.15, 0.75),
    "deepseek/deepseek-r1": (0.70, 2.50),

    # Qwen
    "qwen/qwen3.6-plus:free": (0.0, 0.0),

    # Meta
    "meta-llama/llama-3.3-70b-instruct:free": (0.0, 0.0),
    "meta-llama/llama-3.1-8b-instruct": (0.02, 0.05),

    # Google
    "google/gemma-3-27b-it:free": (0.0, 0.0),

    # Anthropic
    "anthropic/claude-3.5-sonnet": (3.00, 15.00),
    "anthropic/claude-3.7-sonnet": (3.00, 15.00),

    # OpenAI
    "openai/gpt-4o": (2.50, 10.00),
    "openai/gpt-4o-mini": (0.15, 0.60),
}

# Default token estimates per LLM call
# Reasoning models burn more tokens on thinking
DEFAULT_TOKENS_PER_CALL = {
    "input": 200,
    "output_normal": 80,        # non-reasoning model
    "output_reasoning": 800,    # reasoning model uses more tokens
}

REASONING_MODELS = {
    "stepfun/step-3.5-flash",
    "minimax/minimax-m2.7",
    "deepseek/deepseek-r1",
    "openai/o1-mini",
    "openai/o1",
}


def is_reasoning(model: str) -> bool:
    """Heuristic: does this model use reasoning tokens?"""
    return any(rm in model for rm in REASONING_MODELS)


def get_pricing(model: str) -> tuple[float, float]:
    """
    Return (input, output) price per million tokens for a model.
    Returns (0, 0) for unknown models with a warning.
    """
    if model in MODEL_PRICING:
        return MODEL_PRICING[model]
    # Try base name without suffix
    base = model.replace(":free", "")
    if base in MODEL_PRICING:
        return MODEL_PRICING[base]
    return (0.0, 0.0)


def estimate_cost(
    model: str,
    n_calls: int,
    input_tokens_per_call: int = None,
    output_tokens_per_call: int = None,
) -> dict:
    """
    Estimate the cost of an experiment in USD.

    Args:
        model: model identifier (e.g. "stepfun/step-3.5-flash")
        n_calls: total number of API calls
        input_tokens_per_call: estimated input tokens per call (default: 200)
        output_tokens_per_call: estimated output tokens per call
            (default: 80 for non-reasoning, 800 for reasoning)

    Returns:
        dict with cost breakdown
    """
    in_per_m, out_per_m = get_pricing(model)

    if input_tokens_per_call is None:
        input_tokens_per_call = DEFAULT_TOKENS_PER_CALL["input"]
    if output_tokens_per_call is None:
        if is_reasoning(model):
            output_tokens_per_call = DEFAULT_TOKENS_PER_CALL["output_reasoning"]
        else:
            output_tokens_per_call = DEFAULT_TOKENS_PER_CALL["output_normal"]

    total_input = n_calls * input_tokens_per_call
    total_output = n_calls * output_tokens_per_call

    input_cost = (total_input / 1_000_000) * in_per_m
    output_cost = (total_output / 1_000_000) * out_per_m
    total_cost = input_cost + output_cost

    return {
        "model": model,
        "n_calls": n_calls,
        "input_tokens": total_input,
        "output_tokens": total_output,
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": total_cost,
        "is_reasoning": is_reasoning(model),
    }


def print_estimate(model: str, n_calls: int, **kwargs):
    """Pretty-print a cost estimate."""
    est = estimate_cost(model, n_calls, **kwargs)

    if est["total_cost"] == 0:
        in_per_m, out_per_m = get_pricing(model)
        if in_per_m == 0 and out_per_m == 0 and ":free" in model:
            cost_str = "FREE (rate-limited)"
        else:
            cost_str = "unknown (model pricing not in registry)"
    else:
        cost_str = f"${est['total_cost']:.4f}"

    note = " [reasoning model]" if est["is_reasoning"] else ""

    print(f"Estimated cost: {cost_str}{note}", flush=True)
    print(
        f"  {est['n_calls']} API calls × "
        f"({est['input_tokens']//est['n_calls'] if est['n_calls'] else 0} in / "
        f"{est['output_tokens']//est['n_calls'] if est['n_calls'] else 0} out tokens)",
        flush=True,
    )

    return est
