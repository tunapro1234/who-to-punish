"""
Preflight checks: fail fast with actionable error messages.

Run before any expensive experiment to catch common configuration errors:
  - Missing API key
  - Unknown model name
  - Network issues

These checks are best-effort — they print warnings rather than block execution
unless something is clearly broken.
"""

import os
import sys


class PreflightError(Exception):
    """Raised when a preflight check fails fatally."""


def check_api_key(provider: str = "openrouter") -> str:
    """
    Verify the relevant API key is set in the environment.
    Returns the key value, raises PreflightError if missing.
    """
    env_vars = {
        "openrouter": ["OPEN_ROUTER_API_KEY", "OPENROUTER_API_KEY"],
        "anthropic": ["ANTHROPIC_API_KEY"],
        "openai": ["OPENAI_API_KEY"],
    }

    candidates = env_vars.get(provider, [f"{provider.upper()}_API_KEY"])
    for var in candidates:
        if os.environ.get(var):
            return os.environ[var]

    msg = (
        f"\n\n  Missing API key for {provider}.\n"
        f"  Set one of these environment variables:\n"
    )
    for var in candidates:
        msg += f"    export {var}='your-key-here'\n"
    msg += (
        f"\n  For OpenRouter, get a key at: https://openrouter.ai/settings/keys\n"
    )
    raise PreflightError(msg)


def check_model(model_name: str) -> bool:
    """
    Verify the model exists, either in our local pricing registry or
    on OpenRouter. If the model is unknown locally, queries OpenRouter
    and caches the pricing if found.

    Raises PreflightError if the model doesn't exist on OpenRouter.
    """
    from .analysis.cost import MODEL_PRICING, _fetch_from_openrouter

    base = model_name.replace(":free", "")
    if model_name in MODEL_PRICING or base in MODEL_PRICING:
        return True

    # Not in local registry — try OpenRouter
    pricing = _fetch_from_openrouter(model_name)
    if pricing is not None:
        MODEL_PRICING[model_name] = pricing
        in_per_m, out_per_m = pricing
        if in_per_m > 0 or out_per_m > 0:
            print(
                f"  [auto-registered] {model_name}: "
                f"${in_per_m:.2f}/M in, ${out_per_m:.2f}/M out",
                flush=True,
            )
        return True

    raise PreflightError(
        f"\n\n  Model '{model_name}' not found on OpenRouter.\n"
        f"  Check the model identifier — see https://openrouter.ai/models\n"
        f"  Examples: stepfun/step-3.5-flash, deepseek/deepseek-v3.2,\n"
        f"            meta-llama/llama-3.1-8b-instruct\n"
    )


def check_otree_server(server_url: str, timeout: float = 5.0) -> bool:
    """
    Verify an oTree server is reachable. Returns True if up, raises
    PreflightError otherwise.
    """
    import requests
    try:
        r = requests.get(server_url, timeout=timeout, allow_redirects=True)
        if r.status_code >= 500:
            raise PreflightError(
                f"\n\n  oTree server at {server_url} returned {r.status_code}.\n"
                f"  Check the server logs: docker logs $(docker ps -q --filter name=otree)\n"
            )
        return True
    except requests.exceptions.ConnectionError:
        raise PreflightError(
            f"\n\n  Cannot connect to oTree server at {server_url}.\n"
            f"  Start it with:\n"
            f"    docker compose up -d\n\n"
            f"  Or if running locally:\n"
            f"    cd tests/otree_server && otree devserver\n"
        )
    except requests.exceptions.Timeout:
        raise PreflightError(
            f"\n\n  oTree server at {server_url} is not responding (timeout).\n"
        )


def preflight_for_experiment(model: str, provider: str = "openrouter"):
    """Run all checks needed before an LLM-based experiment."""
    check_api_key(provider)
    check_model(model)


def preflight_for_otree(server_url: str, model: str, provider: str = "openrouter"):
    """Run all checks needed before an oTree-based experiment."""
    check_api_key(provider)
    check_model(model)
    check_otree_server(server_url)
