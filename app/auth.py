"""LangGraph auth handler for per-request API key injection."""

from typing import Any

from langgraph_sdk import Auth

auth = Auth()


def is_valid_key(api_key: str) -> bool:
    is_valid = False  # your API key validation logic
    return is_valid


@auth.authenticate
async def authenticate(headers: dict) -> dict[str, Any]:
    api_key = headers.get(b"x-api-key")
    if not api_key or not is_valid_key(api_key):
        raise Auth.exceptions.HTTPException(status_code=401, detail="Invalid API key")

    return {"identity": "user", "anthropic_api_key": api_key}


@auth.on
async def on_request(
    ctx: Auth.types.AuthContext,
    value: dict[str, Any],
) -> None:
    """Inject the Anthropic API key into the run's configurable config.

    Runs before the graph executes, making the key available in Context.anthropic_api_key.
    """
    api_key: str = getattr(ctx.user, "anthropic_api_key", "")
    if api_key:
        config = value.get("config")
        if config is None:
            value["config"] = {"configurable": {"anthropic_api_key": api_key}}
        else:
            config.setdefault("configurable", {})["anthropic_api_key"] = api_key
