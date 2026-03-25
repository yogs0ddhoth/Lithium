"""Utility & helper functions."""

import logging
import re

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_message_text(msg: BaseMessage) -> str:
    """Get the text content of a message."""
    content = msg.content
    if isinstance(content, str):
        return content
    elif isinstance(content, dict):
        return content.get("text", "")
    else:
        txts = [c if isinstance(c, str) else (c.get("text") or "") for c in content]
        return "".join(txts).strip()


def load_chat_model(fully_specified_name: str) -> BaseChatModel:
    """Load a chat model from a fully specified name.

    Args:
        fully_specified_name (str): String in the format 'provider/model'.
    """
    _, model = fully_specified_name.split("/", maxsplit=1)

    return init_chat_model(model, temperature=0)


def pascal_to_snake(name: str):
    # Insert an underscore before any uppercase character that has a lowercase
    # character or digit before it, then convert the whole string to lowercase.
    snake_case = re.sub("([a-z0-9])([A-Z])", r"\1_\2", name).lower()
    return snake_case


def normalize_whitespace(str: str):
    return re.sub(r"\s+", " ", str).strip()
