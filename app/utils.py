"""Utility & helper functions."""

import logging
import re
import xml.etree.ElementTree as ET
from typing import ClassVar

import xml_pydantic
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage

logging.basicConfig(level=logging.DEBUG)
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


def load_chat_model(fully_specified_name: str, **kwargs) -> BaseChatModel:
    """Load a chat model from a fully specified name.

    Args:
        fully_specified_name (str): String in the format 'provider/model'.

        kwargs: Additional model-specific keyword args to pass to the underlying chat model's __init__ method. Common parameters include:

        temperature (int): Model temperature for controlling randomness.

        max_tokens: Maximum number of output tokens.

        timeout: Maximum time (in seconds) to wait for a response.

        max_retries: Maximum number of retry attempts for failed requests.

        base_url: Custom API endpoint URL.

        rate_limiter: A [BaseRateLimiter][langchain_core.rate_limiters.BaseRateLimiter] instance to control request rate.
        Refer to the specific model provider's integration reference for all available parameters.
    """
    provider, model = fully_specified_name.split("/", maxsplit=1)

    return init_chat_model(model, model_provider=provider, **kwargs)


def pascal_to_snake(name: str):
    """Insert an underscore before any uppercase character that has a lowercase character or digit before it, then convert the whole string to lowercase."""
    snake_case = re.sub("([a-z0-9])([A-Z])", r"\1_\2", name).lower()
    return snake_case


def normalize_whitespace(str: str) -> str:
    """Collapse all runs of whitespace in ``str`` to a single space and strip the ends."""
    return re.sub(r"\s+", " ", str).strip()


class XmlDto:
    """Mixin that adds ``model_dump_xml()`` to dynamic xml_pydantic DTOs.

    Subclasses must set ``_root_tag`` to the XML element name to use as the
    serialisation root.

    Example::

        class MyDto(xml_pydantic.define_model("MyDto", SCHEMA), XmlDto):
            _root_tag = "my_element"
    """

    _root_tag: ClassVar[str] = ""

    def model_dump_xml(self) -> str:
        """Serialize the model to an XML string."""
        return xml_pydantic.serializers.model_to_xml_string(
            self,
            root_tag=self._root_tag,  # type: ignore[arg-type]
        )


def load_xml_prompt(path: str) -> str:
    """Load a system prompt from a well formed xml file. All elements within the root will be serialized and concatenated into a single prompt string designed for performance with Claude models."""
    return " ".join(
        normalize_whitespace(ET.tostring(e, encoding="unicode"))
        for e in ET.parse(path).getroot()
    )
