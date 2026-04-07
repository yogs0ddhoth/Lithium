"""Shared test helpers for the xml-pydantic test suite."""

import re


def normalize_whitespace(s: str) -> str:
    """Collapse all whitespace runs to a single space and strip edges."""
    return re.sub(r"\s+", " ", s).strip()
