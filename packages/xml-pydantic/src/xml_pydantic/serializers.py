"""Convert Pydantic v2 model instances (or plain ``model_dump()`` dicts) into ``xml.etree.ElementTree`` XML trees.

Conversion rules

---------------
Scalar field     →  <fieldname>value</fieldname>
Object field     →  <fieldname> whose children are the nested object's fields
Array of objects →  <fieldname> containing one <singularized_name> subtree per item
Array of scalars →  <fieldname> containing one <item> element per value
None / null      →  self-closing <fieldname /> (no text content)
Boolean          →  lowercase "true" / "false" to match XML / JSON convention

"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Any

from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Singularisation helpers
# ---------------------------------------------------------------------------

# Ordered: longest suffix first so 'buses'→'bus' beats plain 's' rule.
SINGULAR_RULES: list[tuple[str, str]] = [
    ("ives", "ife"),  # knives  → knife
    ("ves", "f"),  # leaves  → leaf
    ("ies", "y"),  # cities  → city
    ("ses", "s"),  # buses   → bus
    ("xes", "x"),  # boxes   → box
    ("zes", "z"),  # buzzes  → buzz
    ("s", ""),  # items   → item  /  dogs → dog
]


def __singularize(word: str) -> str:
    """Return a naive singular form of *word* for use as a child element tag.

    Falls back to ``{word}_item`` when no rule produces a non-empty result
    (e.g. the word is already singular, like ``data``).

    Examples:
    --------
    >>> _singularize("addresses")
    'address'
    >>> _singularize("cities")
    'city'
    >>> _singularize("tags")
    'tag'
    >>> _singularize("data")
    'data_item'
    """
    lower = word.lower()
    for suffix, replacement in SINGULAR_RULES:
        if lower.endswith(suffix) and len(lower) > len(suffix):
            root_part = word[: len(word) - len(suffix)]
            singular = root_part + replacement
            # Guard against empty result (e.g. "s" → "")
            return singular if singular else f"{word}_item"
    return f"{word}_item"


# ---------------------------------------------------------------------------
# Scalar rendering
# ---------------------------------------------------------------------------


def __render_scalar(value: Any) -> str:
    """Convert a Python scalar to its XML text representation.

    * ``bool``   → ``"true"`` / ``"false"``  (not Python's ``True``/``False``)
    * Everything else → ``str(value)``
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


# ---------------------------------------------------------------------------
# Core recursive builders
# ---------------------------------------------------------------------------


def __append_value(parent: ET.Element, tag: str, value: Any, attrib: dict) -> None:
    """Create a ``<tag>`` child element under *parent* and populate it based on the Python type of *value*.

    Dispatch table
    ~~~~~~~~~~~~~~
    None      →  empty / self-closing element (no ``.text``)
    dict      →  recursively expand fields as child elements
    list      →  expand via :func:`_append_list`
    scalar    →  set ``.text`` to the rendered string
    """
    index: int = int(attrib.pop("index", 0))

    if index > 0:
        attrib["index"] = str(index)

    elem = ET.SubElement(parent, tag, attrib)

    if value is None:
        return  # self-closing <tag />

    if isinstance(value, dict):
        __append_dict(elem, value, {})
    elif isinstance(value, list):
        __append_list(elem, tag, value, {})
    else:
        elem.text = __render_scalar(value)


def __append_dict(parent: ET.Element, data: dict[str, Any], attrib: dict) -> None:
    """Append one child element per ``(key, value)`` pair in *data*."""
    for key, val in data.items():
        __append_value(parent, key, val, attrib)


def __append_list(
    parent: ET.Element, parent_tag: str, items: list[Any], attrib: dict
) -> None:
    """Populate *parent* with child elements sourced from *items*.

    Item type   →  child tag used
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    dict        →  singularized form of *parent_tag*
    nested list →  singularized form of *parent_tag* (recursive)
    scalar      →  literal ``item``
    """
    index: int = attrib.pop("index", 0)

    singular = __singularize(parent_tag)

    for i, item in enumerate(items):
        singular = f"{singular}"
        if isinstance(item, dict):
            # Compound object → named subtree
            __append_value(parent, singular, item, {"index": index + i + 1})
        elif isinstance(item, list):
            # Nested list → treat as a compound child and recurse
            __append_value(parent, singular, item, {"index": index + i + 1})
        else:
            # Scalar → generic <item> tag as per spec
            child = ET.SubElement(parent, singular, {"index": str(index + i + 1)})
            if item is not None:
                child.text = __render_scalar(item)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def model_to_xml(
    model: BaseModel,
    *,
    root_tag: str | None = None,
) -> ET.Element:
    """Convert a Pydantic v2 ``BaseModel`` instance to an XML element tree.

    Parameters
    ----------
    model:
        Any Pydantic v2 model instance (including dynamically generated ones
        from ``datamodel-code-generator``).
    root_tag:
        Tag name for the XML root element.  Defaults to the model class name.

    Returns:
    -------
    xml.etree.ElementTree.Element
        Root element of the generated XML tree.  Pass it to
        ``ET.ElementTree(root).write(...)`` or :func:`model_to_xml_string`.
    """
    tag = root_tag or type(model).__name__
    root = ET.Element(tag)
    dumped = model.model_dump()
    if isinstance(dumped, list):
        __append_list(root, tag, dumped, {})
    else:
        __append_dict(root, dumped, {})
    return root


def dict_to_xml(
    data: dict[str, Any],
    *,
    root_tag: str = "root",
) -> ET.Element:
    """Convert a plain dictionary — such as the output of ``model.model_dump()`` — to an XML element tree.

    Useful when you already have the dictionary and don't need to hold the
    model instance around.

    Parameters
    ----------
    data:
        A (possibly nested) dict as produced by ``BaseModel.model_dump()``.
    root_tag:
        Tag name for the root element (default ``"root"``).

    Returns:
    -------
    xml.etree.ElementTree.Element
    """
    root = ET.Element(root_tag)
    __append_dict(root, data, {})
    return root


def model_to_xml_string(
    model: BaseModel,
    *,
    root_tag: str | None = None,
    pretty: bool = True,
    xml_declaration: bool = False,
) -> str:
    """Serialise a Pydantic v2 model instance to an XML string.

    Parameters
    ----------
    model:
        Any Pydantic v2 model instance.
    root_tag:
        Tag name for the root element.  Defaults to the model class name.
    pretty:
        Indent the output for human readability (default ``True``).
    xml_declaration:
        Prepend ``<?xml version='1.0' encoding='us-ascii'?>``
        (default ``False``).

    Returns:
    -------
    str
    """
    root = model_to_xml(model, root_tag=root_tag)
    if pretty:
        ET.indent(root)
    return ET.tostring(
        root,
        encoding="unicode",
        xml_declaration=xml_declaration,
    )
