"""Parse XML files / strings / Elements into JSON Schema Python dicts.

Authoring convention
--------------------
Every XML element maps to one JSON Schema subschema.

data-* attributes
~~~~~~~~~~~~~~~~~
Attributes prefixed with ``data-`` carry JSON Schema keywords.  Names follow
*kebab-case* HTML style and are automatically converted to the *camelCase* or
``$``-prefixed form required by JSON Schema:

    data-type="string"            →  {"type": "string"}
    data-min-length="3"           →  {"minLength": 3}
    data-exclusive-maximum="100"  →  {"exclusiveMaximum": 100}
    data-ref="#/$defs/Address"    →  {"$ref": "#/$defs/Address"}
    data-id="https://…"           →  {"$id": "https://…"}
    data-required="true"          →  element is added to parent's required[]

Values are coerced to the most appropriate Python type:

    • ``enum``, ``const``, ``default``, ``examples``  → JSON-parsed
    • Numeric keywords (minimum, maxLength, …)         → int / float
    • Boolean keywords (uniqueItems, readOnly, …)      → bool
    • ``additionalProperties``                         → bool or JSON
    • ``type`` given as a JSON array                   → list[str]
    • Everything else                                  → str

Type dispatch
~~~~~~~~~~~~~
Simple types – ``string``, ``number``, ``integer``, ``boolean``, ``null``
    Schema keywords are read from ``data-*`` attributes.
    Unless ``data-description`` is already present, the element is stripped
    of its ``data-*`` attributes and serialised via ``ET.tostring()``; the
    result becomes the ``"description"`` value.

Object type – ``data-type="object"`` (or element has child elements and no
    explicit type)
    Each child element becomes a named property.  Children whose
    ``data-required="true"`` are collected into the ``"required"`` array.
    Children whose tag matches a structural JSON Schema keyword are treated
    as schema keywords rather than properties (see list below).

Array type – ``data-type="array"``
    • Single child element  →  ``{"items": <child schema>}``
    • Multiple children     →  ``{"prefixItems": [<schemas …>]}``
    • No children           →  ``data-items-*`` attributes are extracted as
                               an inline items schema (e.g.
                               ``data-items-type="string"``).
    When children are present, any ``data-items-*`` attributes are ignored.

Structural child tags (not treated as object properties)
    allOf · anyOf · oneOf          →  combiner arrays
    not · if · then · else         →  single subschemas
    additionalProperties           →  subschema (boolean via data-* on parent)
    propertyNames · contains
    unevaluatedProperties
    unevaluatedItems
    definitions · defs             →  ``$defs`` mapping

Limitations
-----------
* ``patternProperties`` — regex patterns are not valid XML element names, so
  this keyword cannot be expressed as child elements.  Use
  ``data-additional-properties="false"`` on the parent instead, or supply the
  full value as JSON via ``data-pattern-properties='{"^S_": {"type":"string"}}'``.
* ``items`` as a child of an *object* element is treated as a regular property
  named ``"items"``, not the array keyword.
* Schema element names that collide with a structural tag (e.g. a property
  genuinely named ``"not"``) cannot be expressed directly; use ``$ref`` to a
  ``$defs`` entry instead.
"""

from __future__ import annotations

import copy
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _normalize_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

__SIMPLE_TYPES: frozenset[str] = frozenset(
    {"string", "number", "integer", "boolean", "null"}
)

# Child tags that expand to JSON Schema combiner arrays.
__COMBINER_TAGS: frozenset[str] = frozenset({"allOf", "anyOf", "oneOf"})

# Child tags that expand to a single subschema keyword.
__UNARY_KEYWORD_TAGS: frozenset[str] = frozenset(
    {
        "not",
        "if",
        "then",
        "else",
        "additionalProperties",
        "propertyNames",
        "contains",
        "unevaluatedProperties",
        "unevaluatedItems",
    }
)

# Child tags that map to the ``$defs`` keyword.
__DEFS_TAGS: frozenset[str] = frozenset({"definitions", "defs"})

# All structural child tags (never become object properties).
__STRUCTURAL_CHILD_TAGS: frozenset[str] = (
    __COMBINER_TAGS | __UNARY_KEYWORD_TAGS | __DEFS_TAGS
)

# Suffixes (after "data-") that map to ``$``-prefixed JSON Schema keywords.
__DOLLAR_SUFFIXES: frozenset[str] = frozenset(
    {
        "ref",
        "id",
        "schema",
        "defs",
        "anchor",
        "dynamicRef",
        "dynamicAnchor",
        "comment",
    }
)

# Keywords whose values must be numeric.
__NUMERIC_KEYWORDS: frozenset[str] = frozenset(
    {
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
        "minLength",
        "maxLength",
        "minItems",
        "maxItems",
        "minContains",
        "maxContains",
        "minProperties",
        "maxProperties",
    }
)

# Keywords whose values must be boolean.
__BOOL_KEYWORDS: frozenset[str] = frozenset(
    {"uniqueItems", "readOnly", "writeOnly", "deprecated", "nullable"}
)

# Keywords whose values must be JSON-parsed.
__JSON_KEYWORDS: frozenset[str] = frozenset({"enum", "const", "default", "examples"})

# Suffix to exclude when building a field's own schema (handled at parent level).
__SKIP_REQUIRED: frozenset[str] = frozenset({"required"})


# ---------------------------------------------------------------------------
# Attribute name helpers
# ---------------------------------------------------------------------------


def __kebab_to_camel(s: str) -> str:
    """Convert a kebab-case string to camelCase.

    >>> _kebab_to_camel("min-length")
    'minLength'
    >>> _kebab_to_camel("exclusive-maximum")
    'exclusiveMaximum'
    >>> _kebab_to_camel("type")
    'type'
    """
    parts = s.split("-")
    return parts[0] + "".join(p.capitalize() for p in parts[1:])


def __suffix_to_keyword(suffix: str) -> str:
    """Map the portion of an attribute name after ``data-`` to a JSON Schema keyword.

    Dollar-prefixed keywords (ref, id, schema, …) receive a ``$`` prefix.
    All others undergo kebab-case → camelCase conversion.

    >>> _suffix_to_keyword("ref")
    '$ref'
    >>> _suffix_to_keyword("min-length")
    'minLength'
    >>> _suffix_to_keyword("type")
    'type'
    """
    if suffix in __DOLLAR_SUFFIXES:
        return f"${suffix}"
    return __kebab_to_camel(suffix)


# ---------------------------------------------------------------------------
# Attribute value coercion
# ---------------------------------------------------------------------------


def __coerce_value(keyword: str, raw: str) -> Any:
    """Coerce *raw* to the Python type appropriate for *keyword*.

    Coercion priority:
    1. ``type`` given as JSON array  →  list
    2. JSON keywords                 →  JSON-parsed value
    3. Numeric keywords              →  int / float
    4. Boolean keywords              →  bool
    5. ``additionalProperties``      →  bool, or JSON, or str
    6. Default                       →  str
    """
    # Multi-type: data-type='["string", "null"]'
    if keyword == "type" and raw.lstrip().startswith("["):
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return raw

    if keyword in __JSON_KEYWORDS:
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return raw

    if keyword in __NUMERIC_KEYWORDS:
        try:
            v = json.loads(raw)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                return v
        except (json.JSONDecodeError, ValueError):
            pass
        return raw

    if keyword in __BOOL_KEYWORDS:
        low = raw.lower()
        if low in ("true", "1", "yes"):
            return True
        if low in ("false", "0", "no"):
            return False
        return raw

    if keyword == "additionalProperties":
        low = raw.lower()
        if low == "true":
            return True
        if low == "false":
            return False
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return raw

    return raw


# ---------------------------------------------------------------------------
# Element inspection helpers
# ---------------------------------------------------------------------------


def __extract_data_attrs(element: ET.Element) -> dict[str, str]:
    """Return ``{suffix_after_data_: raw_value}`` for every ``data-*`` attribute."""
    return {
        attr[5:]: value
        for attr, value in element.attrib.items()
        if attr.startswith("data-")
    }


def __build_base_schema(
    data_attrs: dict[str, str],
    *,
    skip: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    """Convert ``{suffix: raw_value}`` pairs into a partial JSON Schema dict.

    Parameters
    ----------
    data_attrs:
        Mapping of (post-``data-``) suffix → raw attribute value.
    skip:
        Suffixes to exclude from output (e.g. ``{"required"}``).
    """
    schema: dict[str, Any] = {}
    for suffix, raw in data_attrs.items():
        if suffix in skip:
            continue
        kw = __suffix_to_keyword(suffix)
        schema[kw] = __coerce_value(kw, raw)
    return schema


def __strip_data_attrs(element: ET.Element) -> ET.Element:
    """Return a deep copy of *element* with every ``data-*`` attribute removed."""
    el = copy.deepcopy(element)
    for node in el.iter():
        for attr in [k for k in node.attrib if k.startswith("data-")]:
            del node.attrib[attr]
    return el


def __serialize_element(element: ET.Element) -> str:
    """Serialise *element* — stripped of ``data-*`` attributes — to a Unicode string."""
    return _normalize_whitespace(
        ET.tostring(__strip_data_attrs(element), encoding="unicode")
    )


def __dispatch_type(data_attrs: dict[str, str]) -> str:
    """Return the primary type string used for branching.

    When ``data-type`` holds a JSON array (e.g. ``'["string","null"]'``),
    the first element is returned so that structural dispatch still works.
    Returns ``""`` when no type attribute is present.
    """
    raw = data_attrs.get("type", "")
    if not raw:
        return ""
    coerced = __coerce_value("type", raw)
    if isinstance(coerced, list):
        return coerced[0] if coerced else ""
    return str(coerced)


# ---------------------------------------------------------------------------
# Core recursive conversion
# ---------------------------------------------------------------------------


def __element_to_schema(element: ET.Element) -> dict[str, Any]:
    """Recursively convert *element* to a JSON Schema dict."""
    data_attrs = __extract_data_attrs(element)
    dtype = __dispatch_type(data_attrs)

    if dtype in __SIMPLE_TYPES:
        return __simple_to_schema(element, data_attrs)

    if dtype == "array":
        return __array_to_schema(element, data_attrs)

    # Explicit object OR inferred from having child elements.
    if dtype == "object" or (not dtype and len(element) > 0):
        return __object_to_schema(element, data_attrs)

    # Untyped leaf element (no data-type, no children).
    return __simple_to_schema(element, data_attrs)


def __simple_to_schema(
    element: ET.Element, data_attrs: dict[str, str]
) -> dict[str, Any]:
    """Build a schema for a simple / terminal type.

    The element (sans ``data-*`` attributes) and its entire subtree are
    serialised via ``ET.tostring()`` and stored as ``"description"`` unless:

    * ``data-description`` is already present in *data_attrs*, or
    * the schema already contains ``$ref`` (reference schemas carry no
      auto-description).
    """
    schema = __build_base_schema(data_attrs, skip=__SKIP_REQUIRED)
    if "$ref" not in schema and "description" not in schema:
        schema["description"] = __serialize_element(element)
    return schema


def __object_to_schema(
    element: ET.Element, data_attrs: dict[str, str]
) -> dict[str, Any]:
    """Build an ``{"type": "object", …}`` schema from *element* and its children."""
    schema = __build_base_schema(data_attrs, skip=__SKIP_REQUIRED)
    schema.setdefault("type", "object")

    properties: dict[str, Any] = {}
    required: list[str] = []

    for child in element:
        tag = child.tag

        # ── $defs / definitions ─────────────────────────────────────────────
        if tag in __DEFS_TAGS:
            schema["$defs"] = {gc.tag: __element_to_schema(gc) for gc in child}
            continue

        # ── Combiner keywords (allOf / anyOf / oneOf) ───────────────────────
        # Each grandchild of the combiner element is one subschema.
        if tag in __COMBINER_TAGS:
            schema[tag] = [__element_to_schema(gc) for gc in child]
            continue

        # ── Unary structural keywords ────────────────────────────────────────
        # The child element itself IS the subschema for the keyword.
        if tag in __UNARY_KEYWORD_TAGS:
            schema[tag] = __element_to_schema(child)
            continue

        # ── Regular property ─────────────────────────────────────────────────
        child_data = __extract_data_attrs(child)
        if child_data.get("required", "").lower() in ("true", "1", "yes"):
            required.append(tag)
        properties[tag] = __element_to_schema(child)

    if properties:
        schema["properties"] = properties
    if required:
        schema["required"] = required

    return schema


def __array_to_schema(
    element: ET.Element, data_attrs: dict[str, str]
) -> dict[str, Any]:
    """Build a ``{"type": "array", …}`` schema from *element* and its children.

    Child elements take precedence over ``data-items-*`` inline attributes.
    """
    schema = __build_base_schema(data_attrs, skip=__SKIP_REQUIRED)
    schema.setdefault("type", "array")

    children = list(element)

    if not children:
        # No child elements: derive items from data-items-* attributes.
        items_inline = __items_from_inline_attrs(data_attrs)
        if items_inline:
            schema["items"] = items_inline
        return schema

    if len(children) == 1:
        schema["items"] = __element_to_schema(children[0])
    else:
        # Multiple children → tuple validation via prefixItems.
        schema["prefixItems"] = [__element_to_schema(c) for c in children]

    return schema


def __items_from_inline_attrs(data_attrs: dict[str, str]) -> dict[str, Any] | None:
    """Extract ``data-items-*`` attributes into an inline items schema.

    For example::

        data-items-type="string"
        data-items-format="uuid"

    produces ``{"type": "string", "format": "uuid"}``.

    Returns ``None`` when no matching attributes are present.
    """
    items: dict[str, Any] = {}
    prefix = "items-"
    for suffix, raw in data_attrs.items():
        if suffix.startswith(prefix):
            sub = suffix[len(prefix) :]
            kw = __suffix_to_keyword(sub)
            items[kw] = __coerce_value(kw, raw)
    return items or None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def from_element(element: ET.Element) -> dict[str, Any]:
    """Convert an ``xml.etree.ElementTree.Element`` to a JSON Schema dict.

    Parameters
    ----------
    element:
        Root element of an XML tree (e.g. from ``ET.fromstring()`` or
        ``ET.parse().getroot()``).

    Returns:
    -------
    dict[str, Any]
        JSON Schema–compatible Python dict ready for ``json.dumps()``,
        Pydantic's ``model_validate()``, or any JSON Schema validator.
    """
    return __element_to_schema(element)


def from_string(xml_string: str) -> dict[str, Any]:
    """Parse a well-formed XML string into a JSON Schema dict.

    Parameters
    ----------
        xml_string:
            UTF-8 ``str`` (or ``bytes``) containing well-formed XML.

    Returns:
    -------
        dict[str, Any]
    """
    return __element_to_schema(ET.fromstring(xml_string))


def from_file(path: str | Path) -> dict[str, Any]:
    """Parse an XML file into a JSON Schema dict.

    Parameters
    ----------
        path:
            Path to a well-formed XML file.

    Returns:
    -------
        dict[str, Any]
    """
    return __element_to_schema(ET.parse(str(path)).getroot())
