import textwrap
from typing import Any

from deepdiff import DeepDiff
from xml_pydantic.schema import from_string


def parse_schema(xml: str) -> dict[str, Any]:
    """Parse dedented XML into a JSON Schema dict."""
    return from_string(textwrap.dedent(xml).strip())


def test_parse_flat_object() -> None:
    """Flat object with scalar fields."""
    expected = {
        "type": "object",
        "title": "Person",
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "properties": {
            "name": {
                "type": "string",
                "description": "<name> Full <em>legal</em> name of the person. </name>",
            },
            "age": {
                "type": "integer",
                "minimum": 0,
                "maximum": 150,
                "description": "<age />",
            },
            "score": {
                "type": "number",
                "exclusiveMinimum": 0.0,
                "description": "<score />",
            },
            "active": {"type": "boolean", "default": True, "description": "<active />"},
        },
        "required": ["name"],
    }
    actual = parse_schema(
        """
          <Person
              data-type="object"
              data-title="Person"
              data-schema="https://json-schema.org/draft/2020-12/schema">
            <name data-type="string" data-required="true">
              Full <em>legal</em> name of the person.
            </name>
            <age  data-type="integer"
                  data-minimum="0"
                  data-maximum="150" />
            <score data-type="number" data-exclusive-minimum="0.0" />
            <active data-type="boolean" data-default="true" />
          </Person>
          """,
    )
    assert DeepDiff(expected, actual) == {}


def test_parse_nested_object() -> None:
    """Nested object."""
    expected = {
        "type": "object",
        "properties": {
            "id": {"type": "string", "format": "uuid", "description": "<id />"},
            "address": {
                "type": "object",
                "properties": {
                    "street": {"type": "string", "description": "<street />"},
                    "city": {"type": "string", "description": "<city />"},
                    "postcode": {
                        "type": "string",
                        "pattern": "[0-9]{5}",
                        "description": "<postcode />",
                    },
                },
            },
        },
        "required": ["id", "address"],
    }
    actual = parse_schema(
        """
          <Employee data-type="object">
            <id      data-type="string" data-format="uuid" data-required="true" />
            <address data-type="object" data-required="true">
              <street   data-type="string" />
              <city     data-type="string" />
              <postcode data-type="string" data-pattern="[0-9]{5}" />
            </address>
          </Employee>
          """,
    )
    assert DeepDiff(expected, actual) == {}


def test_parse_simple_array() -> None:
    """Array with inline data-items-* schema (scalar values)."""
    expected = {
        "type": "object",
        "properties": {
            "tags": {
                "type": "array",
                "itemsType": "string",
                "minItems": 1,
                "uniqueItems": True,
                "items": {"type": "string"},
            }
        },
    }

    actual = parse_schema(
        """
          <Config data-type="object">
            <tags
              data-type="array"
              data-items-type="string"
              data-min-items="1"
              data-unique-items="true" />
          </Config>
          """,
    )
    assert DeepDiff(expected, actual) == {}


def test_parse_object_array() -> None:
    """Array of objects."""
    expected = {
        "type": "object",
        "properties": {
            "lines": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "product": {"type": "string", "description": "<product />"},
                        "qty": {
                            "type": "integer",
                            "minimum": 1,
                            "description": "<qty />",
                        },
                        "unit-price": {
                            "type": "number",
                            "minimum": 0,
                            "description": "<unit-price />",
                        },
                    },
                    "required": ["product", "qty"],
                },
            }
        },
    }
    actual = parse_schema(
        """
          <Order data-type="object">
            <lines data-type="array">
              <line data-type="object">
                <product    data-type="string"  data-required="true" />
                <qty        data-type="integer" data-minimum="1" data-required="true" />
                <unit-price data-type="number"  data-minimum="0" />
              </line>
            </lines>
          </Order>
          """,
    )
    assert DeepDiff(expected, actual) == {}


def test_parse_handle_subtrees() -> None:
    """Simple type element's XML subtree serialised as description."""
    expected = {
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "<title />"},
            "body": {
                "type": "string",
                "description": "<body> <p>Main body in <strong>HTML</strong> format.</p> <ul> <li>Maximum 10,000 characters.</li> <li>Permitted inline tags: <code>em</code>, <code>strong</code>.</li> </ul> </body>",
            },
        },
        "required": ["title"],
    }
    actual = parse_schema(
        """
          <Article data-type="object">
            <title data-type="string" data-required="true" />
            <body  data-type="string">
              <p>Main body in <strong>HTML</strong> format.</p>
              <ul>
                <li>Maximum 10,000 characters.</li>
                <li>Permitted inline tags: <code>em</code>, <code>strong</code>.</li>
              </ul>
            </body>
          </Article>
          """,
    )
    assert DeepDiff(expected, actual) == {}


def test_parse_json_refs() -> None:
    """JSON $ref and $defs (definitions)."""
    expected = {
        "type": "object",
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$defs": {
            "Money": {
                "type": "object",
                "properties": {
                    "amount": {
                        "type": "number",
                        "minimum": 0,
                        "description": "<amount />",
                    },
                    "currency": {
                        "type": "string",
                        "pattern": "[A-Z]{3}",
                        "description": "<currency />",
                    },
                },
                "required": ["amount", "currency"],
            }
        },
        "properties": {
            "id": {"type": "string", "format": "uuid", "description": "<id />"},
            "total": {"$ref": "#/$defs/Money"},
            "discount": {"$ref": "#/$defs/Money"},
        },
        "required": ["total"],
    }
    actual = parse_schema(
        """
          <Invoice data-type="object"
                  data-schema="https://json-schema.org/draft/2020-12/schema">
            <definitions>
              <Money data-type="object">
                <amount   data-type="number" data-minimum="0"           data-required="true" />
                <currency data-type="string" data-pattern="[A-Z]{3}"    data-required="true" />
              </Money>
            </definitions>
            <id       data-type="string" data-format="uuid" />
            <total    data-ref="#/$defs/Money" data-required="true" />
            <discount data-ref="#/$defs/Money" />
          </Invoice>
          """,
    )
    assert DeepDiff(expected, actual) == {}


def test_parse_all_of() -> None:
    """JSON allOf composition."""
    expected = {
        "type": "object",
        "allOf": [
            {"$ref": "#/$defs/User"},
            {
                "type": "object",
                "properties": {
                    "role": {
                        "type": "string",
                        "enum": ["admin", "superadmin"],
                        "description": "<role />",
                    },
                    "permissions": {
                        "type": "array",
                        "items": {"type": "string", "description": "<permission />"},
                    },
                },
                "required": ["role"],
            },
        ],
    }

    actual = parse_schema(
        """
          <AdminUser data-type="object">
            <allOf>
              <base data-ref="#/$defs/User" />
              <extra data-type="object">
                <role data-type="string"
                      data-enum='["admin", "superadmin"]'
                      data-required="true" />
                <permissions data-type="array">
                  <permission data-type="string" />
                </permissions>
              </extra>
            </allOf>
          </AdminUser>
          """,
    )
    assert DeepDiff(expected, actual) == {}


def test_parse_json_nullables() -> None:
    """Nullable field via multi-type array."""
    expected = {
        "type": "object",
        "properties": {
            "bio": {
                "type": ["string", "null"],
                "description": "<bio> Optional biography. Pass null to clear. </bio>",
            },
            "website": {
                "type": "string",
                "format": "uri",
                "description": "<website />",
            },
        },
    }
    actual = parse_schema(
        """
          <Profile data-type="object">
            <bio data-type='["string", "null"]'>
              Optional biography. Pass null to clear.
            </bio>
            <website data-type="string" data-format="uri" />
          </Profile>
          """,
    )
    assert DeepDiff(expected, actual) == {}


def test_parse_additional_properties() -> None:
    """additionalProperties as a child subschema."""
    expected = {
        "type": "object",
        "additionalProperties": {
            "type": "string",
            "minLength": 1,
            "description": "<additionalProperties />",
        },
        "properties": {"name": {"type": "string", "description": "<name />"}},
        "required": ["name"],
    }
    actual = parse_schema(
        """
          <DynamicConfig data-type="object">
            <name data-type="string" data-required="true" />
            <additionalProperties data-type="string" data-min-length="1" />
          </DynamicConfig>
          """,
    )
    assert DeepDiff(expected, actual) == {}


def test_parse_tuples() -> None:
    """Tuple array via prefixItems."""
    expected = {
        "type": "object",
        "properties": {
            "coordinates": {
                "type": "array",
                "minItems": 2,
                "maxItems": 2,
                "prefixItems": [
                    {
                        "type": "number",
                        "minimum": -180,
                        "maximum": 180,
                        "description": "<longitude> Longitude in decimal degrees. </longitude>",
                    },
                    {
                        "type": "number",
                        "minimum": -90,
                        "maximum": 90,
                        "description": "<latitude> Latitude in decimal degrees. </latitude>",
                    },
                ],
            }
        },
    }
    actual = parse_schema(
        """
          <GeoPoint data-type="object">
            <coordinates data-type="array" data-min-items="2" data-max-items="2">
              <longitude data-type="number" data-minimum="-180" data-maximum="180">
                Longitude in decimal degrees.
              </longitude>
              <latitude  data-type="number" data-minimum="-90"  data-maximum="90">
                Latitude in decimal degrees.
              </latitude>
            </coordinates>
          </GeoPoint>
          """,
    )
    assert DeepDiff(expected, actual) == {}
