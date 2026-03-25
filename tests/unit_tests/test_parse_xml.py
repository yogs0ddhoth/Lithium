# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
import json as _json
import textwrap

from app.models.parse_xml import parse_xml_string


def _show(title: str, xml: str) -> None:
    rule = "─" * 64
    print(f"\n{rule}\n  {title}\n{rule}")
    schema = parse_xml_string(textwrap.dedent(xml).strip())
    print(_json.dumps(schema, indent=2))


# ── 1. Flat object ─────────────────────────────────────────────────────────
_show(
    "1. Flat object with scalar fields",
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

# ── 2. Nested object ───────────────────────────────────────────────────────
_show(
    "2. Nested object",
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

# ── 3. Array – inline items schema via data-items-* ────────────────────────
_show(
    "3. Array with inline data-items-* schema",
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

# ── 4. Array of objects ────────────────────────────────────────────────────
_show(
    "4. Array of objects",
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

# ── 5. Simple type with a rich XML subtree as description ──────────────────
_show(
    "5. Simple type – XML subtree serialised as description",
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

# ── 6. $ref + $defs ────────────────────────────────────────────────────────
_show(
    "6. $ref and $defs (definitions)",
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

# ── 7. allOf composition ───────────────────────────────────────────────────
_show(
    "7. allOf composition",
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

# ── 8. Nullable / multi-type field ─────────────────────────────────────────
_show(
    "8. Nullable field via multi-type array",
    """
        <Profile data-type="object">
          <bio data-type='["string", "null"]'
               data-description="Optional biography. Pass null to clear." />
          <website data-type="string" data-format="uri" />
        </Profile>
        """,
)

# ── 9. additionalProperties as a child subschema ───────────────────────────
_show(
    "9. additionalProperties as child element",
    """
        <DynamicConfig data-type="object">
          <name data-type="string" data-required="true" />
          <additionalProperties data-type="string" data-min-length="1" />
        </DynamicConfig>
        """,
)

# ── 10. Prefix-items tuple validation ──────────────────────────────────────
_show(
    "10. Tuple array via prefixItems",
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
