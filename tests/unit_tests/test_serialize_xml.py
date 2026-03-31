# ---------------------------------------------------------------------------
# Self-contained smoke test / usage example
# ---------------------------------------------------------------------------
from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Any

from pydantic import BaseModel as BM

from app.models.serialize_xml import dict_to_xml, model_to_xml_string
from app.utils import normalize_whitespace

# ---- Define a rich model that exercises every code path ----------------


class Coordinate(BM):
    lat: float
    lon: float


class Tag(BM):
    name: str
    weight: float


class Person(BM):
    id: int
    name: str
    active: bool
    nickname: str | None
    scores: list[int]
    aliases: list[str]
    coordinates: list[Coordinate]
    tags: list[Tag]
    metadata: dict[str, Any]


person = Person(
    id=42,
    name="Alice",
    active=True,
    nickname=None,
    scores=[10, 20, 30],
    aliases=["ally", "wonderland"],
    coordinates=[
        Coordinate(lat=51.5074, lon=-0.1278),
        Coordinate(lat=48.8566, lon=2.3522),
    ],
    tags=[
        Tag(name="admin", weight=1.0),
        Tag(name="user", weight=0.5),
    ],
    metadata={"source": "api", "version": 3},
)

# ---- Expected output ------------------------------------
expected = normalize_whitespace(
    """
<Person>
  <id>42</id>
  <name>Alice</name>
  <active>true</active>
  <nickname />
  <scores>
    <score index="1">10</score>
    <score index="2">20</score>
    <score index="3">30</score>
  </scores>
  <aliases>
    <alias index="1">ally</alias>
    <alias index="2">wonderland</alias>
  </aliases>
  <coordinates>
    <coordinate index="1">
      <lat>51.5074</lat>
      <lon>-0.1278</lon>
    </coordinate>
    <coordinate index="2">
      <lat>48.8566</lat>
      <lon>2.3522</lon>
    </coordinate>
  </coordinates>
  <tags>
    <tag index="1">
      <name>admin</name>
      <weight>1.0</weight>
    </tag>
    <tag index="2">
      <name>user</name>
      <weight>0.5</weight>
    </tag>
  </tags>
  <metadata>
    <source>api</source>
    <version>3</version>
  </metadata>
</Person>"""
)


# ---- model_to_xml path --------------------------------------------------
def test_model_to_xml_string():
    xml_str = model_to_xml_string(person)
    assert normalize_whitespace(xml_str) == expected


# ---- dict_to_xml path --------------------------------------------------
def test_dict_to_xml():
    root_elem = dict_to_xml(person.model_dump(), root_tag="Person")
    ET.indent(
        root_elem
    )  # model_to_xml_string does this by default: model_to_xml_string(...,prety=True)
    xml_str = ET.tostring(root_elem, encoding="unicode")

    assert normalize_whitespace(xml_str) == expected
