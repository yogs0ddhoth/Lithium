# ---------------------------------------------------------------------------
# Self-contained smoke test / usage example
# ---------------------------------------------------------------------------
from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Any

from pydantic import BaseModel as BM

from app.models.serialize_xml import dict_to_xml, model_to_xml_string

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

xml_str = model_to_xml_string(person)
print("------------- From BaseModel -------------")
print(xml_str)

# ---- Expected output (abbreviated) ------------------------------------
# <Person>
#   <id>42</id>
#   <name>Alice</name>
#   <active>true</active>
#   <nickname />
#   <scores>
#     <item>10</item>
#     <item>20</item>
#     <item>30</item>
#   </scores>
#   <aliases>
#     <item>ally</item>
#     <item>wonderland</item>
#   </aliases>
#   <coordinates>
#     <coordinate>
#       <lat>51.5074</lat>
#       <lon>-0.1278</lon>
#     </coordinate>
#     <coordinate>
#       <lat>48.8566</lat>
#       <lon>2.3522</lon>
#     </coordinate>
#   </coordinates>
#   <tags>
#     <tag>
#       <name>admin</name>
#       <weight>1.0</weight>
#     </tag>
#     <tag>
#       <name>user</name>
#       <weight>0.5</weight>
#     </tag>
#   </tags>
#   <metadata>
#     <source>api</source>
#     <version>3</version>
#   </metadata>
# </Person>

# ---- dict_to_xml path --------------------------------------------------
print("------------- From Dict -------------")
root_elem = dict_to_xml(person.model_dump(), root_tag="PersonRecord")
ET.indent(root_elem)
print(ET.tostring(root_elem, encoding="unicode"))
ET.indent(root_elem)
print(ET.tostring(root_elem, encoding="unicode"))
