"""Default prompts used by the agent."""

import xml.etree.ElementTree as ET
from typing import Annotated, Any, List

from pydantic import BaseModel

from app.orient.models import define_model
from app.utils import logger, normalize_whitespace

logger.info("# Loading Prompts...")


def load_xml_prompt(path: str) -> str:
    """Load a system prompt from a well formed xml file. All elements within the root will be serialized and concatenated into a single prompt string designed for performance with Claude models."""
    return " ".join(
        normalize_whitespace(ET.tostring(e, encoding="unicode"))
        for e in ET.parse(path).getroot()
    )


def get_qa_schema(path: str) -> dict[str, Any]:
    """Parse the QA Review from its XML instructions."""
    instructions = ET.parse(path).getroot()

    properties = {}

    for e in instructions:
        if e.tag == "qa":
            qa_props = {}
            for prop in e:
                if prop.tag == "question" or prop.tag == "answer":
                    qa_props[prop.tag] = {
                        "type": "string",
                        "description": normalize_whitespace(
                            ET.tostring(prop, encoding="unicode")
                        ),
                    }
                else:
                    qa_props[prop.tag] = {
                        "type": "array",
                        "description": normalize_whitespace(
                            ET.tostring(prop, encoding="unicode")
                        ),
                        "items": {"type": "string"},
                    }

            properties["results"] = {
                "type": "array",
                "description": "",  # TODO
                "items": {
                    "type": "object",
                    "description": "DTO for Question and Answer.",
                    "properties": qa_props,
                    "required": list(qa_props),
                },
            }
        else:
            properties[e.tag] = normalize_whitespace(ET.tostring(e, encoding="unicode"))

    return {
        "type": "object",
        "description": "DTO for the <validated_qa_results />",
        "properties": properties,
        "required": list(properties),
    }


_QAResults = define_model("QAResults", get_qa_schema("prompts/qa_instructions.xml"))

# TODO: get a better way - look for libraries


def serialize_node(root: ET.Element, val, tag=""):

    match val:
        case val if issubclass(val, BaseModel):
            node = ET.SubElement(root, tag)
            serialize_dict_node(node, val.__getattribute__("model_dump")())
        case [*items]:
            node = ET.SubElement(
                root,
            )
            serialize_list_node(node, items)
        case _:
            root.text = str(val)


def serialize_dict_node(root, dict):
    for key, val in dict.items():
        node = ET.SubElement(root, key)
        serialize_node(node, val)


def serialize_list_node(root: ET.Element, elems: list):

    for i, val in enumerate(elems):
        node = ET.SubElement(root, f"{root.tag}-item-{i}")
        serialize_node(node, val)


class QAResults(_QAResults):
    def model_dump_xml(self) -> str:
        root = ET.Element("validated_qa_results")

        for key, val in self.model_dump().items():
            match key, val:
                case "results", [*val]:
                    qa = ET.SubElement(root, "qa_results")
                    for i, qa_res in enumerate(val):
                        qai = ET.SubElement(qa, f"qa_{i + 1}")
                        for tag, qa_item in qa_res.model_dump().items():
                            if isinstance(qa_item, list):
                                e = ET.SubElement(qai, tag)
                                for j, txt in enumerate(qa_item):
                                    f = ET.SubElement(e, f"{tag}_{j + 1}")
                                    f.text = str(txt)
                            else:
                                e = ET.SubElement(qai, tag)
                                e.text = str(qa_item)
                case _:
                    e = ET.SubElement(root, key)
                    e.text = str(val)


def get_ps_schema(path: str) -> dict[str, Any]:
    """Parse the Problem Statement Schema from its XML instructions."""
    instructions = ET.parse(path).getroot()
    properties = {
        e.tag: {"type": "string", "description": ET.tostring(e, encoding="unicode")}
        for e in instructions
    }
    return {
        "type": "object",
        "description": "DTO For the <problem_statement />.",
        "properties": properties,
        "required": list(properties),
    }


_ProblemStatement = define_model(
    "ProblemStatement", get_ps_schema("prompts/ps_instructions.xml")
)


class ProblemStatement(_ProblemStatement):
    """DTO For the <problem_statement />."""

    def model_dump_xml(self) -> str:
        """Serialize the model to an XML string."""
        root = ET.Element("problem_statement")

        for tag, txt in self.model_dump().items():
            # preserves inner '<','>' characters inside txt string which would otherwise be sanitized by the xml parser
            c = ET.fromstring(f"<{tag}>{txt}</{tag}>")
            root.append(c)

        return ET.tostring(root, encoding="unicode")


REVIEWER_PROMPT = load_xml_prompt("prompts/qa_review.xml")

SYNTHESIS_PROMPT = load_xml_prompt("prompts/ps_synthesis.xml")

SYSTEM_PROMPT = load_xml_prompt("prompts/orient.xml")

logger.info("# Prompts loaded.")
