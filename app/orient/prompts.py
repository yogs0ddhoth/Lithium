"""Default prompts used by the agent."""

import xml.etree.ElementTree as ET

from app.models import define_model, model_to_xml_string, parse_xml_file
from app.utils import logger, normalize_whitespace

logger.info("# Loading Prompts...")


def load_xml_prompt(path: str) -> str:
    """Load a system prompt from a well formed xml file. All elements within the root will be serialized and concatenated into a single prompt string designed for performance with Claude models."""
    return " ".join(
        normalize_whitespace(ET.tostring(e, encoding="unicode"))
        for e in ET.parse(path).getroot()
    )


QA_SCHEMA = parse_xml_file("prompts/qa_instructions.xml")
_QAResults = define_model("QAResults", QA_SCHEMA)


class QAResults(_QAResults):
    """DTO for the <validated_qa_results />."""

    def model_dump_xml(self) -> str:
        """Serialize the model to an XML string."""
        return model_to_xml_string(self, root_tag="validated_qa_results")


PROBLEM_STATEMENT_SCHEMA = parse_xml_file("prompts/ps_instructions.xml")

_ProblemStatement = define_model("ProblemStatement", PROBLEM_STATEMENT_SCHEMA)


class ProblemStatement(_ProblemStatement):
    """DTO For the <problem_statement />."""

    def model_dump_xml(self) -> str:
        """Serialize the model to an XML string."""
        return model_to_xml_string(
            self,
            root_tag="problem_statement",
        )


REVIEWER_PROMPT = load_xml_prompt("prompts/qa_review.xml")

SYNTHESIS_PROMPT = load_xml_prompt("prompts/ps_synthesis.xml")

SYSTEM_PROMPT = load_xml_prompt("prompts/orient.xml")

logger.info("# Prompts loaded.")
