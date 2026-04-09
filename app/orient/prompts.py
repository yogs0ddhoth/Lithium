"""Default prompts used by the agent."""

import xml_pydantic

from app.utils import XmlDto, load_xml_prompt, logger

logger.info("# Loading Prompts...")

REVIEWER_PROMPT = load_xml_prompt("prompts/orient/qa_review.xml")

QA_SCHEMA = xml_pydantic.schema.from_file("prompts/orient/qa_results.schema.xml")


class QAResults(xml_pydantic.define_model("QAResults", QA_SCHEMA), XmlDto):
    """DTO for the `<validated_qa_results />`."""

    _root_tag = "validated_qa_results"


SYNTHESIS_PROMPT = load_xml_prompt("prompts/orient/problem_statement_synthesis.xml")

PROBLEM_STATEMENT_SCHEMA = xml_pydantic.schema.from_file(
    "prompts/orient/problem_statement.schema.xml"
)


class ProblemStatement(
    xml_pydantic.define_model("ProblemStatement", PROBLEM_STATEMENT_SCHEMA), XmlDto
):
    """DTO for the `<problem_statement />`."""

    _root_tag = "problem_statement"


AGENT_PROMPT = load_xml_prompt("prompts/orient/system.xml")

logger.info("# Prompts loaded.")
