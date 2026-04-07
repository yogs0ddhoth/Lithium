"""Default prompts used by the agent."""

import xml_pydantic

from app.utils import load_xml_prompt, logger

logger.info("# Loading Prompts...")


QA_SCHEMA = xml_pydantic.schema.from_file("prompts/qa_results.schema.xml")
_QAResults = xml_pydantic.define_model("QAResults", QA_SCHEMA)


class QAResults(_QAResults):
    """DTO for the `<validated_qa_results />`."""

    def model_dump_xml(self) -> str:
        """Serialize the model to an XML string."""
        return xml_pydantic.serializers.model_to_xml_string(
            self, root_tag="validated_qa_results"
        )


PROBLEM_STATEMENT_SCHEMA = xml_pydantic.schema.from_file(
    "prompts/problem_statement.schema.xml"
)

_ProblemStatement = xml_pydantic.define_model(
    "ProblemStatement", PROBLEM_STATEMENT_SCHEMA
)


class ProblemStatement(_ProblemStatement):
    """DTO For the `<problem_statement />`."""

    def model_dump_xml(self) -> str:
        """Serialize the model to an XML string."""
        return xml_pydantic.serializers.model_to_xml_string(
            self,
            root_tag="problem_statement",
        )


REVIEWER_PROMPT = load_xml_prompt("prompts/qa_review.xml")

SYNTHESIS_PROMPT = load_xml_prompt("prompts/problem_statement_synthesis.xml")

AGENT_PROMPT = load_xml_prompt("prompts/orient.xml")

logger.info("# Prompts loaded.")
