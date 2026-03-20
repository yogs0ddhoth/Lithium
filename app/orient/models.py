"""Define custom DTO's for domain models."""

import xml.etree.ElementTree as ET
from typing import Annotated

from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from app.orient.prompts import instructions


class ProblemStatement(TypedDict):
    """DTO for the Problem Statement Map."""

    executive_summary: Annotated[str, Field(description=instructions.executive_summary)]
    background: Annotated[str, Field(description=instructions.background)]
    users_and_stakeholders: Annotated[
        str, Field(description=instructions.users_and_stakeholders)
    ]
    problem_and_need: Annotated[str, Field(description=instructions.problem_and_need)]
    evidence: Annotated[str, Field(description=instructions.evidence)]
    constraints: Annotated[str, Field(description=instructions.constraints)]
    success_criteria_and_metrics: Annotated[
        str, Field(description=instructions.success_criteria_and_metrics)
    ]
    assumptions: Annotated[str, Field(description=instructions.assumptions)]
    principles_and_values: Annotated[
        str, Field(description=instructions.principles_and_values)
    ]
    risks_and_gaps: Annotated[str, Field(description=instructions.risks_and_gaps)]
    open_questions: Annotated[str, Field(description=instructions.open_questions)]
    next_steps: Annotated[str, Field(description=instructions.next_steps)]


class SynthesisResults(BaseModel):
    """DTO for a <problem_statement />."""

    problem_statement: Annotated[
        ProblemStatement,
        Field(
            description="<problem_statement>DTO for the Problem Statement Map.</problem_statement>"
        ),
    ]

    def model_dump_xml(self) -> str:
        """Serialize the model to an XML string."""
        root = ET.Element("problem_statement")

        for tag, txt in self.problem_statement.items():
            e = ET.SubElement(root, tag)
            e.text = str(txt)

        return ET.tostring(root, encoding="unicode")


class QA(TypedDict):
    """DTO for Question and Answer."""

    question: Annotated[str, Field(description="restate the <question> verbatim")]
    answer: Annotated[
        str,
        Field(
            description="<response_if_sufficient_information_exists />"
            " OR "
            "<response_if_insufficient_information_to_support_an_answer>NOT ANSWERABLE FROM SUMMARY</response_if_insufficient_information_to_support_an_answer>"
        ),
    ]
    evidence: list[
        Annotated[
            str,
            Field(
                description="cited section of the summary that supports the <answer>"
            ),
        ]
    ]
    missing_information: list[
        Annotated[
            str,
            Field(description="information needed for a to support a complete answer"),
        ]
    ]


class QASummary(TypedDict):
    """DTO for Q&A Coverage."""

    answered_count: Annotated[
        int,
        Field(
            description="The number of <questions /> that were answered with: "
            "<response_if_sufficient_information_exists />, including partial answers."
        ),
    ]
    not_answerable_count: Annotated[
        int,
        Field(
            description="The number of <questions /> that were answered with: "
            "<response_if_insufficient_information_to_support_an_answer>NOT ANSWERABLE FROM SUMMARY</response_if_insufficient_information_to_support_an_answer>"
        ),
    ]
    gap_themes: Annotated[
        str,
        Field(
            description="themes and patterns identifiable from all the <missing_information />"
        ),
    ]


class QAResults(BaseModel):
    """DTO for model response."""

    qa: Annotated[
        list[QA],
        Field(default_factory=list, description="The list of answered questions"),
    ]
    summary: Annotated[
        QASummary, Field(..., description="A summary of your responses.")
    ]

    def model_dump_xml(self) -> str:
        """Serialize the model to an XML string."""
        root = ET.Element("validated_qa_results")

        for tag, txt in self.summary.items():
            e = ET.SubElement(root, tag)
            e.text = str(txt)

        qa = ET.SubElement(root, "qa")

        for i, qa_item in enumerate(self.qa):
            qai = ET.SubElement(qa, f"QA_{i + 1}")
            for tag, item in qa_item.items():
                match item:
                    case [*items]:
                        list = ET.SubElement(qai, tag)
                        for j, item in enumerate(items):
                            e = ET.SubElement(list, f"{tag}_{j + 1}")
                            e.text = str(item)
                    case item if isinstance(item, str):
                        e = ET.SubElement(qai, tag)
                        e.text = str(item)
        return ET.tostring(root, encoding="unicode")
