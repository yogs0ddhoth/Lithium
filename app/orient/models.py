"""Define custom DTO's for domain models."""

import xml.etree.ElementTree as ET
from typing import Annotated, Any

from datamodel_code_generator import GenerateConfig
from datamodel_code_generator.dynamic import generate_dynamic_models
from datamodel_code_generator.enums import DataModelType
from datamodel_code_generator.format import Formatter
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


def define_model(
    name: str,
    schema: dict[str, Any],
    output_model_type=DataModelType.PydanticV2BaseModel,
) -> type[BaseModel]:
    """Dynamically define a Model with data validation.

    Args:
        name (str): the Model Class Name.
        schema (dict[str, Any]): JSON Schema or OpenAPI schema as dict
        output_model_type (datamodel_code_generator.enums.DataModelType): [See docs for supported output types](https://datamodel-code-generator.koxudaxi.dev/output-model-types/#output-model-types)

    Returns:
        A new pydantic.v2.BaseModel derived Class
    [See Docs](https://datamodel-code-generator.koxudaxi.dev/dynamic-model-generation/?h=dynam#dynamic-model-generation)
    """
    config = GenerateConfig(
        class_name=name,
        output_model_type=output_model_type,
        formatters=[Formatter.RUFF_FORMAT, Formatter.RUFF_CHECK],
    )

    match generate_dynamic_models(schema, config=config)[name]:
        case model if issubclass(model, BaseModel):
            return model
        case not_model:
            raise ValueError(f"Expected a subclass of {BaseModel}, Got: {not_model}")


class QA(TypedDict):
    """DTO for Question and Answer."""

    question: Annotated[str, Field(description="restate the <question /> verbatim")]
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
