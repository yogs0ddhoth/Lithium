"""Define custom DTO's for domain models."""

from typing import Any

from datamodel_code_generator import GenerateConfig
from datamodel_code_generator.dynamic import generate_dynamic_models
from datamodel_code_generator.enums import DataModelType
from datamodel_code_generator.format import Formatter
from pydantic import BaseModel

import xml_pydantic.schema as schema
import xml_pydantic.serializers as serializers


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


__all__ = ["serializers", "schema"]
