from typing import Any, Literal

from pydantic import BaseModel


class ParameterSpec(BaseModel):
    """Type of a function parameter or of a return value."""

    type: Literal["number", "string", "boolean", "integer"]


class FunctionDefinition(BaseModel):
    """A function the LLM is allowed to choose, as described in the input."""

    name: str
    description: str
    parameters: dict[str, ParameterSpec]
    returns: ParameterSpec


class PromptItem(BaseModel):
    """One natural-language request to translate into a function call."""

    prompt: str


class FunctionCall(BaseModel):
    """One entry of the output file."""

    prompt: str
    name: str
    parameters: dict[str, Any]
