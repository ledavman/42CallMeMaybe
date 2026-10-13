import json
from typing import Any, TypeVar

from pydantic import ValidationError, BaseModel

from .errors import InputError
from .models import FunctionDefinition, PromptItem

T = TypeVar("T", bound=BaseModel)


def _read_json(path: str) -> Any:
    """
    Read and parse a JSON file, raising InputError on any failure.
    """
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        raise InputError(f"File not found: '{path}'") from None
    except json.JSONDecodeError as e:
        raise InputError(f"File specified is not a valid JSON File '{path}': line: {e.lineno} col: {e.colno}. {e.msg}") from e
    except OSError as e:
        raise InputError(f"Can't open file '{path}'\n{e.strerror}") from e
    except UnicodeDecodeError as e:
        raise InputError(f"Invalid file: '{path}'\n{e}") from e


def _load_list(path: str, model: type[T]) -> list[T]:
    """Read a JSON array and validate each item against model."""
    raw = _read_json(path)
    if not isinstance(raw, list):
        raise InputError(f"Expected list got {type(raw).__name__}")
    if raw == []:
        raise InputError(f"Empty file given {path}")
    items: list[T] = []
    for index, item in enumerate(raw):
        try:
            items.append(model.model_validate(item))
        except ValidationError as e:
            raise InputError(f"in {path} Item with index {index} causing {e}") from e
    return items

def load_functions(path: str) -> list[FunctionDefinition]:
    """Load and validate the function definitions."""
    functions = _load_list(path, FunctionDefinition)
    return functions


def load_prompts(path: str) -> list[PromptItem]:
    """Load and validate the prompts."""
    prompts = _load_list(path, PromptItem)
    return prompts