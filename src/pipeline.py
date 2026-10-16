from typing import Any

from pydantic import BaseModel

from .decoder import EncodeFn, LogitsFn, generate_choice
from .models import FunctionCall, FunctionDefinition, PromptItem
from .prompt import build_prompt
from .values import (
    generate_boolean, generate_number, generate_string,
)
from.errors import DecodingError


class Context(BaseModel):
    """Everything computed once and shared by every prompt."""

    logits_fn: LogitsFn
    encode: EncodeFn
    functions: list[FunctionDefinition]
    vocab: dict[int, str]
    name_candidates: dict[int, str]
    num_tokens: dict[int, str]
    str_tokens: dict[int, str]
    quote_id: int
    comma_id: int
    brace_id: int

def build_call(item: PromptItem, ctx: Context) -> FunctionCall:
    """Turn one prompt into a function call, token by token."""
    input_ids = ctx.encode(build_prompt(ctx.functions, item.prompt))

    names = [f.name for f in ctx.functions]
    name = generate_choice(
        ctx.logits_fn, input_ids, ctx.name_candidates, names, ctx.quote_id
    )
    input_ids.extend(ctx.encode(name))
    func = next(f for f in ctx.functions if f.name == name)

    input_ids.extend(ctx.encode('", "parameters": {'))
    params: dict[str, Any] = {}
    entries = list(func.parameters.items())
    closer = ""
    for index, (pname, spec) in enumerate(entries):
        end_id = ctx.brace_id if index == len(entries) - 1 else ctx.comma_id
        separator = ", " if index > 0 else ""
        opener = '"' if spec.type == "string" else ""
        input_ids.extend(ctx.encode(f'{closer}{separator}"{pname}": {opener}'))

        if spec.type in ("number", "integer"):
            number = generate_number(
                ctx.logits_fn, input_ids, ctx.num_tokens, end_id
            )
            params[pname] = int(number) if spec.type == "integer" else number
        elif spec.type == "string":
            params[pname] = generate_string(
                ctx.logits_fn, input_ids, ctx.str_tokens, ctx.quote_id
            )
        else:
            params[pname] = generate_boolean(
                ctx.logits_fn, input_ids, ctx.vocab, end_id, ctx.encode
            )
        closer = '"' if spec.type == "string" else ""

    return FunctionCall(prompt=item.prompt, name=name, parameters=params)


def find_token_id(vocab: dict[int, str], text: str) -> int:
    """Return the id of the token whose text is exactly ``text``."""
    result: int = ({txt: id for id, txt in vocab.items()}[text])
    if not result:
        raise DecodingError(f"'{text}' not found in specified vocab")
    return result


def make_context(model, functions) -> Context:
    