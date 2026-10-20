from typing import Any

from pydantic import BaseModel

from .decoder import LogitsFn, generate_choice, candidate_tokens
from .models import FunctionCall, FunctionDefinition, PromptItem
from .prompt import build_prompt
from .values import (
    generate_boolean, generate_number, generate_string, EncodeFn, number_tokens, string_tokens, first_string_tokens
)
from.errors import DecodingError
from llm_sdk import Small_LLM_Model
from .vocab import load_vocab


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
    end_ids: set[int]
    first_tokens: dict[int, str]

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
                ctx.logits_fn, input_ids, ctx.str_tokens, ctx.end_ids, ctx.first_tokens,
            )
        else:
            params[pname] = generate_boolean(
                ctx.logits_fn, input_ids, ctx.vocab, end_id, ctx.encode
            )
        closer = '"' if spec.type == "string" else ""

    return FunctionCall(prompt=item.prompt, name=name, parameters=params)


def find_token_id(vocab: dict[int, str], text: str) -> int:
    """Return the id of the token whose text is exactly ``text``."""
    for token_id, token_text in vocab.items():
        if token_text == text:
            return token_id
    raise DecodingError(f"'{text}' not found in specified vocab")


def string_end_tokens(vocab: dict[int, str]) -> set[int]:
    """Ids of the tokens whose text starts with a double quote."""
    quoted_ids = set()
    for id, text in vocab.items():
        if text.startswith('"'):
            quoted_ids.add(id)
    return quoted_ids


def make_context(
    model: Small_LLM_Model, functions: list[FunctionDefinition]
) -> Context:
    """Precompute everything that does not depend on the prompt."""
    vocab = load_vocab(model.get_path_to_vocab_file())
    names = [f.name for f in functions]

    def encode(text: str) -> list[int]:
        ids: list[int] = model.encode(text).tolist()[0]
        return ids

    return Context(
        logits_fn=model.get_logits_from_input_ids,
        encode=encode,
        functions=functions,
        vocab=vocab,
        name_candidates=candidate_tokens(vocab, names),
        num_tokens=number_tokens(vocab),
        str_tokens=string_tokens(vocab),
        quote_id=find_token_id(vocab, '"'),
        comma_id=find_token_id(vocab, ","),
        brace_id=find_token_id(vocab, "}"),
        end_ids=string_end_tokens(vocab),
        first_tokens=first_string_tokens(str_tokens)
    )
    
