import re
import json

from typing import Callable, Any
from .decoder import LogitsFn, pick_token, generate_choice, candidate_tokens
from .errors import DecodingError

NUMBER_PREFIX = re.compile(r"-?(\d+(\.\d*)?)?")
NUMBER_COMPLETE = re.compile(r"-?\d+(\.\d+)?")
MAX_NUMBER_TOKENS = 20
MAX_STRING_TOKENS = 100


def number_tokens(vocab: dict[int, str]) -> dict[int, str]:
    """Tokens made only of digits, '.' and '-' (computed once)."""
    return {
        token_id: text
        for token_id, text in vocab.items()
        if re.fullmatch(r"[0-9.\-]+", text)
    }


def generate_number(
    logits_fn: LogitsFn,
    input_ids: list[int],
    digit_tokens: dict[int, str],
    stop_id: int,
) -> float:
    """Generate one number under constraint.

    ``input_ids`` is extended in place with the generated digits.
    """
    text = ""
    for _ in range(MAX_NUMBER_TOKENS):
        allowed = {
            token_id
            for token_id, token_text in digit_tokens.items()
            if NUMBER_PREFIX.fullmatch(text + token_text)
        }
        if NUMBER_COMPLETE.fullmatch(text):
            allowed.add(stop_id)
        token_id = pick_token(logits_fn(input_ids), allowed)
        if token_id == stop_id:
            break
        text += digit_tokens[token_id]
        input_ids.append(token_id)
    if not NUMBER_COMPLETE.fullmatch(text):
        raise DecodingError(f"could not build a valid number: {text!r}")
    return float(text)


def string_tokens(vocab: dict[int, str]) -> dict[int, str]:
    """Tokens that can safely appear inside a JSON string."""
    result = {}

    for token_id, text in vocab.items():
        if not text or not text.isprintable():
            continue
        try:
            json.loads('"' + text + '"')
        except json.JSONDecodeError:
            continue
        result[token_id] = text

    return result


def first_string_tokens(str_tokens: dict[int, str]) -> dict[int, str]:
    """String tokens allowed as the first token of a value."""
    return {
        token_id: text
        for token_id, text in str_tokens.items()
        if not text.startswith(" ") or text.strip() == ""
    }


def generate_string(
    logits_fn: LogitsFn,
    input_ids: list[int],
    str_tokens: dict[int, str],
    end_ids: set[int],
    first_tokens: dict[int, str]
) -> str | Any:
    """Generate one escaped JSON string value."""
    text = ""
    first_allowed = set(first_tokens) | end_ids
    later_allowed = set(str_tokens) | end_ids

    for _ in range(MAX_STRING_TOKENS):
        allowed = first_allowed if text == "" else later_allowed
        token_id = pick_token(logits_fn(input_ids), allowed)

        if token_id in end_ids:
            break

        text += str_tokens[token_id]
        input_ids.append(token_id)

    try:
        return json.loads('"' + text + '"')
    except json.JSONDecodeError as exc:
        raise DecodingError(
            f"could not build a valid JSON string: {text!r}"
        ) from exc


EncodeFn = Callable[[str], list[int]]


def generate_boolean(
    logits_fn: LogitsFn,
    input_ids: list[int],
    vocab: dict[int, str],
    end_id: int,
    encode: EncodeFn,
) -> bool:
    """Generate true or false under constraint (extends input_ids)."""
    choices = ["true", "false"]
    candidates = candidate_tokens(vocab, choices)
    word = generate_choice(logits_fn, input_ids, candidates, choices, end_id)
    input_ids.extend(encode(word))
    return word == "true"
