import re

from .decoder import LogitsFn, pick_token
from .errors import DecodingError

NUMBER_PREFIX = re.compile(r"-?(\d+(\.\d*)?)?")
NUMBER_COMPLETE = re.compile(r"-?\d+(\.\d+)?")
MAX_NUMBER_TOKENS = 20


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