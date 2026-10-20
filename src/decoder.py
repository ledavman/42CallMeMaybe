import numpy as np
from .errors import DecodingError
from typing import Callable

LogitsFn = Callable[[list[int]], list[float]]


def pick_token(logits: list[float], allowed: set[int]) -> int:
    """Return the allowed token id with the highest logit."""
    if not allowed:
        raise DecodingError("no valid token available")
    ids = np.fromiter(allowed, dtype=np.int64)
    try:
        scores = np.asarray(logits)[ids]
        return int(ids[np.argmax(scores)])
    except IndexError as e:
        raise DecodingError(e) from e


def candidate_tokens(vocab: dict[int, str],
                     names: list[str]) -> dict[int, str]:
    """Keep only the tokens whose text appears inside some function name."""
    candidates: dict[int, str] = {}
    for token_id, token in vocab.items():
        if token == "":
            continue
        if any(token in name for name in names):
            candidates.update({token_id: token})
    return candidates


def allowed_name_tokens(prefix: str, candidates: dict[int, str],
                        names: list[str], end_id: int) -> set[int]:
    """Return the token ids that keep ``prefix`` a valid name prefix."""
    valid_ids: set[int] = set()
    for token_id, text in candidates.items():
        if any(name.startswith(prefix + text) for name in names):
            valid_ids.add(token_id)
    if prefix in names:
        valid_ids.add(end_id)
    return valid_ids


def generate_choice(
    logits_fn: LogitsFn,
    prompt_ids: list[int],
    candidates: dict[int, str],
    choices: list[str],
    end_id: int,
) -> str:
    """Generate one function name, token by token, under constraint."""
    input_ids = list(prompt_ids)
    prefix = ""
    while True:
        allowed = allowed_name_tokens(prefix, candidates, choices, end_id)
        if not allowed:
            raise DecodingError(f"no valid token after {prefix!r}")
        if len(allowed) == 1:
            token_id = next(iter(allowed))
        else:
            logits = logits_fn(input_ids)
            token_id = pick_token(logits, allowed)
        if token_id == end_id:
            break
        prefix += candidates[token_id]
        input_ids.append(token_id)
    return prefix
