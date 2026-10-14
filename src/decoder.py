import numpy as np

from .errors import DecodingError


def pick_token(logits: list[float], allowed: set[int]) -> int:
    """Return the allowed token id with the highest logit."""
    if not allowed:
        raise DecodingError("no valid token available")
    ids = np.fromiter(allowed, dtype=np.int64)
    scores = np.asarray(logits)[ids]
    try:
        return int(ids[np.argmax(scores)])
    except IndexError as e:
        raise DecodingError(e)