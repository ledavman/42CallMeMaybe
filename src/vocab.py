from .loaders import read_json
from .errors import InputError


def _bytes_to_unicode() -> dict[int, str]:
    """Map each byte to the printable character used in the vocab."""
    printable = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    mapping: dict[int, str] = {b: chr(b) for b in printable}
    extra = 0
    for byte in range(256):
        if byte not in mapping:
            mapping[byte] = chr(256 + extra)
            extra += 1
    return mapping


_UNICODE_TO_BYTE: dict[str, int] = {
    char: byte for byte, char in _bytes_to_unicode().items()
}


def token_to_text(token: str) -> str:
    """Convert a vocab token string into the real text it stands for."""
    raw = bytes(_UNICODE_TO_BYTE[c] for c in token)
    return raw.decode("utf-8")


def load_vocab(path: str) -> dict[int, str]:
    """Return the mapping token id -> real text for every vocab entry."""
    raw = read_json(path)
    if not isinstance(raw, dict):
        raise InputError(f"{path} Expected dict got {type(raw).__name__}")
    vocab: dict[int, str] = {}
    for token, token_id in raw.items():
        if not isinstance(token_id, int):
            raise InputError(f"{token} got a non int token_id")
        try:
            vocab[token_id] = token_to_text(token)
        except (KeyError, UnicodeDecodeError):
            continue
    return vocab
