import json
from pathlib import Path

from .errors import OutputError
from .models import FunctionCall


def write_output(path: str, results: list[FunctionCall]) -> None:
    """Write the results to ``path`` as a JSON array."""
    data = [call.model_dump() for call in results]
    try:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except OSError as e:
        raise OutputError(f"Cannot write '{path}': {e.strerror}") from e
