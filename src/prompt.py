from .models import FunctionDefinition


def build_prompt(functions: list[FunctionDefinition], request: str) -> str:
    """Build the text the model will continue with a function name."""
    lines: list[str] = [
        "You pick the function that matches the request.",
        "",
        "Functions:",
    ]
    for func in functions:
        lines.append(f"- {func.name}: {func.description}")
    lines.append(f"Request: {request}")
    lines.append('Answer: {"name": "')
    return "\n".join(lines)
