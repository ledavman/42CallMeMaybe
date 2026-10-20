import sys

from llm_sdk import Small_LLM_Model

from .errors import AppError
from .loaders import load_functions, load_prompts
from .loaders import parse_args
from .pipeline import make_context, run_all
from .writer import write_output


def main() -> int:
    args = parse_args()
    try:
        functions = load_functions(args.functions_definition)
        prompts = load_prompts(args.input)
        model = Small_LLM_Model()
        ctx = make_context(model, functions)
        results = run_all(prompts, ctx)
        write_output(args.output, results)
    except (AppError, OSError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
