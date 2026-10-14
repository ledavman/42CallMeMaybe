import sys

from .errors import InputError
from .loaders import load_functions, load_prompts, parse_args


def main() -> int:
    args = parse_args()
    try:
        functions = load_functions(args.functions_definition)
        prompts = load_prompts(args.input)
    except InputError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    print(f"{len(functions)} functions, {len(prompts)} prompts loaded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())