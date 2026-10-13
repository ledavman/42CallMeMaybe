import argparse


def parse_args() -> argparse.Namespace:
    """Parse the command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Translate natural-language prompts into function calls."
    )
    parser.add_argument(
        "--functions_definition",
        default="data/input/functions_definition.json",
        help="path to the functions definition file",
    )
    parser.add_argument(
        "--input",
        default="data/input/function_calling_tests.json",
        help="path to the prompts file",
    )
    parser.add_argument(
        "--output",
        default="data/output/function_calls.json",
        help="path to the functions calls outputs file",
    )
    return parser.parse_args()