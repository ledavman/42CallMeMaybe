import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(...)
    parser.add_argument("--functions_definition", default=...)
    # idem pour --input et --output
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ...
    return 0


if __name__ == "__main__":
    raise SystemExit(main())