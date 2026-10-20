class AppError(Exception):
    """Base class for every expected error of the program."""


class InputError(AppError):
    """Raised when an input file is missing, unreadable or invalid."""


class DecodingError(AppError):
    """Raised when an Error occurs while decoding the file."""


class OutputError(AppError):
    """Raised when an Error occurs while writing the output file."""
