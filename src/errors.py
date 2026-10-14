class InputError(Exception):
    """Raised when an input file is missing, unreadable or invalid."""

class DecodingError(Exception):
    """Raised when an Error occurs while decoding the file."""