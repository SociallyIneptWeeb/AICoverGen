"""
This module contains custom exceptions that are raised by the backend.
"""


class InputMissingError(ValueError):
    """
    Raised when an input is missing.
    """

    pass


class InvalidPathError(OSError):
    """
    Raised when a path is invalid.
    """

    pass


class PathNotFoundError(OSError):
    """
    Raised when a path is not found.
    """

    pass


class PathExistsError(OSError):
    """
    Raised when a path already exists.
    """

    pass


class FileTypeError(ValueError):
    """
    Raised when a file is of the wrong type.
    """

    pass
