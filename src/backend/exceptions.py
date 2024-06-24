class InputMissingError(ValueError):
    pass


class InvalidPathError(OSError):
    pass


class PathNotFoundError(OSError):
    pass


class PathExistsError(OSError):
    pass


class FileTypeError(ValueError):
    pass
