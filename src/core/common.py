"""Common utility functions for the core of the Ultimate RVC project."""

import hashlib
import json
import shutil
from collections.abc import Sequence
from pathlib import Path

from pydantic import AnyHttpUrl, TypeAdapter, ValidationError
from rich import print as rprint

import gradio as gr

from common import AUDIO_DIR
from typing_extra import Json, StrPath

from core.exceptions import Entity, HttpUrlError, NotFoundError

INTERMEDIATE_AUDIO_BASE_DIR = AUDIO_DIR / "intermediate"
OUTPUT_AUDIO_DIR = AUDIO_DIR / "output"


def display_progress(
    message: str,
    percentage: float | None = None,
    progress_bar: gr.Progress | None = None,
) -> None:
    """
    Display progress message and percentage in console or Gradio
    progress bar.

    Parameters
    ----------
    message : str
        Message to display.
    percentage : float, optional
        Percentage to display.
    progress_bar : gr.Progress, optional
        The Gradio progress bar to update.

    """
    if progress_bar is None:
        rprint(message)
    else:
        progress_bar(percentage, desc=message)


def remove_suffix_after(text: str, occurrence: str) -> str:
    """
    Remove suffix after the first occurrence of a substring in a string.

    Parameters
    ----------
    text : str
        The string to remove the suffix from.
    occurrence : str
        The substring to remove the suffix after.

    Returns
    -------
    str
        The string with the suffix removed.

    """
    location = text.rfind(occurrence)
    if location == -1:
        return text
    return text[: location + len(occurrence)]


def copy_files_to_new_dir(files: Sequence[StrPath], directory: StrPath) -> None:
    """
    Copy files to a new directory.

    Parameters
    ----------
    files : Sequence[StrPath]
        Paths to the files to copy.
    directory : StrPath
        Path to the directory to copy the files to.

    Raises
    ------
    NotFoundError
        If a file does not exist.

    """
    dir_path = Path(directory)
    dir_path.mkdir(parents=True)
    for file in files:
        file_path = Path(file)
        if not file_path.exists():
            raise NotFoundError(entity=Entity.FILE, location=file_path)
        shutil.copyfile(file_path, dir_path / file_path.name)


def copy_file_safe(src: StrPath, dest: StrPath) -> Path:
    """
    Copy a file to a new location, appending a number if a file with the
    same name already exists.

    Parameters
    ----------
    src : strPath
        The source file path.
    dest : strPath
        The candidate destination file path.

    Returns
    -------
    Path
        The final destination file path.

    """
    dest_path = Path(dest)
    src_path = Path(src)
    dest_dir = dest_path.parent
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_file = dest_path
    counter = 1

    while dest_file.exists():
        dest_file = dest_dir / f"{dest_path.stem} ({counter}){src_path.suffix}"
        counter += 1

    shutil.copyfile(src, dest_file)
    return dest_file


def json_dumps(thing: Json) -> str:
    """
    Dump a JSON-serializable object to a JSON string.

    Parameters
    ----------
    thing : Json
        The JSON-serializable object to dump.

    Returns
    -------
    str
        The JSON string representation of the object.

    """
    return json.dumps(thing, ensure_ascii=False, indent=4)


def json_dump(thing: Json, file: StrPath) -> None:
    """
    Dump a JSON-serializable object to a JSON file.

    Parameters
    ----------
    thing : Json
        The JSON-serializable object to dump.
    file : StrPath
        The path to the JSON file.

    """
    with Path(file).open("w", encoding="utf-8") as fp:
        json.dump(thing, fp, ensure_ascii=False, indent=4)


def json_load(file: StrPath, encoding: str = "utf-8") -> Json:
    """
    Load a JSON-serializable object from a JSON file.

    Parameters
    ----------
    file : StrPath
        The path to the JSON file.
    encoding : str, default='utf-8'
        The encoding of the JSON file.

    Returns
    -------
    Json
        The JSON-serializable object loaded from the JSON file.

    """
    with Path(file).open(encoding=encoding) as fp:
        return json.load(fp)


def get_hash(thing: Json, size: int = 5) -> str:
    """
    Get the hash of a JSON-serializable object.

    Parameters
    ----------
    thing : Json
        The JSON-serializable object to hash.
    size : int, default=5
        The size of the hash in bytes.

    Returns
    -------
    str
        The hash of the JSON-serializable object.

    """
    return hashlib.blake2b(
        json_dumps(thing).encode("utf-8"),
        digest_size=size,
    ).hexdigest()


# NOTE consider increasing size to 16 otherwise we might have problems
# with hash collisions
def get_file_hash(file: StrPath, size: int = 5) -> str:
    """
    Get the hash of a file.

    Parameters
    ----------
    file : StrPath
        The path to the file.
    size : int, default=5
        The size of the hash in bytes.

    Returns
    -------
    str
        The hash of the file.

    """
    with Path(file).open("rb") as fp:
        file_hash = hashlib.file_digest(fp, lambda: hashlib.blake2b(digest_size=size))
    return file_hash.hexdigest()


def validate_url(url: str) -> None:
    """
    Validate a HTTP-based URL.

    Parameters
    ----------
    url : str
        The URL to validate.

    Raises
    ------
    HttpUrlError
        If the URL is invalid.

    """
    try:
        TypeAdapter(AnyHttpUrl).validate_python(url)
    except ValidationError:
        raise HttpUrlError(url) from None
