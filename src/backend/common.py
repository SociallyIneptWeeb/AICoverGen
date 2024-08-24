"""Common utility functions for the backend."""

from typing import Any
from typings.extra import StrOrBytesPath

import hashlib
import json
import os
import shutil

import gradio as gr

from backend.exceptions import PathNotFoundError

from common import AUDIO_DIR, RVC_MODELS_DIR

INTERMEDIATE_AUDIO_DIR = os.path.join(AUDIO_DIR, "intermediate")
OUTPUT_AUDIO_DIR = os.path.join(AUDIO_DIR, "output")


def display_progress(
    message: str,
    percentage: float | None = None,
    progress_bar: gr.Progress | None = None,
) -> None:
    """
    Display progress message and percentage in console or Gradio progress bar.

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
        print(message)
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
    else:
        return text[: location + len(occurrence)]


def copy_files_to_new_folder(file_paths: list[str], folder_path: str) -> None:
    """
    Copy files to a new folder.

    Parameters
    ----------
    file_paths : list[str]
        List of file paths to copy.
    folder_path : str
        Path of the folder to copy the files to.

    Raises
    ------
    PathNotFoundError
        If a file does not exist.
    """
    os.makedirs(folder_path)
    for file_path in file_paths:
        if not os.path.exists(file_path):
            raise PathNotFoundError(f"File not found: {file_path}")
        shutil.copyfile(
            file_path, os.path.join(folder_path, os.path.basename(file_path))
        )


def get_path_stem(path: str) -> str:
    """
    Get the stem of a file path.

    The stem is the name of the file that the path points to,
    not including its extension.

    Parameters
    ----------
    path : str
        The file path.

    Returns
    -------
    str
        The stem of the file path.
    """
    return os.path.splitext(os.path.basename(path))[0]


def json_dumps(thing: Any) -> str:
    """
    Dump a Python object to a JSON string.

    Parameters
    ----------
    thing : Any
        The object to dump.

    Returns
    -------
    str
        The JSON string representation of the object.
    """
    return json.dumps(
        thing, ensure_ascii=False, sort_keys=True, indent=4, separators=(",", ": ")
    )


def json_dump(thing: Any, path: StrOrBytesPath) -> None:
    """
    Dump a Python object to a JSON file.

    Parameters
    ----------
    thing : Any
        The object to dump.
    path : str
        The path of the JSON file.
    """
    with open(path, "w", encoding="utf-8") as file:
        json.dump(
            thing,
            file,
            ensure_ascii=False,
            sort_keys=True,
            indent=4,
            separators=(",", ": "),
        )


def json_load(path: StrOrBytesPath, encoding: str = "utf-8") -> Any:
    """
    Load a Python object from a JSON file.

    Parameters
    ----------
    path : str
        The path of the JSON file.
    encoding : str, default='utf-8'
        The encoding of the file.

    Returns
    -------
    Any
        The Python object loaded from the JSON file.
    """
    with open(path, encoding=encoding) as file:
        return json.load(file)


def get_hash(thing: Any, size: int = 5) -> str:
    """
    Get a hash of a Python object.

    Parameters
    ----------
    thing : Any
        The object to hash.
    size : int, default=5
        The size of the hash in bytes.

    Returns
    -------
    str
        The hash of the object.
    """
    return hashlib.blake2b(
        json_dumps(thing).encode("utf-8"), digest_size=size
    ).hexdigest()


# TODO consider increasing size to 16
# otherwise we might have problems with hash collisions
def get_file_hash(filepath: StrOrBytesPath, size: int = 5) -> str:
    """
    Get the hash of a file.

    Parameters
    ----------
    filepath : str
        The path of the file.
    size : int, default=5
        The size of the hash in bytes.

    Returns
    -------
    str
        The hash of the file.
    """
    with open(filepath, "rb") as f:
        file_hash = hashlib.file_digest(f, lambda: hashlib.blake2b(digest_size=size))
    return file_hash.hexdigest()


def get_rvc_model(voice_model: str) -> tuple[str, str]:
    """
    Get the RVC model file and optional index file for a voice model.

    When no index file exists, an empty string is returned.

    Parameters
    ----------
    voice_model : str
        The name of the voice model.

    Returns
    -------
    model_path : str
        The path of the RVC model file.
    index_path : str
        The path of the RVC index file.

    Raises
    ------
    PathNotFoundError
        If the directory of the voice model does not exist or
        if no model file exists in the directory.
    """
    rvc_model_filename, rvc_index_filename = None, None
    model_dir = os.path.join(RVC_MODELS_DIR, voice_model)
    if not os.path.exists(model_dir):
        raise PathNotFoundError(
            f"Voice model directory '{voice_model}' does not exist."
        )
    for file in os.listdir(model_dir):
        ext = os.path.splitext(file)[1]
        if ext == ".pth":
            rvc_model_filename = file
        if ext == ".index":
            rvc_index_filename = file

    if rvc_model_filename is None:
        raise PathNotFoundError(f"No model file exists in {model_dir}.")

    return os.path.join(model_dir, rvc_model_filename), (
        os.path.join(model_dir, rvc_index_filename) if rvc_index_filename else ""
    )
