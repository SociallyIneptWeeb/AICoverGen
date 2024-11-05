"""Module which defines functions to manage audio files."""

import operator
import shutil
from collections.abc import Sequence
from pathlib import Path

import gradio as gr

from typing_extra import StrPath

from core.common import (
    INTERMEDIATE_AUDIO_BASE_DIR,
    OUTPUT_AUDIO_DIR,
    display_progress,
)
from core.exceptions import (
    Entity,
    InvalidLocationError,
    Location,
    NotFoundError,
    NotProvidedError,
    UIMessage,
)


def get_saved_output_audio() -> list[tuple[str, str]]:
    """
    Get the name and path of all output audio files.

    Returns
    -------
    list[tuple[str, Path]]
        A list of tuples containing the name and path of each output
        audio file.

    """
    if OUTPUT_AUDIO_DIR.is_dir():
        named_output_files = [
            (file_path.name, str(file_path)) for file_path in OUTPUT_AUDIO_DIR.iterdir()
        ]
        return sorted(named_output_files, key=operator.itemgetter(0))
    return []


def delete_intermediate_audio(
    directories: Sequence[StrPath],
    progress_bar: gr.Progress | None = None,
    percentage: float = 0.5,
) -> None:
    """
    Delete provided directories containing intermediate audio files.

    The provided directories must be located in the root of the
    intermediate audio base directory.

    Parameters
    ----------
    directories : Sequence[StrPath]
        Paths to directories containing intermediate audio files to
        delete.
    progress_bar : gr.Progress, optional
        Gradio progress bar to update.
    percentage : float, default=0.5
        Percentage to display in the progress bar.

    Raises
    ------
    NotProvidedError
        If no paths are provided.
    NotFoundError
        if a provided path does not point to an existing directory.
    InvalidLocationError
        If a provided path does not point to a location in the root of
        the intermediate audio base directory.

    """
    if not directories:
        raise NotProvidedError(entity=Entity.DIRECTORIES, ui_msg=UIMessage.NO_SONG_DIRS)
    display_progress(
        "[~] Deleting directories ...",
        percentage,
        progress_bar,
    )
    for directory in directories:
        dir_path = Path(directory)
        if not dir_path.is_dir():
            raise NotFoundError(entity=Entity.DIRECTORY, location=dir_path)
        if dir_path.parent != INTERMEDIATE_AUDIO_BASE_DIR:
            raise InvalidLocationError(
                entity=Entity.DIRECTORY,
                location=Location.INTERMEDIATE_AUDIO_ROOT,
                path=dir_path,
            )
        shutil.rmtree(dir_path)


def delete_all_intermediate_audio(
    progress_bar: gr.Progress | None = None,
    percentage: float = 0.5,
) -> None:
    """
    Delete all intermediate audio files.

    Parameters
    ----------
    progress_bar : gr.Progress, optional
        Gradio progress bar to update.
    percentage : float, default=0.5
        Percentage to display in the progress bar.

    """
    display_progress(
        "[~] Deleting all intermediate audio files...",
        percentage,
        progress_bar,
    )
    if INTERMEDIATE_AUDIO_BASE_DIR.is_dir():
        shutil.rmtree(INTERMEDIATE_AUDIO_BASE_DIR)


def delete_output_audio(
    files: Sequence[StrPath],
    progress_bar: gr.Progress | None = None,
    percentage: float = 0.5,
) -> None:
    """
    Delete provided output audio files.

    The provided files must be located in the root of the output audio
    directory.

    Parameters
    ----------
    files : Sequence[StrPath]
        Paths to the output audio files to delete.
    progress_bar : gr.Progress, optional
        Gradio progress bar to update.
    percentage : float, default=0.5
        Percentage to display in the progress bar.

    Raises
    ------
    NotProvidedError
        If no paths are provided.
    NotFoundError
        If a provided path does not point to an existing file.
    InvalidLocationError
        If a provided path does not point to a location in the root of
        the output audio directory.

    """
    if not files:
        raise NotProvidedError(
            entity=Entity.FILES,
            ui_msg=UIMessage.NO_OUTPUT_AUDIO_FILES,
        )
    display_progress(
        "[~] Deleting output audio files...",
        percentage,
        progress_bar,
    )
    for file in files:
        file_path = Path(file)
        if not file_path.is_file():
            raise NotFoundError(entity=Entity.FILE, location=file_path)
        if file_path.parent != OUTPUT_AUDIO_DIR:
            raise InvalidLocationError(
                entity=Entity.FILE,
                location=Location.OUTPUT_AUDIO_ROOT,
                path=file_path,
            )
        file_path.unlink()


def delete_all_output_audio(
    progress_bar: gr.Progress | None = None,
    percentage: float = 0.5,
) -> None:
    """
    Delete all output audio files.

    Parameters
    ----------
    progress_bar : gr.Progress, optional
        Gradio progress bar to update.
    percentage : float, default=0.5
        Percentage to display in the progress bar.

    """
    display_progress("[~] Deleting all output audio files...", percentage, progress_bar)
    if OUTPUT_AUDIO_DIR.is_dir():
        shutil.rmtree(OUTPUT_AUDIO_DIR)


def delete_all_audio(
    progress_bar: gr.Progress | None = None,
    percentage: float = 0.5,
) -> None:
    """
    Delete all audio files.

    Parameters
    ----------
    progress_bar : gr.Progress, optional
        Gradio progress bar to update.
    percentage : float, default=0.5
        Percentage to display in the progress bar.

    """
    display_progress("[~] Deleting all audio files...", percentage, progress_bar)
    if INTERMEDIATE_AUDIO_BASE_DIR.is_dir():
        shutil.rmtree(INTERMEDIATE_AUDIO_BASE_DIR)
    if OUTPUT_AUDIO_DIR.is_dir():
        shutil.rmtree(OUTPUT_AUDIO_DIR)
