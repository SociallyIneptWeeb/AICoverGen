import os
from pathlib import PurePath
import shutil

import gradio as gr

from backend.exceptions import InputMissingError, InvalidPathError, PathNotFoundError
from backend.common import display_progress, TEMP_AUDIO_DIR


def delete_intermediate_audio(
    song_inputs: list[str],
    progress_bar: gr.Progress | None = None,
    percentage: float = 0.0,
) -> str:
    if not song_inputs:
        raise InputMissingError(
            "Song inputs missing! Please provide a non-empty list of song directories"
        )
    display_progress(
        "[~] Deleting intermediate audio files for selected songs...",
        percentage,
        progress_bar,
    )
    for song_input in song_inputs:
        if not os.path.isdir(song_input):
            raise PathNotFoundError(f"Song directory '{song_input}' does not exist.")

        if not PurePath(song_input).parent == PurePath(TEMP_AUDIO_DIR):
            raise InvalidPathError(
                f"Song directory '{song_input}' is not located in the intermediate audio root directory."
            )
        shutil.rmtree(song_input)
    return "[+] Successfully deleted intermediate audio files for selected songs!"


def delete_all_intermediate_audio(
    progress_bar: gr.Progress | None = None,
    percentages: list[float] = [0.0],
) -> str:
    if len(percentages) != 1:
        raise ValueError("Percentages must be a list of length 1.")
    display_progress("[~] Deleting all audio files...", percentages[0], progress_bar)
    if os.path.isdir(TEMP_AUDIO_DIR):
        shutil.rmtree(TEMP_AUDIO_DIR)

    return "[+] All intermediate audio files successfully deleted!"
