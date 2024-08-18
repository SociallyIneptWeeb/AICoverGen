import os
from pathlib import PurePath
import shutil

import gradio as gr

from backend.exceptions import InputMissingError, InvalidPathError, PathNotFoundError
from backend.common import display_progress, INTERMEDIATE_AUDIO_DIR, OUTPUT_AUDIO_DIR
from common import GRADIO_TEMP_DIR


def get_output_audio() -> list[tuple[str, str]]:
    if os.path.isdir(OUTPUT_AUDIO_DIR):
        named_output_files = [
            (file_name, os.path.join(OUTPUT_AUDIO_DIR, file_name))
            for file_name in os.listdir(OUTPUT_AUDIO_DIR)
        ]
        return sorted(named_output_files, key=lambda x: x[0])
    return []


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

        if not PurePath(song_input).parent == PurePath(INTERMEDIATE_AUDIO_DIR):
            raise InvalidPathError(
                f"Song directory '{song_input}' is not located in the intermediate audio root directory."
            )
        shutil.rmtree(song_input)
    return "[+] Successfully deleted intermediate audio files for selected songs!"


def delete_all_intermediate_audio(
    progress_bar: gr.Progress | None = None, percentage: float = 0.0
) -> str:
    display_progress(
        "[~] Deleting all intermediate audio files...", percentage, progress_bar
    )
    if os.path.isdir(INTERMEDIATE_AUDIO_DIR):
        shutil.rmtree(INTERMEDIATE_AUDIO_DIR)

    return "[+] All intermediate audio files successfully deleted!"


def delete_output_audio(
    output_audio_files: list[str],
    progress_bar: gr.Progress | None = None,
    percentage: float = 0.0,
) -> str:
    if not output_audio_files:
        raise InputMissingError(
            "Output audio files missing! Please provide a non-empty list of output audio files."
        )
    display_progress(
        "[~] Deleting selected output audio files...", percentage, progress_bar
    )
    for output_audio_file in output_audio_files:
        if not os.path.isfile(output_audio_file):
            raise PathNotFoundError(
                f"Output audio file '{output_audio_file}' does not exist."
            )
        if not PurePath(output_audio_file).parent == PurePath(OUTPUT_AUDIO_DIR):
            raise InvalidPathError(
                f"Output audio file '{output_audio_file}' is not located in the output audio root directory."
            )
        os.remove(output_audio_file)
    return "[+] Successfully deleted selected output audio files!"


def delete_all_output_audio(
    progress_bar: gr.Progress | None = None, percentage: float = 0.0
) -> str:
    display_progress("[~] Deleting all output audio files...", percentage, progress_bar)
    if os.path.isdir(OUTPUT_AUDIO_DIR):
        shutil.rmtree(OUTPUT_AUDIO_DIR)

    return "[+] All output audio files successfully deleted!"


def delete_all_audio(
    progress_bar: gr.Progress | None = None, percentage: float = 0.0
) -> str:
    display_progress("[~] Deleting all audio files...", percentage, progress_bar)
    if os.path.isdir(INTERMEDIATE_AUDIO_DIR):
        shutil.rmtree(INTERMEDIATE_AUDIO_DIR)
    if os.path.isdir(OUTPUT_AUDIO_DIR):
        shutil.rmtree(OUTPUT_AUDIO_DIR)

    return "[+] All audio files successfully deleted!"


def delete_gradio_temp_dir() -> None:
    if os.path.isdir(GRADIO_TEMP_DIR):
        shutil.rmtree(GRADIO_TEMP_DIR)
