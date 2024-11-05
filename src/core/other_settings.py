"""Module which defines functions used for managing various settings."""

import shutil

import gradio as gr

from common import TEMP_DIR

from core.common import display_progress


def delete_temp_files(
    progress_bar: gr.Progress | None = None,
    percentage: float = 0.5,
) -> None:
    """

    Delete all temporary files.

    Parameters
    ----------
    progress_bar : gr.Progress, optional
        Progress bar to update.
    percentage : float, optional
        The percentage to display in the progress bar.

    """
    display_progress("Deleting all temporary files...", percentage, progress_bar)
    if TEMP_DIR.is_dir():
        shutil.rmtree(TEMP_DIR)
