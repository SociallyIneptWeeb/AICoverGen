"""Module which defines the code for the "Manage audio" tab."""

from collections.abc import Sequence
from functools import partial

import gradio as gr

from backend.manage_audio import (
    delete_all_audio,
    delete_all_intermediate_audio,
    delete_all_output_audio,
    delete_intermediate_audio,
    delete_output_audio,
)

from frontend.common import (
    PROGRESS_BAR,
    confirm_box_js,
    confirmation_harness,
    render_msg,
    update_cached_songs,
    update_output_audio,
)


def render(
    song_dirs: Sequence[gr.Dropdown],
    cached_song_1click: gr.Dropdown,
    cached_song_multi: gr.Dropdown,
    intermediate_audio: gr.Dropdown,
    output_audio: gr.Dropdown,
) -> None:
    """
    Render "Manage audio" tab.

    Parameters
    ----------
    song_dirs : Sequence[gr.Dropdown]
        Dropdown components for selecting song directories in the
        "Multi-step generation" tab.
    cached_song_1click : gr.Dropdown
        Dropdown for selecting a cached song in the
        "One-click generation" tab
    cached_song_multi : gr.Dropdown
        Dropdown for selecting a cached song in the
        "Multi-step generation" tab
    intermediate_audio : gr.Dropdown
        Dropdown for selecting intermediate audio files to delete in the
        "Delete audio" tab.
    output_audio : gr.Dropdown
        Dropdown for selecting output audio files to delete in the
        "Delete audio" tab.

    """
    dummy_checkbox = gr.Checkbox(visible=False)
    with gr.Tab("Delete audio"):
        with gr.Accordion("Intermediate audio", open=False), gr.Row(equal_height=False):
            with gr.Column():
                intermediate_audio.render()
                intermediate_audio_btn = gr.Button(
                    "Delete selected",
                    variant="secondary",
                )
                all_intermediate_audio_btn = gr.Button(
                    "Delete all",
                    variant="primary",
                )
            with gr.Column():
                intermediate_audio_msg = gr.Textbox(
                    label="Output message",
                    interactive=False,
                )
        with gr.Accordion("Output audio", open=False), gr.Row(equal_height=False):
            with gr.Column():
                output_audio.render()
                output_audio_btn = gr.Button(
                    "Delete selected",
                    variant="secondary",
                )
                all_output_audio_btn = gr.Button(
                    "Delete all",
                    variant="primary",
                )
            with gr.Column():
                output_audio_msg = gr.Textbox(
                    label="Output message",
                    interactive=False,
                )
        with gr.Accordion("All audio", open=True), gr.Row():
            all_audio_btn = gr.Button("Delete", variant="primary")
            all_audio_msg = gr.Textbox(label="Output message", interactive=False)

        intermediate_audio_click = intermediate_audio_btn.click(
            partial(
                confirmation_harness(delete_intermediate_audio),
                progress_bar=PROGRESS_BAR,
            ),
            inputs=[dummy_checkbox, intermediate_audio],
            outputs=intermediate_audio_msg,
            js=confirm_box_js(
                "Are you sure you want to delete the selected song directories?",
            ),
        ).success(
            partial(
                render_msg,
                "[-] Successfully deleted the selected song directories!",
            ),
            outputs=intermediate_audio_msg,
            show_progress="hidden",
        )

        all_intermediate_audio_click = all_intermediate_audio_btn.click(
            partial(
                confirmation_harness(delete_all_intermediate_audio),
                progress_bar=PROGRESS_BAR,
            ),
            inputs=dummy_checkbox,
            outputs=intermediate_audio_msg,
            js=confirm_box_js(
                "Are you sure you want to delete all intermediate audio files?",
            ),
        ).success(
            partial(
                render_msg,
                "[-] Successfully deleted all intermediate audio files!",
            ),
            outputs=intermediate_audio_msg,
            show_progress="hidden",
        )

        output_audio_click = output_audio_btn.click(
            partial(
                confirmation_harness(delete_output_audio),
                progress_bar=PROGRESS_BAR,
            ),
            inputs=[dummy_checkbox, output_audio],
            outputs=output_audio_msg,
            js=confirm_box_js(
                "Are you sure you want to delete the selected output audio files?",
            ),
        ).success(
            partial(
                render_msg,
                "[-] Successfully deleted the selected output audio files!",
            ),
            outputs=output_audio_msg,
            show_progress="hidden",
        )

        all_output_audio_click = all_output_audio_btn.click(
            partial(
                confirmation_harness(delete_all_output_audio),
                progress_bar=PROGRESS_BAR,
            ),
            inputs=dummy_checkbox,
            outputs=output_audio_msg,
            js=confirm_box_js(
                "Are you sure you want to delete all output audio files?",
            ),
        ).success(
            partial(render_msg, "[-] Successfully deleted all output audio files!"),
            outputs=output_audio_msg,
            show_progress="hidden",
        )

        all_audio_click = all_audio_btn.click(
            partial(
                confirmation_harness(delete_all_audio),
                progress_bar=PROGRESS_BAR,
            ),
            inputs=dummy_checkbox,
            outputs=all_audio_msg,
            js=confirm_box_js("Are you sure you want to delete all audio files?"),
        ).success(
            partial(render_msg, "[-] Successfully deleted all audio files!"),
            outputs=all_audio_msg,
            show_progress="hidden",
        )

        _, _, all_audio_update = [
            click_event.success(
                partial(
                    update_cached_songs,
                    3 + len(song_dirs),
                    [],
                    [0],
                ),
                outputs=[
                    intermediate_audio,
                    cached_song_1click,
                    cached_song_multi,
                    *song_dirs,
                ],
                show_progress="hidden",
            )
            for click_event in [
                intermediate_audio_click,
                all_intermediate_audio_click,
                all_audio_click,
            ]
        ]

        for click_event in [
            output_audio_click,
            all_output_audio_click,
            all_audio_update,
        ]:
            click_event.success(
                partial(update_output_audio, 1, [], [0]),
                outputs=[output_audio],
                show_progress="hidden",
            )
