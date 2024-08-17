from functools import partial
import gradio as gr

from backend.manage_audio import (
    delete_all_intermediate_audio,
    delete_intermediate_audio,
    delete_output_audio,
    delete_all_output_audio,
    delete_all_audio,
)
from frontend.common import (
    identity,
    confirm_box_js,
    confirmation_harness,
    update_cached_input_songs,
    update_output_audio,
    PROGRESS_BAR,
)


def render(
    dummy_deletion_checkbox: gr.Checkbox,
    delete_confirmation: gr.State,
    song_dir_dropdowns: list[gr.Dropdown],
    cached_input_songs_dropdown: gr.Dropdown,
    cached_input_songs_dropdown2: gr.Dropdown,
    intermediate_audio_to_delete: gr.Dropdown,
    output_audio_to_delete: gr.Dropdown,
):
    with gr.Tab("Delete audio"):
        with gr.Accordion("Intermediate audio", open=False):
            with gr.Row():
                with gr.Column():
                    intermediate_audio_to_delete.render()
                    delete_intermediate_audio_btn = gr.Button(
                        "Delete selected",
                        variant="secondary",
                    )
                    delete_all_intermediate_audio_btn = gr.Button(
                        "Delete all", variant="primary"
                    )
                with gr.Row():
                    intermediate_audio_delete_msg = gr.Textbox(
                        label="Output message", interactive=False
                    )
        with gr.Accordion("Output audio", open=False):
            with gr.Row():
                with gr.Column():
                    output_audio_to_delete.render()
                    delete_output_audio_btn = gr.Button(
                        "Delete selected", variant="secondary"
                    )
                    delete_all_output_audio_btn = gr.Button(
                        "Delete all", variant="primary"
                    )
                with gr.Row():
                    output_audio_delete_msg = gr.Textbox(
                        label="Output message", interactive=False
                    )
        with gr.Accordion("All audio", open=True):
            with gr.Row():
                delete_all_audio_btn = gr.Button("Delete", variant="primary")
                delete_all_audio_msg = gr.Textbox(
                    label="Output message", interactive=False
                )

        delete_intermediate_audio_click = delete_intermediate_audio_btn.click(
            identity,
            inputs=dummy_deletion_checkbox,
            outputs=delete_confirmation,
            js=confirm_box_js(
                "Are you sure you want to delete intermediate audio files for the selected songs?"
            ),
            show_progress="hidden",
        ).then(
            partial(
                confirmation_harness(delete_intermediate_audio),
                progress_bar=PROGRESS_BAR,
            ),
            inputs=[delete_confirmation, intermediate_audio_to_delete],
            outputs=intermediate_audio_delete_msg,
        )

        delete_all_intermediate_audio_click = delete_all_intermediate_audio_btn.click(
            identity,
            inputs=dummy_deletion_checkbox,
            outputs=delete_confirmation,
            js=confirm_box_js(
                "Are you sure you want to delete all intermediate audio files?"
            ),
            show_progress="hidden",
        ).then(
            partial(
                confirmation_harness(delete_all_intermediate_audio),
                progress_bar=PROGRESS_BAR,
            ),
            inputs=delete_confirmation,
            outputs=intermediate_audio_delete_msg,
        )

        delete_output_audio_click = delete_output_audio_btn.click(
            identity,
            inputs=dummy_deletion_checkbox,
            outputs=delete_confirmation,
            js=confirm_box_js(
                "Are you sure you want to delete the selected output audio files?"
            ),
            show_progress="hidden",
        ).then(
            partial(
                confirmation_harness(delete_output_audio),
                progress_bar=PROGRESS_BAR,
            ),
            inputs=[delete_confirmation, output_audio_to_delete],
            outputs=output_audio_delete_msg,
        )

        delete_all_output_audio_click = delete_all_output_audio_btn.click(
            identity,
            inputs=dummy_deletion_checkbox,
            outputs=delete_confirmation,
            js=confirm_box_js(
                "Are you sure you want to delete all output audio files?"
            ),
            show_progress="hidden",
        ).then(
            partial(
                confirmation_harness(delete_all_output_audio),
                progress_bar=PROGRESS_BAR,
            ),
            inputs=delete_confirmation,
            outputs=output_audio_delete_msg,
        )

        delete_all_audio_click = delete_all_audio_btn.click(
            identity,
            inputs=dummy_deletion_checkbox,
            outputs=delete_confirmation,
            js=confirm_box_js("Are you sure you want to delete all audio files?"),
            show_progress="hidden",
        ).then(
            partial(confirmation_harness(delete_all_audio), progress_bar=PROGRESS_BAR),
            inputs=delete_confirmation,
            outputs=delete_all_audio_msg,
        )

        for click_event in [
            delete_intermediate_audio_click,
            delete_all_intermediate_audio_click,
        ]:
            click_event.success(
                partial(
                    update_cached_input_songs, 3 + len(song_dir_dropdowns), [], [0]
                ),
                outputs=[
                    intermediate_audio_to_delete,
                    cached_input_songs_dropdown,
                    cached_input_songs_dropdown2,
                    *song_dir_dropdowns,
                ],
                show_progress="hidden",
            )

        for click_event in [delete_output_audio_click, delete_all_output_audio_click]:
            click_event.success(
                partial(update_output_audio, 1, [], [0]),
                outputs=[output_audio_to_delete],
                show_progress="hidden",
            )

        delete_all_audio_click.success(
            partial(update_output_audio, 1, [], [0]),
            outputs=[output_audio_to_delete],
            show_progress="hidden",
        ).then(
            partial(update_cached_input_songs, 3 + len(song_dir_dropdowns), [], [0]),
            outputs=[
                intermediate_audio_to_delete,
                cached_input_songs_dropdown,
                cached_input_songs_dropdown2,
                *song_dir_dropdowns,
            ],
            show_progress="hidden",
        )
