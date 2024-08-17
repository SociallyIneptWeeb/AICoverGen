import os
from argparse import ArgumentParser

import gradio as gr
import asyncio


from common import GRADIO_TEMP_DIR

from backend.generate_song_cover import get_named_song_dirs
from backend.manage_voice_models import get_current_models
from backend.manage_audio import delete_gradio_temp_dir, get_output_audio

from frontend.tabs.one_click_generation import render as render_one_click_tab
from frontend.tabs.multi_step_generation import render as render_multi_step_tab
from frontend.tabs.manage_models import render as render_manage_models_tab
from frontend.tabs.manage_audio import render as render_manage_audio_tab


def _init_app() -> tuple[gr.Dropdown, ...]:
    delete_gradio_temp_dir()
    updated_rvc_model_dropdowns = tuple(
        gr.Dropdown(choices=get_current_models()) for _ in range(3)
    )
    updated_song_dir_dropdowns = tuple(
        gr.Dropdown(choices=get_named_song_dirs()) for _ in range(10)
    )
    updated_output_audio_dropdown = (gr.Dropdown(choices=get_output_audio()),)
    return (
        updated_rvc_model_dropdowns
        + updated_song_dir_dropdowns
        + updated_output_audio_dropdown
    )


os.environ["GRADIO_TEMP_DIR"] = GRADIO_TEMP_DIR

if os.name == "nt":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

with gr.Blocks(title="Ultimate RVC") as app:

    gr.Label("Ultimate RVC ❤️", show_label=False)

    dummy_deletion_checkbox = gr.Checkbox(visible=False)
    delete_confirmation = gr.State(False)
    song_dir_dropdowns = [
        gr.Dropdown(
            label="Song directory",
            info="Directory where intermediate audio files are stored and loaded from locally. When a new song is retrieved, its directory is chosen by default.",
            render=False,
        )
        for _ in range(7)
    ]
    cached_input_songs_dropdown, cached_input_songs_dropdown2 = [
        gr.Dropdown(
            label="Song input",
            info="Select a song from the list of cached songs.",
            visible=False,
            render=False,
        )
        for _ in range(2)
    ]
    intermediate_audio_to_delete = gr.Dropdown(
        label="Songs with intermediate audio files",
        multiselect=True,
        info="Select one or more songs to delete their asssociated intermediate audio files.",
        render=False,
    )
    output_audio_to_delete = gr.Dropdown(
        label="Output audio files",
        multiselect=True,
        info="Select one or more output audio files to delete.",
        render=False,
    )
    rvc_model, rvc_model2 = [
        gr.Dropdown(label="Voice model", render=False) for _ in range(2)
    ]
    rvc_models_to_delete = gr.Dropdown(
        label="Voice models", multiselect=True, render=False
    )

    generate_buttons = [
        gr.Button(
            label,
            variant="primary",
            visible=visible,
            render=False,
            scale=scale,
        )
        for label, visible, scale, in [
            ("Retrieve song", True, 1),
            ("Separate vocals/instrumentals", True, 1),
            ("Separate main/backup vocals", True, 1),
            ("De-reverb vocals", True, 1),
            ("Convert vocals", True, 1),
            ("Post-process vocals", True, 1),
            ("Pitch shift background", True, 1),
            ("Mix song cover", True, 1),
            ("Generate", True, 2),
            ("Generate step-by-step", False, 1),
        ]
    ]

    # main tab
    with gr.Tab("Generate song covers"):
        render_one_click_tab(
            generate_buttons,
            song_dir_dropdowns,
            cached_input_songs_dropdown,
            cached_input_songs_dropdown2,
            rvc_model,
            intermediate_audio_to_delete,
            output_audio_to_delete,
        )
        render_multi_step_tab(
            generate_buttons,
            song_dir_dropdowns,
            cached_input_songs_dropdown,
            cached_input_songs_dropdown2,
            rvc_model2,
            intermediate_audio_to_delete,
            output_audio_to_delete,
        )
    with gr.Tab("Manage models"):
        render_manage_models_tab(
            dummy_deletion_checkbox,
            delete_confirmation,
            rvc_models_to_delete,
            rvc_model,
            rvc_model2,
        )
    with gr.Tab("Manage audio"):

        render_manage_audio_tab(
            dummy_deletion_checkbox,
            delete_confirmation,
            song_dir_dropdowns,
            cached_input_songs_dropdown,
            cached_input_songs_dropdown2,
            intermediate_audio_to_delete,
            output_audio_to_delete,
        )

    app.load(
        _init_app,
        outputs=[
            rvc_model,
            rvc_model2,
            rvc_models_to_delete,
            intermediate_audio_to_delete,
            cached_input_songs_dropdown,
            cached_input_songs_dropdown2,
            *song_dir_dropdowns,
            output_audio_to_delete,
        ],
        show_progress="hidden",
    )

    app.unload(delete_gradio_temp_dir)


if __name__ == "__main__":

    parser = ArgumentParser(
        description="Generate a song cover song in the song_output/id directory.",
        add_help=True,
    )
    parser.add_argument(
        "--share",
        action="store_true",
        dest="share_enabled",
        default=False,
        help="Enable sharing",
    )
    parser.add_argument(
        "--listen",
        action="store_true",
        default=False,
        help="Make the WebUI reachable from your local network.",
    )
    parser.add_argument(
        "--listen-host", type=str, help="The hostname that the server will use."
    )
    parser.add_argument(
        "--listen-port", type=int, help="The listening port that the server will use."
    )
    args = parser.parse_args()

    app.queue()
    app.launch(
        share=args.share_enabled,
        server_name=None if not args.listen else (args.listen_host or "0.0.0.0"),
        server_port=args.listen_port,
    )
