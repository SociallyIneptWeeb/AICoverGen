"""
Main application for the Ultimate RVC project.

Each tab of the application is defined in its own module in the
`frontend/tabs` directory.Components that are accessed across multiple
tabs are passed as arguments to the render functions in the respective
modules.
"""

import asyncio
import os
from argparse import ArgumentParser

import gradio as gr

from common import GRADIO_TEMP_DIR

from backend.common import delete_gradio_temp
from backend.generate_song_cover import get_named_song_dirs
from backend.manage_audio import get_saved_output_audio
from backend.manage_models import get_saved_model_names

from frontend.tabs.manage_audio import render as render_manage_audio_tab
from frontend.tabs.manage_models import render as render_manage_models_tab
from frontend.tabs.multi_step_generation import render as render_multi_step_tab
from frontend.tabs.one_click_generation import render as render_one_click_tab


def _init_app() -> list[gr.Dropdown]:
    """
    Initialize app by deleting any existing Gradio temp directory and
    updating the choices of all dropdowns.

    Returns
    -------
    tuple[gr.Dropdown, ...]
        Updated dropdowns for selecting voice models, cached songs,
        and output audio files.

    """
    delete_gradio_temp()
    models = [gr.Dropdown(choices=get_saved_model_names()) for _ in range(3)]
    cached_songs = [gr.Dropdown(choices=get_named_song_dirs()) for _ in range(10)]
    output_audio = [gr.Dropdown(choices=get_saved_output_audio())]
    return models + cached_songs + output_audio


os.environ["GRADIO_TEMP_DIR"] = str(GRADIO_TEMP_DIR)

if os.name == "nt":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

css = """
h1 { text-align: center; margin-top: 20px; margin-bottom: 20px; }
"""

with gr.Blocks(title="Ultimate RVC", css=css) as app:

    gr.HTML("<h1>Ultimate RVC ❤️</h1>")

    dummy_checkbox = gr.Checkbox(visible=False)
    confirmation = gr.State(value=False)
    song_dirs = [
        gr.Dropdown(
            label="Song directory",
            info=(
                "Directory where intermediate audio files are stored and loaded from"
                " locally. When a new song is retrieved, its directory is chosen by"
                " default."
            ),
            render=False,
        )
        for _ in range(7)
    ]
    cached_song_1click, cached_song_multi = [
        gr.Dropdown(
            label="Source",
            info="Select a song from the list of cached songs.",
            visible=False,
            render=False,
        )
        for _ in range(2)
    ]
    intermediate_audio = gr.Dropdown(
        label="Song directories",
        multiselect=True,
        info=(
            "Select one or more song directories containing intermediate audio"
            " files to delete."
        ),
        render=False,
    )
    output_audio = gr.Dropdown(
        label="Output audio files",
        multiselect=True,
        info="Select one or more output audio files to delete.",
        render=False,
    )
    model_1click, model_multi = [
        gr.Dropdown(
            label="Voice model",
            render=False,
            info="Select a voice model to use for converting vocals.",
        )
        for _ in range(2)
    ]
    model_delete = gr.Dropdown(label="Voice models", multiselect=True, render=False)

    generate_btns = [
        gr.Button(label, variant="primary", render=False, scale=scale)
        for label, scale, in [
            ("Retrieve song", 1),
            ("Separate vocals/instrumentals", 1),
            ("Separate main/backup vocals", 1),
            ("De-reverb vocals", 1),
            ("Convert vocals", 1),
            ("Post-process vocals", 1),
            ("Pitch shift background", 1),
            ("Mix song", 1),
            ("Generate", 2),
        ]
    ]

    # main tab
    with gr.Tab("Generate song covers"):
        render_one_click_tab(
            generate_btns,
            song_dirs,
            cached_song_1click,
            cached_song_multi,
            model_1click,
            intermediate_audio,
            output_audio,
        )
        render_multi_step_tab(
            generate_btns,
            song_dirs,
            cached_song_1click,
            cached_song_multi,
            model_multi,
            intermediate_audio,
            output_audio,
        )
    with gr.Tab("Manage models"):
        render_manage_models_tab(
            dummy_checkbox,
            confirmation,
            model_delete,
            model_1click,
            model_multi,
        )
    with gr.Tab("Manage audio"):

        render_manage_audio_tab(
            dummy_checkbox,
            confirmation,
            song_dirs,
            cached_song_1click,
            cached_song_multi,
            intermediate_audio,
            output_audio,
        )

    app.load(
        _init_app,
        outputs=[
            model_1click,
            model_multi,
            model_delete,
            intermediate_audio,
            cached_song_1click,
            cached_song_multi,
            *song_dirs,
            output_audio,
        ],
        show_progress="hidden",
    )

    app.unload(delete_gradio_temp)


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
        "--listen-host",
        type=str,
        help="The hostname that the server will use.",
    )
    parser.add_argument(
        "--listen-port",
        type=int,
        help="The listening port that the server will use.",
    )
    args = parser.parse_args()

    app.queue()
    app.launch(
        share=args.share_enabled,
        server_name=(
            None if not args.listen else (args.listen_host or "0.0.0.0")  # noqa: S104
        ),
        server_port=args.listen_port,
    )
