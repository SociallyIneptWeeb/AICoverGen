"""
Web application for the Ultimate RVC project.

Each tab of the application is defined in its own module in the
`web/tabs` directory. Components that are accessed across multiple
tabs are passed as arguments to the render functions in the respective
modules.
"""

from typing import Annotated

import os

import gradio as gr

import typer

from ultimate_rvc.common import AUDIO_DIR, MODELS_DIR, TEMP_DIR
from ultimate_rvc.core.generate.song_cover import get_named_song_dirs
from ultimate_rvc.core.manage.audio import get_saved_output_audio
from ultimate_rvc.core.manage.models import get_saved_model_names
from ultimate_rvc.web.tabs.manage_audio import render as render_manage_audio_tab
from ultimate_rvc.web.tabs.manage_models import render as render_manage_models_tab
from ultimate_rvc.web.tabs.multi_step_generation import render as render_multi_step_tab
from ultimate_rvc.web.tabs.one_click_generation import render as render_one_click_tab
from ultimate_rvc.web.tabs.other_settings import render as render_other_settings_tab

app_wrapper = typer.Typer()


def _init_app() -> list[gr.Dropdown]:
    """
    Initialize the Ultimate RVC web application by updating the choices
    of all dropdown components.

    Returns
    -------
    tuple[gr.Dropdown, ...]
        Updated dropdowns for selecting voice models, cached songs,
        and output audio files.

    """
    model_names = get_saved_model_names()
    named_song_dirs = get_named_song_dirs()
    models = [
        gr.Dropdown(
            choices=model_names,
            value=None if not model_names else model_names[0],
        )
        for _ in range(2)
    ]
    model_delete = [gr.Dropdown(choices=model_names)]
    cached_songs = [gr.Dropdown(choices=named_song_dirs) for _ in range(3)]
    song_dirs = [
        gr.Dropdown(
            choices=named_song_dirs,
            value=None if not named_song_dirs else named_song_dirs[0][1],
        )
        for _ in range(5)
    ]
    output_audio = [gr.Dropdown(choices=get_saved_output_audio())]
    return models + model_delete + cached_songs + song_dirs + output_audio


def render_app() -> gr.Blocks:
    """
    Render the Ultimate RVC web application.

    Returns
    -------
    gr.Blocks
        The rendered web application.

    """
    css = """
    h1 { text-align: center; margin-top: 20px; margin-bottom: 20px; }
    """
    cache_delete_frequency = 86400  # every 24 hours check for files to delete
    cache_delete_cutoff = 86400  # and delete files older than 24 hours

    with gr.Blocks(
        title="Ultimate RVC",
        css=css,
        delete_cache=(cache_delete_frequency, cache_delete_cutoff),
    ) as app:
        gr.HTML("<h1>Ultimate RVC 🧡</h1>")
        song_dirs = [
            gr.Dropdown(
                label="Song directory",
                info=(
                    "Directory where intermediate audio files are stored and loaded"
                    " from locally. When a new song is retrieved, its directory is"
                    " chosen by default."
                ),
                render=False,
            )
            for _ in range(5)
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

        # main tab
        with gr.Tab("Generate song covers"):
            render_one_click_tab(
                song_dirs,
                cached_song_1click,
                cached_song_multi,
                model_1click,
                intermediate_audio,
                output_audio,
            )
            render_multi_step_tab(
                song_dirs,
                cached_song_1click,
                cached_song_multi,
                model_multi,
                intermediate_audio,
                output_audio,
            )
        with gr.Tab("Manage models"):
            render_manage_models_tab(
                model_delete,
                model_1click,
                model_multi,
            )
        with gr.Tab("Manage audio"):
            render_manage_audio_tab(
                song_dirs,
                cached_song_1click,
                cached_song_multi,
                intermediate_audio,
                output_audio,
            )
        with gr.Tab("Other settings"):
            render_other_settings_tab()

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
    return app


app = render_app()


@app_wrapper.command()
def start_app(
    share: Annotated[
        bool,
        typer.Option("--share", "-s", help="Enable sharing"),
    ] = False,
    listen: Annotated[
        bool,
        typer.Option(
            "--listen",
            "-l",
            help="Make the web application reachable from your local network.",
        ),
    ] = False,
    listen_host: Annotated[
        str | None,
        typer.Option(
            "--listen-host",
            "-h",
            help="The hostname that the server will use.",
        ),
    ] = None,
    listen_port: Annotated[
        int | None,
        typer.Option(
            "--listen-port",
            "-p",
            help="The listening port that the server will use.",
        ),
    ] = None,
) -> None:
    """Run the Ultimate RVC web application."""
    os.environ["GRADIO_TEMP_DIR"] = str(TEMP_DIR)
    gr.set_static_paths([MODELS_DIR, AUDIO_DIR])
    app.queue()
    app.launch(
        share=share,
        server_name=(None if not listen else (listen_host or "0.0.0.0")),  # noqa: S104
        server_port=listen_port,
    )


if __name__ == "__main__":
    app_wrapper()
