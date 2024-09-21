"""Module which defines the code for the "One-click generation" tab."""

from collections.abc import Sequence
from functools import partial
from pathlib import Path

import gradio as gr

from typing_extra import AudioExt, F0Method, SampleRate

from backend.generate_song_cover import run_pipeline

from frontend.common import (
    PROGRESS_BAR,
    EventArgs,
    chain_event_listeners,
    exception_harness,
    show_hop_slider,
    toggle_visible_component,
    update_cached_songs,
    update_output_audio,
    update_song_cover_name,
    update_value,
)
from frontend.typing_extra import RunPipelineHarnessArgs, SourceType


def _run_pipeline_harness(*args: *RunPipelineHarnessArgs) -> tuple[Path | None, ...]:
    """
    Run the song cover generation pipeline in a harness which displays
    a progress bar, re-raises exceptions as Gradio errors, and returns
    the output of the pipeline.

    If the pipeline outputs only a single path, then that output is
    extended with a None value for each intermediate audio component.

    Parameters
    ----------
    *args : *RunPipelineHarnessArgs
        Arguments to forward to the song cover generation pipeline.

    Returns
    -------
    tuple[str | None, ...]
        The output of the song cover generation pipeline, potentially
        extended with None values.

    """
    res = exception_harness(run_pipeline)(*args, progress_bar=PROGRESS_BAR)
    if isinstance(res, tuple):
        return res
    return (None,) * 11 + (res,)


def _toggle_intermediate_audio(
    visible: bool,
) -> list[gr.Accordion | gr.Audio]:
    """
    Toggle the visibility of intermediate audio accordions and their
    associated components.

    Parameters
    ----------
    visible : bool
        Visibility status of the intermediate audio accordions and
        their associated components.

    Returns
    -------
    list[gr.Accordion | gr.Audio]
        The intermediate audio accordions and their associated
        components with updated visibility.

    """
    audio_components = [gr.Audio(value=None) for _ in range(11)]
    accordions = [gr.Accordion(open=False) for _ in range(7)]
    return [gr.Accordion(visible=visible, open=False), *accordions, *audio_components]


def render(
    generate_btns: Sequence[gr.Button],
    song_dirs: Sequence[gr.Dropdown],
    cached_song_1click: gr.Dropdown,
    cached_song_multi: gr.Dropdown,
    model_1click: gr.Dropdown,
    intermediate_audio: gr.Dropdown,
    output_audio: gr.Dropdown,
) -> None:
    """
    Render "One-click generation" tab.

    Parameters
    ----------
    generate_btns : Sequence[gr.Button]
        Buttons used for audio generation in the
        "One-click generation" tab and the "Multi-step generation" tab.
    song_dirs : Sequence[gr.Dropdown]
        Dropdowns for selecting song directories in the
        "Multi-step generation" tab.
    cached_song_1click : gr.Dropdown
        Dropdown for selecting a cached song in the
        "One-click generation" tab
    cached_song_multi : gr.Dropdown
        Dropdown for selecting a cached song in the
        "Multi-step generation" tab
    model_1click : gr.Dropdown
        Dropdown for selecting voice model in the
        "One-click generation" tab.
    intermediate_audio : gr.Dropdown
        Dropdown for selecting intermediate audio files to delete in the
        "Delete audio" tab.
    output_audio : gr.Dropdown
        Dropdown for selecting output audio files to delete in the
        "Delete audio" tab.

    """
    with gr.Tab("One-click generation"):
        (
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            generate_btn,
        ) = generate_btns

        with gr.Accordion("Main options"), gr.Row():
            with gr.Column():
                source_type = gr.Dropdown(
                    list(SourceType),
                    value=SourceType.PATH,
                    label="Source type",
                    type="index",
                    info="The type of source to retrieve a song from.",
                )
                source = gr.Textbox(
                    label="Source",
                    info=(
                        "Link to a song on YouTube or the full path of a local"
                        " audio file."
                    ),
                )
                local_file = gr.Audio(label="Source", type="filepath", visible=False)
                cached_song_1click.render()
                source_type.input(
                    partial(toggle_visible_component, 3),
                    inputs=source_type,
                    outputs=[source, local_file, cached_song_1click],
                    show_progress="hidden",
                )

                local_file.change(
                    update_value,
                    inputs=local_file,
                    outputs=source,
                    show_progress="hidden",
                )
                cached_song_1click.input(
                    update_value,
                    inputs=cached_song_1click,
                    outputs=source,
                    show_progress="hidden",
                )

            with gr.Column():
                model_1click.render()

            with gr.Column():
                n_octaves = gr.Slider(
                    -3,
                    3,
                    value=0,
                    step=1,
                    label="Vocal pitch shift",
                    info=(
                        "The number of octaves to pitch-shift converted vocals by."
                        " Use 1 for male-to-female and -1 for vice-versa."
                    ),
                )
                n_semitones = gr.Slider(
                    -12,
                    12,
                    value=0,
                    step=1,
                    label="Overall pitch shift",
                    info=(
                        "The number of semi-tones to pitch-shift converted vocals,"
                        " instrumentals, and backup vocals by."
                    ),
                )

        with gr.Accordion("Vocal conversion options", open=False), gr.Row():
            index_rate = gr.Slider(
                0,
                1,
                value=0.5,
                label="Index rate",
                info=(
                    "How much of the accent in the voice model to keep in the converted"
                    " vocals. Increase to bias the conversion towards the accent of the"
                    " voice model."
                ),
            )
            filter_radius = gr.Slider(
                0,
                7,
                value=3,
                step=1,
                label="Filter radius",
                info=(
                    "If >=3: apply median filtering to harvested pitch results."
                    " Can help reduce breathiness in the converted vocals."
                ),
            )
            rms_mix_rate = gr.Slider(
                0,
                1,
                value=0.25,
                label="RMS mix rate",
                info=(
                    "How much to mimic the loudness (0) of the input vocals or a fixed"
                    " loudness (1)."
                ),
            )
            protect = gr.Slider(
                0,
                0.5,
                value=0.33,
                label="Protect rate",
                info=(
                    "Protection of voiceless consonants and breath sounds. Decrease to"
                    " increase protection at the cost of indexing accuracy. Set to 0.5"
                    " to disable."
                ),
            )
            with gr.Column():
                f0_method = gr.Dropdown(
                    list(F0Method),
                    value=F0Method.RMVPE,
                    label="Pitch detection algorithm",
                    info=(
                        "The method to use for pitch detection. Best option is RMVPE"
                        " (clarity in vocals), then Mangio-CREPE (smoother vocals)."
                    ),
                )
                hop_length = gr.Slider(
                    32,
                    320,
                    value=128,
                    step=1,
                    visible=False,
                    label="Hop length",
                    info=(
                        "How often the CREPE-based pitch detection algorithm checks for"
                        " pitch changes. Measured in milliseconds. Lower values lead to"
                        " longer conversion times and a higher risk of voice cracks,"
                        " but better pitch accuracy."
                    ),
                )
                f0_method.change(
                    show_hop_slider,
                    inputs=f0_method,
                    outputs=hop_length,
                    show_progress="hidden",
                )
        with gr.Accordion("Audio mixing options", open=False):
            gr.Markdown("")
            gr.Markdown("### Reverb control on converted vocals")
            with gr.Row():
                room_size = gr.Slider(
                    0,
                    1,
                    value=0.15,
                    label="Room size",
                    info=(
                        "Size of the room which reverb effect simulates. Increase for"
                        " longer reverb time."
                    ),
                )
                wet_level = gr.Slider(
                    0,
                    1,
                    value=0.2,
                    label="Wetness level",
                    info="Loudness of converted vocals with reverb effect applied.",
                )
                dry_level = gr.Slider(
                    0,
                    1,
                    value=0.8,
                    label="Dryness level",
                    info="Loudness of converted vocals without reverb effect applied.",
                )
                damping = gr.Slider(
                    0,
                    1,
                    value=0.7,
                    label="Damping level",
                    info="Absorption of high frequencies in reverb effect.",
                )

            gr.Markdown("")
            gr.Markdown("### Volume controls (dB)")
            with gr.Row():
                main_gain = gr.Slider(-20, 20, value=0, step=1, label="Main vocals")
                inst_gain = gr.Slider(-20, 20, value=0, step=1, label="Instrumentals")
                backup_gain = gr.Slider(-20, 20, value=0, step=1, label="Backup vocals")
        with gr.Accordion("Audio output options", open=False):
            with gr.Row():
                output_name = gr.Textbox(
                    value=partial(
                        update_song_cover_name,
                        None,
                        update_placeholder=True,
                    ),
                    inputs=[cached_song_1click, model_1click],
                    label="Output name",
                    info=(
                        "If no name is provided, a suitable name will be generated"
                        " automatically."
                    ),
                    placeholder="Ultimate RVC song cover",
                )
                output_sr = gr.Dropdown(
                    choices=list(SampleRate),
                    value=SampleRate.HZ_44100,
                    label="Output sample rate",
                    info="The sample rate to save the generated song cover in.",
                )
                output_format = gr.Dropdown(
                    list(AudioExt),
                    value=AudioExt.MP3,
                    label="Output format",
                    info="The format to save the generated song cover in.",
                )
            with gr.Row():
                show_intermediate_audio = gr.Checkbox(
                    label="Show intermediate audio",
                    value=False,
                    info=(
                        "Show generated intermediate audio tracks when song cover"
                        " generation completes. Leave unchecked to optimize"
                        " performance."
                    ),
                )

        intermediate_audio_accordions = [
            gr.Accordion(label, open=False, render=False)
            for label in [
                "Step 0: song retrieval",
                "Step 1: vocals/instrumentals separation",
                "Step 2: main vocals/ backup vocals separation",
                "Step 3: main vocals cleanup",
                "Step 4: conversion of main vocals",
                "Step 5: post-processing of converted vocals",
                "Step 6: pitch shift of background tracks",
            ]
        ]
        (
            song_retrieval_accordion,
            vocals_separation_accordion,
            main_vocals_separation_accordion,
            vocal_cleanup_accordion,
            vocal_conversion_accordion,
            vocals_postprocessing_accordion,
            pitch_shift_accordion,
        ) = intermediate_audio_accordions
        intermediate_audio_tracks = [
            gr.Audio(label=label, type="filepath", interactive=False, render=False)
            for label in [
                "Song",
                "Vocals",
                "Instrumentals",
                "Main vocals",
                "Backup vocals",
                "De-reverbed main vocals",
                "Main vocals reverb",
                "Converted vocals",
                "Post-processed vocals",
                "Pitch-shifted instrumentals",
                "Pitch-shifted backup vocals",
            ]
        ]
        (
            song,
            vocals_track,
            instrumentals_track,
            main_vocals_track,
            backup_vocals_track,
            main_vocals_dereverbed_track,
            main_vocals_reverb_track,
            converted_vocals_track,
            postprocessed_vocals_track,
            instrumentals_shifted_track,
            backup_vocals_shifted_track,
        ) = intermediate_audio_tracks
        with gr.Accordion(
            "Access intermediate audio tracks",
            open=False,
            visible=False,
        ) as intermediate_audio_accordion:
            song_retrieval_accordion.render()
            with song_retrieval_accordion:
                song.render()
            vocals_separation_accordion.render()
            with vocals_separation_accordion, gr.Row():
                vocals_track.render()
                instrumentals_track.render()
            main_vocals_separation_accordion.render()
            with main_vocals_separation_accordion, gr.Row():
                main_vocals_track.render()
                backup_vocals_track.render()
            vocal_cleanup_accordion.render()
            with vocal_cleanup_accordion, gr.Row():
                main_vocals_dereverbed_track.render()
                main_vocals_reverb_track.render()
            vocal_conversion_accordion.render()
            with vocal_conversion_accordion:
                converted_vocals_track.render()
            vocals_postprocessing_accordion.render()
            with vocals_postprocessing_accordion:
                postprocessed_vocals_track.render()
            pitch_shift_accordion.render()
            with pitch_shift_accordion, gr.Row():
                instrumentals_shifted_track.render()
                backup_vocals_shifted_track.render()

        show_intermediate_audio.change(
            _toggle_intermediate_audio,
            inputs=show_intermediate_audio,
            outputs=[
                intermediate_audio_accordion,
                *intermediate_audio_accordions,
                *intermediate_audio_tracks,
            ],
            show_progress="hidden",
        )

        with gr.Row():
            reset_btn = gr.Button(value="Reset settings", scale=2)
            generate_btn.render()
            song_cover = gr.Audio(label="Song cover", scale=3)

        generate_event_args_list = [
            EventArgs(
                _run_pipeline_harness,
                inputs=[
                    source,
                    model_1click,
                    n_octaves,
                    n_semitones,
                    f0_method,
                    index_rate,
                    filter_radius,
                    rms_mix_rate,
                    protect,
                    hop_length,
                    room_size,
                    wet_level,
                    dry_level,
                    damping,
                    main_gain,
                    inst_gain,
                    backup_gain,
                    output_sr,
                    output_format,
                    output_name,
                    show_intermediate_audio,
                ],
                outputs=[*intermediate_audio_tracks, song_cover],
            ),
            EventArgs(
                partial(
                    update_cached_songs,
                    3 + len(song_dirs),
                    [],
                    [2],
                ),
                outputs=[
                    cached_song_1click,
                    cached_song_multi,
                    intermediate_audio,
                    *song_dirs,
                ],
                name="then",
                show_progress="hidden",
            ),
            EventArgs(
                partial(update_output_audio, 1, [], [0]),
                outputs=[output_audio],
                name="then",
                show_progress="hidden",
            ),
        ]
        chain_event_listeners(
            generate_btn,
            generate_event_args_list,
            [*generate_btns, show_intermediate_audio],
        )
        reset_btn.click(
            lambda: [
                0,
                0,
                0.5,
                3,
                0.25,
                0.33,
                F0Method.RMVPE,
                128,
                0.15,
                0.2,
                0.8,
                0.7,
                0,
                0,
                0,
                SampleRate.HZ_44100,
                AudioExt.MP3,
                False,
            ],
            outputs=[
                n_octaves,
                n_semitones,
                index_rate,
                filter_radius,
                rms_mix_rate,
                protect,
                f0_method,
                hop_length,
                room_size,
                wet_level,
                dry_level,
                damping,
                main_gain,
                inst_gain,
                backup_gain,
                output_sr,
                output_format,
                show_intermediate_audio,
            ],
            show_progress="hidden",
        )
