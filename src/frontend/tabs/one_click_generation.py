"""Module which defines the code for the "One-click generation" tab."""

from collections.abc import Sequence
from functools import partial

import gradio as gr

from typing_extra import AudioExt, F0Method, SampleRate

from backend.generate_song_cover import run_pipeline

from frontend.common import (
    PROGRESS_BAR,
    exception_harness,
    toggle_visible_component,
    update_cached_songs,
    update_output_audio,
    update_song_cover_name,
    update_value,
)
from frontend.typing_extra import ConcurrencyId, SourceType


def _toggle_intermediate_audio(
    visible: bool,
) -> list[gr.Accordion]:
    """
    Toggle the visibility of intermediate audio accordions.

    Parameters
    ----------
    visible : bool
        Visibility status of the intermediate audio accordions.

    Returns
    -------
    list[gr.Accordion]
        The intermediate audio accordions.

    """
    accordions = [gr.Accordion(open=False) for _ in range(7)]
    return [gr.Accordion(visible=visible, open=False), *accordions]


def render(
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

        with gr.Accordion("Vocal conversion options", open=False):
            with gr.Row():
                index_rate = gr.Slider(
                    0,
                    1,
                    value=0.5,
                    label="Index rate",
                    info=(
                        "How much of the accent in the voice model to keep in the"
                        " converted vocals. Increase to bias the conversion towards the"
                        " accent of the voice model."
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
                        "How much to mimic the loudness (0) of the input vocals or a"
                        " fixed loudness (1)."
                    ),
                )
            with gr.Row():
                protect = gr.Slider(
                    0,
                    0.5,
                    value=0.33,
                    label="Protect rate",
                    info=(
                        "Protection of voiceless consonants and breath sounds. Decrease"
                        " to increase protection at the cost of indexing accuracy. Set"
                        " to 0.5 to disable."
                    ),
                )
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
                    label="Hop length",
                    info=(
                        "How often the CREPE-based pitch detection algorithm checks for"
                        " pitch changes. Measured in milliseconds. Lower values lead to"
                        " longer conversion times and a higher risk of voice cracks,"
                        " but better pitch accuracy."
                    ),
                )
        with gr.Accordion("Audio mixing options", open=False):
            gr.Markdown("")
            gr.Markdown("**Reverb control on converted vocals**")
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
            with gr.Row():
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
            gr.Markdown("**Volume controls (dB)**")
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
                        "Show intermediate audio tracks generated during song cover"
                        " generation."
                    ),
                )

        intermediate_audio_accordions = [
            gr.Accordion(label, open=False, render=False)
            for label in [
                "Step 0: song retrieval",
                "Step 1a: vocals/instrumentals separation",
                "Step 1b: main vocals/ backup vocals separation",
                "Step 1c: main vocals cleanup",
                "Step 2: conversion of main vocals",
                "Step 3: post-processing of converted vocals",
                "Step 4: pitch shift of background tracks",
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
            "Intermediate audio tracks",
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
            ],
            show_progress="hidden",
        )

        with gr.Row():
            reset_btn = gr.Button(value="Reset settings", scale=2)
            generate_btn = gr.Button("Generate", scale=2, variant="primary")
            song_cover = gr.Audio(label="Song cover", scale=3)

        generate_btn.click(
            partial(
                exception_harness(
                    run_pipeline,
                    info_msg="Song cover generated successfully!",
                ),
                progress_bar=PROGRESS_BAR,
            ),
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
            ],
            outputs=[song_cover, *intermediate_audio_tracks],
            concurrency_limit=1,
            concurrency_id=ConcurrencyId.GPU,
        ).success(
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
            show_progress="hidden",
        ).then(
            partial(update_output_audio, 1, [], [0]),
            outputs=[output_audio],
            show_progress="hidden",
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
