from typing import Callable
from typing_extensions import Unpack
from extra_typing import P, MixSongCoverHarnessArgs, RunPipelineHarnessArgs
from functools import partial

import gradio as gr

from frontend.common import (
    EventArgs,
    setup_consecutive_event_listeners_with_toggled_interactivity,
    exception_harness,
    update_cached_input_songs,
    get_song_cover_name_harness,
    toggle_visible_component,
    show_hop_slider,
    update_value,
    PROGRESS_BAR,
)

from backend.generate_song_cover import (
    retrieve_song,
    separate_vocals,
    separate_main_vocals,
    dereverb_vocals,
    convert_vocals,
    postprocess_vocals,
    pitch_shift_background,
    mix_song_cover,
    run_pipeline,
)


def _duplication_harness(
    fun: Callable[P, str | tuple[str, ...]],
) -> Callable[P, tuple[str, ...]]:
    def _wrapped(*args: P.args, **kwargs: P.kwargs) -> tuple[str, ...]:

        res = exception_harness(fun)(*args, **kwargs)
        if not isinstance(res, tuple):
            return (res, res)
        else:
            return (res[0],) + res

    return _wrapped


def _mix_song_cover_harness(
    vocals_path: str,
    instrumentals_path: str,
    backup_vocals_path: str,
    instrumentals_shifted_path: str,
    backup_vocals_shifted_path: str,
    *args: Unpack[MixSongCoverHarnessArgs],
    percentages: list[float],
) -> str:
    return exception_harness(mix_song_cover)(
        vocals_path,
        instrumentals_shifted_path or instrumentals_path,
        backup_vocals_shifted_path or backup_vocals_path,
        *args,
        progress_bar=PROGRESS_BAR,
        percentages=percentages,
    )


def _run_pipeline_harness(
    *args: Unpack[RunPipelineHarnessArgs],
) -> tuple[str | None, ...]:

    res = exception_harness(run_pipeline)(*args, progress_bar=PROGRESS_BAR)
    if isinstance(res, tuple):
        return res
    else:
        return (None,) * 11 + (res,)


def _toggle_intermediate_files_accordion(
    visible: bool,
) -> list[gr.Accordion | gr.Audio]:
    audio_components = list(gr.Audio(value=None) for _ in range(11))
    accordions = list(gr.Accordion(open=False) for _ in range(7))
    return [gr.Accordion(visible=visible, open=False)] + accordions + audio_components


def render(
    generate_buttons: list[gr.Button],
    song_dir_dropdowns: list[gr.Dropdown],
    cached_input_songs_dropdown: gr.Dropdown,
    cached_input_songs_dropdown2: gr.Dropdown,
    rvc_model: gr.Dropdown,
    intermediate_audio_to_delete: gr.Dropdown,
) -> None:

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
            generate_btn2,
        ) = generate_buttons

        current_song_dir = gr.State(None)
        with gr.Accordion("Main options"):
            with gr.Row():
                with gr.Column():
                    song_input_type_dropdown = gr.Dropdown(
                        [
                            "YouTube link/local path",
                            "Local file",
                            "Cached song",
                        ],
                        value="YouTube link/local path",
                        label="Song input type",
                        type="index",
                    )
                    song_input = gr.Text(
                        label="Song input",
                        info="Link to a song on YouTube or the full path of a local audio file.",
                    )
                    local_file = gr.Audio(
                        label="Song input",
                        type="filepath",
                        visible=False,
                    )
                    cached_input_songs_dropdown.render()
                    song_input_type_dropdown.input(
                        partial(toggle_visible_component, 3),
                        inputs=song_input_type_dropdown,
                        outputs=[song_input, local_file, cached_input_songs_dropdown],
                        show_progress="hidden",
                    )

                    local_file.change(
                        update_value,
                        inputs=local_file,
                        outputs=song_input,
                        show_progress="hidden",
                    )
                    cached_input_songs_dropdown.input(
                        update_value,
                        inputs=cached_input_songs_dropdown,
                        outputs=song_input,
                        show_progress="hidden",
                    )

                with gr.Column():
                    rvc_model.render()

                with gr.Column():
                    pitch_change_vocals = gr.Slider(
                        -3,
                        3,
                        value=0,
                        step=1,
                        label="Pitch shift of vocals",
                        info="Shift pitch of converted vocals. Measured in octaves. Generally, use 1 for male-to-female conversions and -1 for vice-versa.",
                    )
                    pitch_change_all = gr.Slider(
                        -12,
                        12,
                        value=0,
                        step=1,
                        label="Overall pitch shift",
                        info="Shift pitch of converted vocals, backup vocals and instrumentals. Measured in semi-tones. Altering this slightly reduces sound quality.",
                    )

        with gr.Accordion("Vocal conversion options", open=False):
            with gr.Row():
                index_rate = gr.Slider(
                    0,
                    1,
                    value=0.5,
                    label="Index rate",
                    info="Controls how much of the accent in the voice model to keep in the converted vocals",
                )
                filter_radius = gr.Slider(
                    0,
                    7,
                    value=3,
                    step=1,
                    label="Filter radius",
                    info="If >=3: apply median filtering to the harvested pitch results. Can reduce breathiness",
                )
                rms_mix_rate = gr.Slider(
                    0,
                    1,
                    value=0.25,
                    label="RMS mix rate",
                    info="Control how much to mimic the loudness (0) of the input vocals or a fixed loudness (1)",
                )
                protect = gr.Slider(
                    0,
                    0.5,
                    value=0.33,
                    label="Protect rate",
                    info="Protect voiceless consonants and breath sounds. Set to 0.5 to disable.",
                )
                with gr.Column():
                    f0_method = gr.Dropdown(
                        ["rmvpe", "mangio-crepe"],
                        value="rmvpe",
                        label="Pitch detection algorithm",
                        info="Best option is rmvpe (clarity in vocals), then mangio-crepe (smoother vocals)",
                    )
                    crepe_hop_length = gr.Slider(
                        32,
                        320,
                        value=128,
                        step=1,
                        visible=False,
                        label="Crepe hop length",
                        info="Lower values leads to longer conversions and higher risk of voice cracks, but better pitch accuracy.",
                    )
                    f0_method.change(
                        show_hop_slider,
                        inputs=f0_method,
                        outputs=crepe_hop_length,
                        show_progress="hidden",
                    )
        with gr.Accordion("Audio mixing options", open=False):
            gr.Markdown("")
            gr.Markdown("### Reverb control on converted vocals")
            with gr.Row():
                reverb_rm_size = gr.Slider(
                    0,
                    1,
                    value=0.15,
                    label="Room size",
                    info="The larger the room, the longer the reverb time",
                )
                reverb_wet = gr.Slider(
                    0,
                    1,
                    value=0.2,
                    label="Wetness level",
                    info="Loudness level of converted vocals with reverb",
                )
                reverb_dry = gr.Slider(
                    0,
                    1,
                    value=0.8,
                    label="Dryness level",
                    info="Loudness level of converted vocals without reverb",
                )
                reverb_damping = gr.Slider(
                    0,
                    1,
                    value=0.7,
                    label="Damping level",
                    info="Absorption of high frequencies in the reverb",
                )

            gr.Markdown("")
            gr.Markdown("### Volume controls (dB)")
            with gr.Row():
                main_gain = gr.Slider(-20, 20, value=0, step=1, label="Main vocals")
                inst_gain = gr.Slider(-20, 20, value=0, step=1, label="Instrumentals")
                backup_gain = gr.Slider(-20, 20, value=0, step=1, label="Backup vocals")
        with gr.Accordion("Audio output options", open=False):
            with gr.Row():
                output_name = gr.Text(
                    label="Output file name",
                    info="If no name is provided, a suitable name will be generated automatically.",
                    placeholder="Ultimate RVC song cover",
                )
                output_sr = gr.Dropdown(
                    choices=[16000, 44100, 48000, 96000, 192000],
                    value=44100,
                    label="Output sample rate",
                )
                output_format = gr.Dropdown(
                    [
                        "mp3",
                        "wav",
                        "flac",
                        "aac",
                        "m4a",
                        "ogg",
                    ],
                    value="mp3",
                    label="Output file format",
                )
            with gr.Row():
                show_intermediate_files = gr.Checkbox(
                    label="Show intermediate audio files",
                    value=False,
                    info="Show generated intermediate audio files when song cover generation completes. Leave unchecked to optimize performance.",
                )
            rvc_model.change(
                partial(get_song_cover_name_harness, None, update_key="placeholder"),
                inputs=[cached_input_songs_dropdown, rvc_model],
                outputs=output_name,
                show_progress="hidden",
            )
            cached_input_songs_dropdown.change(
                partial(get_song_cover_name_harness, None, update_key="placeholder"),
                inputs=[cached_input_songs_dropdown, rvc_model],
                outputs=output_name,
                show_progress="hidden",
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
        (
            original_track,
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
        ) = [
            gr.Audio(label=label, type="filepath", interactive=False, render=False)
            for label in [
                "Input song",
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
        with gr.Accordion(
            "Access intermediate audio files", open=False, visible=False
        ) as intermediate_files_accordion:
            song_retrieval_accordion.render()
            with song_retrieval_accordion:
                original_track.render()
            vocals_separation_accordion.render()
            with vocals_separation_accordion:
                with gr.Row():
                    vocals_track.render()
                    instrumentals_track.render()
            main_vocals_separation_accordion.render()
            with main_vocals_separation_accordion:
                with gr.Row():
                    main_vocals_track.render()
                    backup_vocals_track.render()

            vocal_cleanup_accordion.render()
            with vocal_cleanup_accordion:
                with gr.Row():
                    main_vocals_dereverbed_track.render()
                    main_vocals_reverb_track.render()
            vocal_conversion_accordion.render()
            with vocal_conversion_accordion:
                converted_vocals_track.render()
            vocals_postprocessing_accordion.render()
            with vocals_postprocessing_accordion:
                postprocessed_vocals_track.render()
            pitch_shift_accordion.render()
            with pitch_shift_accordion:
                with gr.Row():
                    instrumentals_shifted_track.render()
                    backup_vocals_shifted_track.render()

        with gr.Row():
            clear_btn = gr.Button(
                value="Reset settings",
                scale=2,
            )
            generate_btn2.render()
            generate_btn.render()
            song_cover_track = gr.Audio(label="Song cover", scale=3)
        show_intermediate_files.change(
            _toggle_intermediate_files_accordion,
            inputs=show_intermediate_files,
            outputs=[
                intermediate_files_accordion,
                song_retrieval_accordion,
                vocals_separation_accordion,
                main_vocals_separation_accordion,
                vocal_cleanup_accordion,
                vocal_conversion_accordion,
                vocals_postprocessing_accordion,
                pitch_shift_accordion,
                original_track,
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
            ],
            show_progress="hidden",
        )
        generate_event_args_list = [
            EventArgs(
                _run_pipeline_harness,
                inputs=[
                    song_input,
                    rvc_model,
                    pitch_change_vocals,
                    pitch_change_all,
                    index_rate,
                    filter_radius,
                    rms_mix_rate,
                    protect,
                    f0_method,
                    crepe_hop_length,
                    reverb_rm_size,
                    reverb_wet,
                    reverb_dry,
                    reverb_damping,
                    main_gain,
                    inst_gain,
                    backup_gain,
                    output_sr,
                    output_format,
                    output_name,
                    show_intermediate_files,
                ],
                outputs=[
                    original_track,
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
                    song_cover_track,
                ],
            ),
            EventArgs(
                partial(update_cached_input_songs, 3 + len(song_dir_dropdowns)),
                outputs=[
                    cached_input_songs_dropdown,
                    intermediate_audio_to_delete,
                    cached_input_songs_dropdown2,
                ]
                + song_dir_dropdowns,
                name="then",
                show_progress="hidden",
            ),
        ]
        setup_consecutive_event_listeners_with_toggled_interactivity(
            generate_btn,
            generate_event_args_list,
            generate_buttons + [show_intermediate_files],
        )
        percentages = [i / 15 for i in range(15)]

        generate_btn2_event_args_list = [
            EventArgs(
                partial(
                    _duplication_harness(retrieve_song),
                    progress_bar=PROGRESS_BAR,
                    percentages=percentages[:3],
                ),
                inputs=[song_input],
                outputs=[song_cover_track, original_track, current_song_dir],
            ),
            EventArgs(
                partial(
                    _duplication_harness(separate_vocals),
                    progress_bar=PROGRESS_BAR,
                    percentages=percentages[3:5],
                ),
                inputs=[original_track, current_song_dir],
                outputs=[song_cover_track, vocals_track, instrumentals_track],
            ),
            EventArgs(
                partial(
                    _duplication_harness(separate_main_vocals),
                    progress_bar=PROGRESS_BAR,
                    percentages=percentages[5:7],
                ),
                inputs=[vocals_track, current_song_dir],
                outputs=[song_cover_track, main_vocals_track, backup_vocals_track],
            ),
            EventArgs(
                partial(
                    _duplication_harness(dereverb_vocals),
                    progress_bar=PROGRESS_BAR,
                    percentages=percentages[7:9],
                ),
                inputs=[main_vocals_track, current_song_dir],
                outputs=[
                    song_cover_track,
                    main_vocals_dereverbed_track,
                    main_vocals_reverb_track,
                ],
            ),
            EventArgs(
                partial(
                    _duplication_harness(convert_vocals),
                    progress_bar=PROGRESS_BAR,
                    percentage=percentages[9],
                ),
                inputs=[
                    main_vocals_dereverbed_track,
                    current_song_dir,
                    rvc_model,
                    pitch_change_vocals,
                    pitch_change_all,
                    index_rate,
                    filter_radius,
                    rms_mix_rate,
                    protect,
                    f0_method,
                    crepe_hop_length,
                ],
                outputs=[song_cover_track, converted_vocals_track],
            ),
            EventArgs(
                partial(
                    _duplication_harness(postprocess_vocals),
                    progress_bar=PROGRESS_BAR,
                    percentage=percentages[10],
                ),
                inputs=[
                    converted_vocals_track,
                    current_song_dir,
                    reverb_rm_size,
                    reverb_wet,
                    reverb_dry,
                    reverb_damping,
                ],
                outputs=[song_cover_track, postprocessed_vocals_track],
            ),
            EventArgs(
                partial(
                    _duplication_harness(pitch_shift_background),
                    progress_bar=PROGRESS_BAR,
                    percentages=percentages[11:13],
                ),
                inputs=[
                    instrumentals_track,
                    backup_vocals_track,
                    current_song_dir,
                    pitch_change_all,
                ],
                outputs=[
                    song_cover_track,
                    instrumentals_shifted_track,
                    backup_vocals_shifted_track,
                ],
            ),
            EventArgs(
                partial(
                    _mix_song_cover_harness,
                    percentages=percentages[13:15],
                ),
                inputs=[
                    postprocessed_vocals_track,
                    instrumentals_track,
                    backup_vocals_track,
                    instrumentals_shifted_track,
                    backup_vocals_shifted_track,
                    current_song_dir,
                    main_gain,
                    inst_gain,
                    backup_gain,
                    output_sr,
                    output_format,
                    output_name,
                ],
                outputs=[song_cover_track],
            ),
            EventArgs(
                partial(update_cached_input_songs, 3 + len(song_dir_dropdowns)),
                outputs=[
                    cached_input_songs_dropdown,
                    intermediate_audio_to_delete,
                    cached_input_songs_dropdown2,
                ]
                + song_dir_dropdowns,
                name="then",
                show_progress="hidden",
            ),
        ]
        setup_consecutive_event_listeners_with_toggled_interactivity(
            generate_btn2,
            generate_btn2_event_args_list,
            generate_buttons + [show_intermediate_files, clear_btn],
        )
        clear_btn.click(
            lambda: [
                0,
                0,
                0.5,
                3,
                0.25,
                0.33,
                "rmvpe",
                128,
                0.15,
                0.2,
                0.8,
                0.7,
                0,
                0,
                0,
                44100,
                "mp3",
                False,
            ],
            outputs=[
                pitch_change_vocals,
                pitch_change_all,
                index_rate,
                filter_radius,
                rms_mix_rate,
                protect,
                f0_method,
                crepe_hop_length,
                reverb_rm_size,
                reverb_wet,
                reverb_dry,
                reverb_damping,
                main_gain,
                inst_gain,
                backup_gain,
                output_sr,
                output_format,
                show_intermediate_files,
            ],
            show_progress="hidden",
        )
