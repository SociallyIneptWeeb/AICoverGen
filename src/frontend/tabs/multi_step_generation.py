"""Module which defines the code for the "Multi-step generation" tab."""

from typing import TYPE_CHECKING

from collections.abc import Sequence
from functools import partial

import gradio as gr

from typing_extra import AudioExt, F0Method, SampleRate

from backend.generate_song_cover import (
    convert,
    dereverb,
    mix_song_cover,
    pitch_shift_background,
    postprocess,
    retrieve_song,
    separate_main_vocals,
    separate_vocals,
)

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
from frontend.typing_extra import SourceType

if TYPE_CHECKING:
    from frontend.typing_extra import UpdateAudioKwArgs


def _update_audio(
    num_components: int,
    output_indices: Sequence[int],
    track: str,
) -> gr.Audio | tuple[gr.Audio, ...]:
    """
    Update the value of a subset of `Audio` components to the given
    audio track.

    Parameters
    ----------
    num_components : int
        The total number of `Audio` components under consideration.
    output_indices : Sequence[int]
        Indices of `Audio` components to update the value for.
    track : str
        Path pointing to an audio track to update the value of the
        indexed `Audio` components with.

    Returns
    -------
    gr.Audio | tuple[gr.Audio, ...]
        Each `Audio` component under consideration with the value of the
        indexed components updated to the given audio track.

    """
    update_args_list: list[UpdateAudioKwArgs] = [{} for _ in range(num_components)]
    for index in output_indices:
        update_args_list[index]["value"] = track
    match update_args_list:
        case [update_args]:
            return gr.Audio(**update_args)
        case _:
            return tuple(gr.Audio(**update_args) for update_args in update_args_list)


def render(
    generate_btns: Sequence[gr.Button],
    song_dirs: Sequence[gr.Dropdown],
    cached_song_1click: gr.Dropdown,
    cached_song_multi: gr.Dropdown,
    model_multi: gr.Dropdown,
    intermediate_audio: gr.Dropdown,
    output_audio: gr.Dropdown,
) -> None:
    """
    Render "Multi-step generation" tab.

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
        "One-click generation" tab.
    cached_song_multi : gr.Dropdown
        Dropdown for selecting a cached song in the
        "Multi-step generation" tab.
    model_multi : gr.Dropdown
        Dropdown for selecting a voice model in the
        "Multi-step generation" tab.
    intermediate_audio : gr.Dropdown
        Dropdown for selecting intermediate audio files to delete in the
        "Delete audio" tab.
    output_audio : gr.Dropdown
        Dropdown for selecting output audio files to delete in the
        "Delete audio" tab.

    """
    with gr.Tab("Multi-step generation"):
        (
            retrieve_song_btn,
            separate_vocals_btn,
            separate_main_vocals_btn,
            dereverb_vocals_btn,
            convert_vocals_btn,
            postprocess_vocals_btn,
            pitch_shift_background_btn,
            mix_btn,
            _,
        ) = generate_btns
        (
            separate_vocals_dir,
            separate_main_vocals_dir,
            dereverb_vocals_dir,
            convert_vocals_dir,
            postprocess_vocals_dir,
            pitch_shift_background_dir,
            mix_dir,
        ) = song_dirs
        current_song_dir = gr.State(None)

        input_tracks = [
            gr.Audio(label=label, type="filepath", render=False)
            for label in [
                "Song",
                "Vocals",
                "Vocals",
                "Vocals",
                "Vocals",
                "Instrumentals",
                "Backup vocals",
                "Main vocals",
                "Instrumentals",
                "Backup vocals",
            ]
        ]
        (
            song_input,
            vocals_track_input,
            main_vocals_track_input,
            dereverbed_vocals_track_input,
            converted_vocals_track_input,
            instrumentals_track_input,
            backup_vocals_track_input,
            effected_vocals_track_input,
            shifted_instrumentals_track_input,
            shifted_backup_vocals_track_input,
        ) = input_tracks

        (
            song_output,
            vocals_track_output,
            instrumentals_track_output,
            main_vocals_track_output,
            backup_vocals_track_output,
            dereverbed_vocals_track_output,
            reverb_track_output,
            converted_vocals_track_output,
            effected_vocals_track_output,
            shifted_instrumentals_track_output,
            shifted_backup_vocals_track_output,
            song_cover_output,
        ) = [
            gr.Audio(label=label, type="filepath", interactive=False, render=False)
            for label in [
                "Song",
                "Vocals",
                "Instrumentals",
                "Main vocals",
                "Backup vocals",
                "De-reverbed vocals",
                "Reverb",
                "Converted vocals",
                "Post-processed vocals",
                "Pitch-shifted instrumentals",
                "Pitch-shifted backup vocals",
                "Song cover",
            ]
        ]

        transfer_defaults = [
            ["Step 1: song"],
            ["Step 2: vocals"],
            ["Step 6: instrumentals"],
            ["Step 3: vocals"],
            ["Step 6: backup vocals"],
            ["Step 4: vocals"],
            [],
            ["Step 5: vocals"],
            ["Step 7: main vocals"],
            ["Step 7: instrumentals"],
            ["Step 7: backup vocals"],
            [],
        ]

        (
            song_transfer_default,
            vocals_transfer_default,
            instrumentals_transfer_default,
            main_vocals_transfer_default,
            backup_vocals_transfer_default,
            dereverbed_vocals_transfer_default,
            reverb_transfer_default,
            converted_vocals_transfer_default,
            effected_vocals_transfer_default,
            shifted_instrumentals_transfer_default,
            shifted_backup_vocals_transfer_default,
            song_cover_transfer_default,
        ) = transfer_defaults

        (
            song_transfer,
            vocals_transfer,
            instrumentals_transfer,
            main_vocals_transfer,
            backup_vocals_transfer,
            dereverbed_vocals_transfer,
            reverb_transfer,
            converted_vocals_transfer,
            effected_vocals_transfer,
            shifted_instrumentals_transfer,
            shifted_backup_vocals_transfer,
            song_cover_transfer,
        ) = [
            gr.Dropdown(
                [
                    "Step 1: song",
                    "Step 2: vocals",
                    "Step 3: vocals",
                    "Step 4: vocals",
                    "Step 5: vocals",
                    "Step 6: instrumentals",
                    "Step 6: backup vocals",
                    "Step 7: main vocals",
                    "Step 7: instrumentals",
                    "Step 7: backup vocals",
                ],
                label="Transfer to",
                info=(
                    "Select the input track(s) to transfer the output track to once"
                    " generation completes."
                ),
                render=False,
                type="index",
                multiselect=True,
                value=value,
            )
            for value in transfer_defaults
        ]

        (
            retrieve_song_reset_btn,
            separate_vocals_reset_btn,
            separate_main_vocals_reset_btn,
            dereverb_vocals_reset_btn,
            convert_vocals_reset_btn,
            postprocess_vocals_reset_btn,
            pitch_shift_background_reset_btn,
            mix_reset_btn,
        ) = [gr.Button(value="Reset settings", render=False) for _ in range(8)]

        with gr.Accordion("Step 0: song retrieval", open=True):
            gr.Markdown("")
            gr.Markdown("**Inputs**")
            with gr.Row():
                with gr.Column():
                    source_type = gr.Dropdown(
                        list(SourceType),
                        value=SourceType.PATH,
                        label="Source type",
                        type="index",
                        info="The type of source to retrieve a song from.",
                    )
                with gr.Column():
                    source = gr.Textbox(
                        label="Source",
                        info=(
                            "Link to a song on YouTube or the full path of a local"
                            " audio file."
                        ),
                    )
                    local_file = gr.Audio(
                        label="Source",
                        type="filepath",
                        visible=False,
                    )
                    cached_song_multi.render()

                source_type.input(
                    partial(toggle_visible_component, 3),
                    inputs=source_type,
                    outputs=[source, local_file, cached_song_multi],
                    show_progress="hidden",
                )

                local_file.change(
                    update_value,
                    inputs=local_file,
                    outputs=source,
                    show_progress="hidden",
                )
                cached_song_multi.input(
                    update_value,
                    inputs=cached_song_multi,
                    outputs=source,
                    show_progress="hidden",
                )
            gr.Markdown("**Outputs**")
            song_output.render()
            song_transfer.render()
            retrieve_song_reset_btn.render()
            retrieve_song_reset_btn.click(
                lambda: gr.Dropdown(value=song_transfer_default),
                outputs=song_transfer,
                show_progress="hidden",
            )

            retrieve_song_btn.render()

            retrieve_song_event_args_list = [
                EventArgs(
                    partial(
                        exception_harness(retrieve_song),
                        progress_bar=PROGRESS_BAR,
                    ),
                    inputs=[source],
                    outputs=[song_output, current_song_dir],
                ),
                EventArgs(
                    partial(
                        update_cached_songs,
                        len(song_dirs) + 2,
                        value_indices=range(len(song_dirs)),
                    ),
                    inputs=[current_song_dir],
                    outputs=([*song_dirs, cached_song_multi, cached_song_1click]),
                    name="then",
                    show_progress="hidden",
                ),
                EventArgs(
                    partial(update_cached_songs, 1, [], [0]),
                    outputs=[intermediate_audio],
                    name="then",
                    show_progress="hidden",
                ),
                EventArgs(
                    partial(_update_audio, len(input_tracks)),
                    inputs=[song_transfer, song_output],
                    outputs=input_tracks,
                    name="then",
                    show_progress="hidden",
                ),
            ]
            chain_event_listeners(
                retrieve_song_btn,
                retrieve_song_event_args_list,
                generate_btns,
            )
        with gr.Accordion("Step 1: vocals/instrumentals separation", open=False):
            gr.Markdown("")
            gr.Markdown("**Inputs**")
            song_input.render()
            separate_vocals_dir.render()
            gr.Markdown("**Outputs**")
            with gr.Row():
                with gr.Column():
                    vocals_track_output.render()
                    vocals_transfer.render()

                with gr.Column():
                    instrumentals_track_output.render()
                    instrumentals_transfer.render()

            separate_vocals_reset_btn.render()
            separate_vocals_reset_btn.click(
                lambda: [
                    gr.Dropdown(value=vocals_transfer_default),
                    gr.Dropdown(value=instrumentals_transfer_default),
                ],
                outputs=[vocals_transfer, instrumentals_transfer],
                show_progress="hidden",
            )
            separate_vocals_btn.render()

            separate_vocals_event_args_list = [
                EventArgs(
                    partial(
                        exception_harness(separate_vocals),
                        progress_bar=PROGRESS_BAR,
                    ),
                    inputs=[song_input, separate_vocals_dir],
                    outputs=[vocals_track_output, instrumentals_track_output],
                ),
                *(
                    EventArgs(
                        partial(_update_audio, len(input_tracks)),
                        inputs=[transfer, output_track],
                        outputs=input_tracks,
                        name="then",
                        show_progress="hidden",
                    )
                    for transfer, output_track in [
                        (vocals_transfer, vocals_track_output),
                        (instrumentals_transfer, instrumentals_track_output),
                    ]
                ),
            ]
            chain_event_listeners(
                separate_vocals_btn,
                separate_vocals_event_args_list,
                generate_btns,
            )

        with gr.Accordion("Step 2: main vocals/ backup vocals separation", open=False):
            gr.Markdown("")
            gr.Markdown("**Inputs**")
            vocals_track_input.render()
            separate_main_vocals_dir.render()
            gr.Markdown("**Outputs**")
            with gr.Row():
                with gr.Column():
                    main_vocals_track_output.render()
                    main_vocals_transfer.render()
                with gr.Column():
                    backup_vocals_track_output.render()
                    backup_vocals_transfer.render()

            separate_main_vocals_reset_btn.render()
            separate_main_vocals_reset_btn.click(
                lambda: [
                    gr.Dropdown(value=main_vocals_transfer_default),
                    gr.Dropdown(value=backup_vocals_transfer_default),
                ],
                outputs=[main_vocals_transfer, backup_vocals_transfer],
                show_progress="hidden",
            )
            separate_main_vocals_btn.render()

            separate_main_vocals_event_args_list = [
                EventArgs(
                    partial(
                        exception_harness(separate_main_vocals),
                        progress_bar=PROGRESS_BAR,
                    ),
                    inputs=[vocals_track_input, separate_main_vocals_dir],
                    outputs=[main_vocals_track_output, backup_vocals_track_output],
                ),
                *(
                    EventArgs(
                        partial(_update_audio, len(input_tracks)),
                        inputs=[transfer, output_track],
                        outputs=input_tracks,
                        name="then",
                        show_progress="hidden",
                    )
                    for transfer, output_track in [
                        (main_vocals_transfer, main_vocals_track_output),
                        (backup_vocals_transfer, backup_vocals_track_output),
                    ]
                ),
            ]

            chain_event_listeners(
                separate_main_vocals_btn,
                separate_main_vocals_event_args_list,
                generate_btns,
            )

        with gr.Accordion("Step 3: vocal cleanup", open=False):
            gr.Markdown("")
            gr.Markdown("**Inputs**")
            main_vocals_track_input.render()
            dereverb_vocals_dir.render()
            gr.Markdown("**Outputs**")
            with gr.Row():
                with gr.Column():
                    dereverbed_vocals_track_output.render()
                    dereverbed_vocals_transfer.render()
                with gr.Column():
                    reverb_track_output.render()
                    reverb_transfer.render()

            dereverb_vocals_reset_btn.render()
            dereverb_vocals_reset_btn.click(
                lambda: [
                    gr.Dropdown(value=dereverbed_vocals_transfer_default),
                    gr.Dropdown(value=reverb_transfer_default),
                ],
                outputs=[dereverbed_vocals_transfer, reverb_transfer],
                show_progress="hidden",
            )
            dereverb_vocals_btn.render()
            dereverb_vocals_event_args_list = [
                EventArgs(
                    partial(
                        exception_harness(dereverb),
                        progress_bar=PROGRESS_BAR,
                    ),
                    inputs=[main_vocals_track_input, dereverb_vocals_dir],
                    outputs=[dereverbed_vocals_track_output, reverb_track_output],
                ),
                *(
                    EventArgs(
                        partial(_update_audio, len(input_tracks)),
                        inputs=[transfer, output_track],
                        outputs=input_tracks,
                        name="then",
                        show_progress="hidden",
                    )
                    for transfer, output_track in [
                        (dereverbed_vocals_transfer, dereverbed_vocals_track_output),
                        (reverb_transfer, reverb_track_output),
                    ]
                ),
            ]

            chain_event_listeners(
                dereverb_vocals_btn,
                dereverb_vocals_event_args_list,
                generate_btns,
            )
        with gr.Accordion("Step 4: vocal conversion", open=False):
            gr.Markdown("")
            gr.Markdown("**Inputs**")
            dereverbed_vocals_track_input.render()
            convert_vocals_dir.render()
            with gr.Row():
                model_multi.render()
                n_octaves = gr.Slider(
                    -3,
                    3,
                    value=0,
                    step=1,
                    label="Pitch shift (octaves)",
                    info=(
                        "The number of octaves to pitch-shift the converted vocals by."
                        " Use 1 for male-to-female and -1 for vice-versa."
                    ),
                )
                n_semitones = gr.Slider(
                    -12,
                    12,
                    value=0,
                    step=1,
                    label="Pitch shift (semi-tones)",
                    info=(
                        "The number of semi-tones to pitch-shift the converted vocals"
                        " by. Altering this slightly reduces sound quality."
                    ),
                )
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
                with gr.Column():
                    f0_method = gr.Dropdown(
                        list(F0Method),
                        value=F0Method.RMVPE,
                        label="Pitch detection algorithm",
                        info=(
                            "The method to use for pitch detection. Best option is"
                            " RMVPE (clarity in vocals), then Mangio-CREPE (smoother"
                            " vocals)."
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
                            "How often the CREPE-based pitch detection algorithm checks"
                            " for pitch changes. Measured in milliseconds. Lower values"
                            " lead to longer conversion times and a higher risk of"
                            " voice cracks, but better pitch accuracy."
                        ),
                    )
                    f0_method.change(
                        show_hop_slider,
                        inputs=f0_method,
                        outputs=hop_length,
                        show_progress="hidden",
                    )

            gr.Markdown("**Outputs**")
            converted_vocals_track_output.render()
            converted_vocals_transfer.render()
            convert_vocals_reset_btn.render()
            convert_vocals_reset_btn.click(
                lambda: [
                    0,
                    0,
                    0.5,
                    3,
                    0.25,
                    0.33,
                    F0Method.RMVPE,
                    128,
                    gr.Dropdown(value=converted_vocals_transfer_default),
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
                    converted_vocals_transfer,
                ],
                show_progress="hidden",
            )
            convert_vocals_btn.render()
            convert_vocals_event_args_list = [
                EventArgs(
                    partial(
                        exception_harness(convert),
                        progress_bar=PROGRESS_BAR,
                    ),
                    inputs=[
                        dereverbed_vocals_track_input,
                        convert_vocals_dir,
                        model_multi,
                        n_octaves,
                        n_semitones,
                        f0_method,
                        index_rate,
                        filter_radius,
                        rms_mix_rate,
                        protect,
                        hop_length,
                    ],
                    outputs=[converted_vocals_track_output],
                ),
                EventArgs(
                    partial(_update_audio, len(input_tracks)),
                    inputs=[
                        converted_vocals_transfer,
                        converted_vocals_track_output,
                    ],
                    outputs=input_tracks,
                    name="then",
                    show_progress="hidden",
                ),
            ]
            chain_event_listeners(
                convert_vocals_btn,
                convert_vocals_event_args_list,
                generate_btns,
            )
        with gr.Accordion("Step 5: vocal post-processing", open=False):
            gr.Markdown("")
            gr.Markdown("**Inputs**")
            converted_vocals_track_input.render()
            postprocess_vocals_dir.render()
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
            gr.Markdown("**Outputs**")

            effected_vocals_track_output.render()
            effected_vocals_transfer.render()

            postprocess_vocals_reset_btn.render()
            postprocess_vocals_reset_btn.click(
                lambda: [
                    0.15,
                    0.2,
                    0.8,
                    0.7,
                    gr.Dropdown(value=effected_vocals_transfer_default),
                ],
                outputs=[
                    room_size,
                    wet_level,
                    dry_level,
                    damping,
                    effected_vocals_transfer,
                ],
                show_progress="hidden",
            )
            postprocess_vocals_btn.render()
            postprocess_vocals_event_args_list = [
                EventArgs(
                    partial(
                        exception_harness(postprocess),
                        progress_bar=PROGRESS_BAR,
                    ),
                    inputs=[
                        converted_vocals_track_input,
                        postprocess_vocals_dir,
                        room_size,
                        wet_level,
                        dry_level,
                        damping,
                    ],
                    outputs=[effected_vocals_track_output],
                ),
                EventArgs(
                    partial(_update_audio, len(input_tracks)),
                    inputs=[
                        effected_vocals_transfer,
                        effected_vocals_track_output,
                    ],
                    outputs=input_tracks,
                    name="then",
                    show_progress="hidden",
                ),
            ]
            chain_event_listeners(
                postprocess_vocals_btn,
                postprocess_vocals_event_args_list,
                generate_btns,
            )
        with gr.Accordion("Step 6: pitch shift of background tracks", open=False):
            gr.Markdown("")
            gr.Markdown("**Inputs**")
            with gr.Row():
                instrumentals_track_input.render()
                backup_vocals_track_input.render()
            pitch_shift_background_dir.render()
            n_semitones_background = gr.Slider(
                -12,
                12,
                value=0,
                step=1,
                label="Pitch shift",
                info=(
                    "The number of semi-tones to pitch-shift the instrumentals and"
                    " backup vocals by."
                ),
            )
            gr.Markdown("**Outputs**")
            with gr.Row():
                with gr.Column():
                    shifted_instrumentals_track_output.render()
                    shifted_instrumentals_transfer.render()
                with gr.Column():
                    shifted_backup_vocals_track_output.render()
                    shifted_backup_vocals_transfer.render()

            pitch_shift_background_reset_btn.render()
            pitch_shift_background_reset_btn.click(
                lambda: [
                    0,
                    gr.Dropdown(value=shifted_instrumentals_transfer_default),
                    gr.Dropdown(value=shifted_backup_vocals_transfer_default),
                ],
                outputs=[
                    n_semitones_background,
                    shifted_instrumentals_transfer,
                    shifted_backup_vocals_transfer,
                ],
                show_progress="hidden",
            )
            pitch_shift_background_btn.render()
            pitch_shift_background_event_args_list = [
                EventArgs(
                    partial(
                        exception_harness(pitch_shift_background),
                        progress_bar=PROGRESS_BAR,
                    ),
                    inputs=[
                        instrumentals_track_input,
                        backup_vocals_track_input,
                        pitch_shift_background_dir,
                        n_semitones_background,
                    ],
                    outputs=[
                        shifted_instrumentals_track_output,
                        shifted_backup_vocals_track_output,
                    ],
                ),
                *(
                    EventArgs(
                        partial(_update_audio, len(input_tracks)),
                        inputs=[transfer, output_track],
                        outputs=input_tracks,
                        name="then",
                        show_progress="hidden",
                    )
                    for transfer, output_track in [
                        (
                            shifted_instrumentals_transfer,
                            shifted_instrumentals_track_output,
                        ),
                        (
                            shifted_backup_vocals_transfer,
                            shifted_backup_vocals_track_output,
                        ),
                    ]
                ),
            ]

            chain_event_listeners(
                pitch_shift_background_btn,
                pitch_shift_background_event_args_list,
                generate_btns,
            )
        with gr.Accordion("Step 7: song mixing", open=False):
            gr.Markdown("")
            gr.Markdown("**Inputs**")
            with gr.Row():
                effected_vocals_track_input.render()
                shifted_instrumentals_track_input.render()
                shifted_backup_vocals_track_input.render()
            mix_dir.render()
            with gr.Row():
                main_gain = gr.Slider(-20, 20, value=0, step=1, label="Main vocals")
                inst_gain = gr.Slider(-20, 20, value=0, step=1, label="Instrumentals")
                backup_gain = gr.Slider(-20, 20, value=0, step=1, label="Backup vocals")
            with gr.Row():
                output_name = gr.Textbox(
                    value=update_song_cover_name,
                    inputs=[effected_vocals_track_input, mix_dir],
                    label="Output name",
                    placeholder="Ultimate RVC song cover",
                    info=(
                        "If no name is provided, a suitable name will be generated"
                        " automatically."
                    ),
                )
                output_sr = gr.Dropdown(
                    choices=list(SampleRate),
                    value=SampleRate.HZ_44100,
                    label="Output sample rate",
                    info="The sample rate to save the generated song in.",
                )
                output_format = gr.Dropdown(
                    list(AudioExt),
                    value=AudioExt.MP3,
                    label="Output format",
                    info="The format to save the generated song in.",
                )
            gr.Markdown("**Outputs**")
            song_cover_output.render()
            song_cover_transfer.render()
            mix_reset_btn.render()
            mix_reset_btn.click(
                lambda: [
                    0,
                    0,
                    0,
                    SampleRate.HZ_44100,
                    AudioExt.MP3,
                    gr.Dropdown(value=song_cover_transfer_default),
                ],
                outputs=[
                    main_gain,
                    inst_gain,
                    backup_gain,
                    output_sr,
                    output_format,
                    song_cover_transfer,
                ],
                show_progress="hidden",
            )
            mix_btn.render()
            mix_btn_event_args_list = [
                EventArgs(
                    partial(
                        exception_harness(mix_song_cover),
                        progress_bar=PROGRESS_BAR,
                    ),
                    inputs=[
                        effected_vocals_track_input,
                        shifted_instrumentals_track_input,
                        shifted_backup_vocals_track_input,
                        mix_dir,
                        main_gain,
                        inst_gain,
                        backup_gain,
                        output_sr,
                        output_format,
                        output_name,
                    ],
                    outputs=[song_cover_output],
                ),
                EventArgs(
                    partial(update_output_audio, 1, [], [0]),
                    outputs=[output_audio],
                    name="then",
                    show_progress="hidden",
                ),
                EventArgs(
                    partial(_update_audio, len(input_tracks)),
                    inputs=[song_cover_transfer, song_cover_output],
                    outputs=input_tracks,
                    name="then",
                    show_progress="hidden",
                ),
            ]

            chain_event_listeners(
                mix_btn,
                mix_btn_event_args_list,
                generate_btns,
            )
