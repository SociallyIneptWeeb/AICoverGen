"""Module which defines the code for the "Multi-step generation" tab."""

from typing import TYPE_CHECKING, Any

from collections.abc import Sequence
from functools import partial

import gradio as gr

from typing_extra import AudioExt, F0Method, SampleRate, SegmentSize, SeparationModel

from backend.generate_song_cover import (
    convert,
    mix_song,
    pitch_shift,
    postprocess,
    retrieve_song,
    separate_audio,
)

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

if TYPE_CHECKING:
    from frontend.typing_extra import UpdateAudioKwArgs


def _update_audio(
    num_components: int,
    output_indices: Sequence[int],
    track: str | None,
    disallow_none: bool = True,
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
    disallow_none : bool, default=True
        Whether to disallow the value of the indexed components to be
        `None`.

    Returns
    -------
    gr.Audio | tuple[gr.Audio, ...]
        Each `Audio` component under consideration with the value of the
        indexed components updated to the given audio track.

    """
    update_args_list: list[UpdateAudioKwArgs] = [{} for _ in range(num_components)]
    for index in output_indices:
        if track or not disallow_none:
            update_args_list[index]["value"] = track
    match update_args_list:
        case [update_args]:
            return gr.Audio(**update_args)
        case _:
            return tuple(gr.Audio(**update_args) for update_args in update_args_list)


def _pair_audio_tracks_and_gain(
    audio_components: Sequence[gr.Audio],
    gain_components: Sequence[gr.Slider],
    data: dict[gr.Audio | gr.Slider, Any],
) -> list[tuple[str, int]]:
    """
    Pair audio tracks and gain levels stored in separate gradio
    components.

    This function is meant to first be partially applied to the sequence
    of audio components and the sequence of slider components containing
    the values that should be combined. The resulting function can then
    be called by an event listener whose inputs is a set containing
    those audio and slider components. The `data` parameter in that case
    will contain a mapping from each of those components to the value
    that the component stores.

    Parameters
    ----------
    audio_components : Sequence[gr.Audio]
        Audio components to pair with gain levels.
    gain_components : Sequence[gr.Slider]
        Gain level components to pair with audio tracks.
    data : dict[gr.Audio | gr.Slider, Any]
        Data from the audio and gain components.

    Returns
    -------
    list[tuple[str, int]]
        Paired audio tracks and gain levels.

    Raises
    ------
    ValueError
        If the number of audio tracks and gain levels are not the same.

    """
    audio_tracks = [data[component] for component in audio_components]
    gain_levels = [data[component] for component in gain_components]
    if len(audio_tracks) != len(gain_levels):
        err_msg = "Number of audio tracks and gain levels must be the same."
        raise ValueError(err_msg)
    return [
        (audio_track, gain_level)
        for audio_track, gain_level in zip(audio_tracks, gain_levels, strict=True)
        if audio_track
    ]


def render(
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
            separate_audio_dir,
            convert_vocals_dir,
            postprocess_vocals_dir,
            pitch_shift_background_dir,
            mix_dir,
        ) = song_dirs
        current_song_dir = gr.State(None)

        input_tracks = [
            gr.Audio(label=label, type="filepath", render=False)
            for label in [
                "Audio",
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
            audio_track_input,
            vocals_track_input,
            converted_vocals_track_input,
            instrumentals_track_input,
            backup_vocals_track_input,
            main_vocals_track_input,
            shifted_instrumentals_track_input,
            shifted_backup_vocals_track_input,
        ) = input_tracks

        (
            song_output,
            primary_stem_output,
            secondary_stem_output,
            converted_vocals_track_output,
            effected_vocals_track_output,
            shifted_instrumentals_track_output,
            shifted_backup_vocals_track_output,
            song_cover_output,
        ) = [
            gr.Audio(label=label, type="filepath", interactive=False, render=False)
            for label in [
                "Song",
                "Primary stem",
                "Secondary stem",
                "Converted vocals",
                "Effected vocals",
                "Pitch-shifted instrumentals",
                "Pitch-shifted backup vocals",
                "Song cover",
            ]
        ]

        transfer_defaults = [
            ["Step 1: audio"],
            ["Step 4: instrumentals"],
            ["Step 2: vocals"],
            ["Step 3: vocals"],
            ["Step 5: main vocals"],
            ["Step 5: instrumentals"],
            ["Step 5: backup vocals"],
            [],
        ]

        (
            song_transfer_default,
            primary_stem_transfer_default,
            secondary_stem_transfer_default,
            converted_vocals_transfer_default,
            effected_vocals_transfer_default,
            shifted_instrumentals_transfer_default,
            shifted_backup_vocals_transfer_default,
            song_cover_transfer_default,
        ) = transfer_defaults

        (
            song_transfer,
            primary_stem_transfer,
            secondary_stem_transfer,
            converted_vocals_transfer,
            effected_vocals_transfer,
            shifted_instrumentals_transfer,
            shifted_backup_vocals_transfer,
            song_cover_transfer,
        ) = [
            gr.Dropdown(
                [
                    "Step 1: audio",
                    "Step 2: vocals",
                    "Step 3: vocals",
                    "Step 4: instrumentals",
                    "Step 4: backup vocals",
                    "Step 5: main vocals",
                    "Step 5: instrumentals",
                    "Step 5: backup vocals",
                ],
                label=f"{label_prefix} destination",
                info=(
                    "Select the input track(s) to transfer the"
                    f" {label_prefix.lower()} to when the 'Transfer"
                    f" {label_prefix.lower()}' button is clicked."
                ),
                render=False,
                type="index",
                multiselect=True,
                value=value,
            )
            for value, label_prefix in zip(
                transfer_defaults,
                [
                    "Song",
                    "Primary stem",
                    "Secondary stem",
                    "Converted vocals",
                    "Effected vocals",
                    "Pitch-shifted instrumentals",
                    "Pitch-shifted backup vocals",
                    "Song cover",
                ],
                strict=True,
            )
        ]

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
            gr.Markdown("**Settings**")
            song_transfer.render()
            gr.Markdown("**Outputs**")
            song_output.render()
            gr.Markdown("**Controls**")
            retrieve_song_btn = gr.Button("Retrieve song", variant="primary")
            song_transfer_btn = gr.Button("Transfer song")
            retrieve_song_reset_btn = gr.Button("Reset settings")

            retrieve_song_reset_btn.click(
                lambda: gr.Dropdown(value=song_transfer_default),
                outputs=song_transfer,
                show_progress="hidden",
            )
            retrieve_song_btn.click(
                partial(
                    exception_harness(retrieve_song),
                    progress_bar=PROGRESS_BAR,
                ),
                inputs=source,
                outputs=[song_output, current_song_dir],
            ).then(
                partial(
                    update_cached_songs,
                    len(song_dirs) + 2,
                    value_indices=range(len(song_dirs)),
                ),
                inputs=current_song_dir,
                outputs=([*song_dirs, cached_song_multi, cached_song_1click]),
                show_progress="hidden",
            ).then(
                partial(update_cached_songs, 1, [], [0]),
                outputs=intermediate_audio,
                show_progress="hidden",
            )

        with gr.Accordion("Step 1: vocal separation", open=False):
            gr.Markdown("")
            gr.Markdown("**Inputs**")
            audio_track_input.render()
            separate_audio_dir.render()
            gr.Markdown("**Settings**")
            with gr.Row():
                separation_model = gr.Dropdown(
                    list(SeparationModel),
                    value=SeparationModel.UVR_MDX_NET_VOC_FT,
                    label="Separation model",
                    info="The model to use for audio separation.",
                )
                segment_size = gr.Radio(
                    list(SegmentSize),
                    value=SegmentSize.SEG_512,
                    label="Segment size",
                    info=(
                        "Size of segments into which the audio is split. Larger"
                        " consumes more resources, but may give better results."
                    ),
                )
            with gr.Row():
                primary_stem_transfer.render()
                secondary_stem_transfer.render()

            gr.Markdown("**Outputs**")
            with gr.Row():
                primary_stem_output.render()
                secondary_stem_output.render()
            gr.Markdown("**Controls**")
            separate_vocals_btn = gr.Button("Separate vocals", variant="primary")
            with gr.Row():
                primary_stem_transfer_btn = gr.Button("Transfer primary stem")
                secondary_stem_transfer_btn = gr.Button("Transfer secondary stem")
            separate_audio_reset_btn = gr.Button("Reset settings")

            separate_audio_reset_btn.click(
                lambda: [
                    SeparationModel.UVR_MDX_NET_VOC_FT,
                    SegmentSize.SEG_512,
                    gr.Dropdown(value=primary_stem_transfer_default),
                    gr.Dropdown(value=secondary_stem_transfer_default),
                ],
                outputs=[
                    separation_model,
                    segment_size,
                    primary_stem_transfer,
                    secondary_stem_transfer,
                ],
                show_progress="hidden",
            )
            separate_vocals_btn.click(
                partial(
                    exception_harness(separate_audio),
                    progress_bar=PROGRESS_BAR,
                ),
                inputs=[
                    audio_track_input,
                    separate_audio_dir,
                    separation_model,
                    segment_size,
                ],
                outputs=[primary_stem_output, secondary_stem_output],
                concurrency_limit=1,
                concurrency_id=ConcurrencyId.GPU,
            )
        with gr.Accordion("Step 2: vocal conversion", open=False):
            gr.Markdown("")
            gr.Markdown("**Inputs**")
            vocals_track_input.render()
            with gr.Row():
                convert_vocals_dir.render()
                model_multi.render()
            gr.Markdown("**Settings**")
            with gr.Row():
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
                    label="Hop length",
                    info=(
                        "How often the CREPE-based pitch detection algorithm checks"
                        " for pitch changes. Measured in milliseconds. Lower values"
                        " lead to longer conversion times and a higher risk of"
                        " voice cracks, but better pitch accuracy."
                    ),
                )

            converted_vocals_transfer.render()
            gr.Markdown("**Outputs**")
            converted_vocals_track_output.render()
            gr.Markdown("**Controls**")
            convert_vocals_btn = gr.Button("Convert vocals", variant="primary")
            converted_vocals_transfer_btn = gr.Button("Transfer converted vocals")
            convert_vocals_reset_btn = gr.Button("Reset settings")

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
            convert_vocals_btn.click(
                partial(
                    exception_harness(convert),
                    progress_bar=PROGRESS_BAR,
                ),
                inputs=[
                    vocals_track_input,
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
                outputs=converted_vocals_track_output,
                concurrency_id=ConcurrencyId.GPU,
                concurrency_limit=1,
            )
        with gr.Accordion("Step 3: vocal post-processing", open=False):
            gr.Markdown("")
            gr.Markdown("**Inputs**")
            converted_vocals_track_input.render()
            postprocess_vocals_dir.render()
            gr.Markdown("**Settings**")
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

            effected_vocals_transfer.render()
            gr.Markdown("**Outputs**")

            effected_vocals_track_output.render()
            gr.Markdown("**Controls**")
            postprocess_vocals_btn = gr.Button(
                "Post-process vocals",
                variant="primary",
            )
            effected_vocals_transfer_btn = gr.Button("Transfer effected vocals")
            postprocess_vocals_reset_btn = gr.Button("Reset settings")

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
            postprocess_vocals_btn.click(
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
                outputs=effected_vocals_track_output,
            )
        with gr.Accordion("Step 4: pitch shift of background audio", open=False):
            gr.Markdown("")
            gr.Markdown("**Inputs**")
            with gr.Row():
                instrumentals_track_input.render()
                backup_vocals_track_input.render()
            pitch_shift_background_dir.render()
            gr.Markdown("**Settings**")
            with gr.Row():
                n_semitones_instrumentals = gr.Slider(
                    -12,
                    12,
                    value=0,
                    step=1,
                    label="Instrumental pitch shift",
                    info="The number of semi-tones to pitch-shift the instrumentals by",
                )
                n_semitones_backup_vocals = gr.Slider(
                    -12,
                    12,
                    value=0,
                    step=1,
                    label="Backup vocal pitch shift",
                    info="The number of semi-tones to pitch-shift the backup vocals by",
                )
            with gr.Row():
                shifted_instrumentals_transfer.render()
                shifted_backup_vocals_transfer.render()

            gr.Markdown("**Outputs**")
            with gr.Row():
                shifted_instrumentals_track_output.render()
                shifted_backup_vocals_track_output.render()
            gr.Markdown("**Controls**")
            with gr.Row():
                pitch_shift_instrumentals_btn = gr.Button(
                    "Pitch shift instrumentals",
                    variant="primary",
                )
                pitch_shift_backup_vocals_btn = gr.Button(
                    "Pitch shift backup vocals",
                    variant="primary",
                )
            with gr.Row():
                shifted_instrumentals_transfer_btn = gr.Button(
                    "Transfer pitch-shifted instrumentals",
                )
                shifted_backup_vocals_transfer_btn = gr.Button(
                    "Transfer pitch-shifted backup vocals",
                )
            pitch_shift_background_reset_btn = gr.Button("Reset settings")

            pitch_shift_background_reset_btn.click(
                lambda: [
                    0,
                    0,
                    gr.Dropdown(value=shifted_instrumentals_transfer_default),
                    gr.Dropdown(value=shifted_backup_vocals_transfer_default),
                ],
                outputs=[
                    n_semitones_instrumentals,
                    n_semitones_backup_vocals,
                    shifted_instrumentals_transfer,
                    shifted_backup_vocals_transfer,
                ],
                show_progress="hidden",
            )
            pitch_shift_instrumentals_btn.click(
                partial(
                    exception_harness(pitch_shift),
                    progress_bar=PROGRESS_BAR,
                    display_msg="Pitch shifting instrumentals...",
                ),
                inputs=[
                    instrumentals_track_input,
                    pitch_shift_background_dir,
                    n_semitones_instrumentals,
                ],
                outputs=shifted_instrumentals_track_output,
            )
            pitch_shift_backup_vocals_btn.click(
                partial(
                    exception_harness(pitch_shift),
                    progress_bar=PROGRESS_BAR,
                    display_msg="Pitch shifting backup vocals...",
                ),
                inputs=[
                    backup_vocals_track_input,
                    pitch_shift_background_dir,
                    n_semitones_backup_vocals,
                ],
                outputs=shifted_backup_vocals_track_output,
            )
        with gr.Accordion("Step 5: song mixing", open=False):
            gr.Markdown("")
            gr.Markdown("**Inputs**")
            with gr.Row():
                main_vocals_track_input.render()
                shifted_instrumentals_track_input.render()
                shifted_backup_vocals_track_input.render()
            mix_dir.render()
            gr.Markdown("**Settings**")
            with gr.Row():
                main_gain = gr.Slider(
                    -20,
                    20,
                    value=0,
                    step=1,
                    label="Main gain",
                    info="The gain to apply to the main vocals.",
                )
                inst_gain = gr.Slider(
                    -20,
                    20,
                    value=0,
                    step=1,
                    label="Instrumentals gain",
                    info="The gain to apply to the instrumentals.",
                )
                backup_gain = gr.Slider(
                    -20,
                    20,
                    value=0,
                    step=1,
                    label="Backup gain",
                    info="The gain to apply to the backup vocals.",
                )
            with gr.Row():
                output_name = gr.Textbox(
                    value=update_song_cover_name,
                    inputs=[main_vocals_track_input, mix_dir],
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
            song_cover_transfer.render()
            gr.Markdown("**Outputs**")
            song_cover_output.render()
            gr.Markdown("**Controls**")
            mix_btn = gr.Button("Mix song cover", variant="primary")
            song_cover_transfer_btn = gr.Button("Transfer song cover")
            mix_reset_btn = gr.Button("Reset settings")

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
            temp_audio_gains = gr.State()
            mix_btn.click(
                partial(
                    _pair_audio_tracks_and_gain,
                    [
                        main_vocals_track_input,
                        shifted_instrumentals_track_input,
                        shifted_backup_vocals_track_input,
                    ],
                    [main_gain, inst_gain, backup_gain],
                ),
                inputs={
                    main_vocals_track_input,
                    shifted_instrumentals_track_input,
                    shifted_backup_vocals_track_input,
                    main_gain,
                    inst_gain,
                    backup_gain,
                },
                outputs=temp_audio_gains,
            ).then(
                partial(exception_harness(mix_song), progress_bar=PROGRESS_BAR),
                inputs=[
                    temp_audio_gains,
                    mix_dir,
                    output_sr,
                    output_format,
                    output_name,
                ],
                outputs=song_cover_output,
            ).then(
                partial(update_output_audio, 1, [], [0]),
                outputs=output_audio,
                show_progress="hidden",
            )

        for btn, transfer, output in [
            (song_transfer_btn, song_transfer, song_output),
            (primary_stem_transfer_btn, primary_stem_transfer, primary_stem_output),
            (
                secondary_stem_transfer_btn,
                secondary_stem_transfer,
                secondary_stem_output,
            ),
            (
                converted_vocals_transfer_btn,
                converted_vocals_transfer,
                converted_vocals_track_output,
            ),
            (
                effected_vocals_transfer_btn,
                effected_vocals_transfer,
                effected_vocals_track_output,
            ),
            (
                shifted_instrumentals_transfer_btn,
                shifted_instrumentals_transfer,
                shifted_instrumentals_track_output,
            ),
            (
                shifted_backup_vocals_transfer_btn,
                shifted_backup_vocals_transfer,
                shifted_backup_vocals_track_output,
            ),
            (song_cover_transfer_btn, song_cover_transfer, song_cover_output),
        ]:
            btn.click(
                partial(_update_audio, len(input_tracks)),
                inputs=[transfer, output],
                outputs=input_tracks,
                show_progress="hidden",
            )
