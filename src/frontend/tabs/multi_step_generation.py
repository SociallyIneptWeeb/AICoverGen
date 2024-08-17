from typings.extra import TransferUpdateArgs

from functools import partial

import gradio as gr

from frontend.common import (
    EventArgs,
    setup_consecutive_event_listeners_with_toggled_interactivity,
    exception_harness,
    update_cached_input_songs,
    update_output_audio,
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
)


def _transfer(
    num_components: int, output_indices: list[int], value: str
) -> gr.Audio | tuple[gr.Audio, ...]:
    update_args: list[TransferUpdateArgs] = [{} for _ in range(num_components)]
    for index in output_indices:
        update_args[index]["value"] = value
    if num_components == 1:
        return gr.Audio(**update_args[0])
    return tuple(gr.Audio(**update_arg) for update_arg in update_args)


def render(
    generate_buttons: list[gr.Button],
    song_dir_dropdowns: list[gr.Dropdown],
    cached_input_songs_dropdown: gr.Dropdown,
    cached_input_songs_dropdown2: gr.Dropdown,
    rvc_model: gr.Dropdown,
    intermediate_audio_to_delete: gr.Dropdown,
    output_audio_to_remove: gr.Dropdown,
) -> None:
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
            _,
        ) = generate_buttons
        (
            separate_vocals_dir,
            separate_main_vocals_dir,
            dereverb_vocals_dir,
            convert_vocals_dir,
            postprocess_vocals_dir,
            pitch_shift_background_dir,
            mix_dir,
        ) = song_dir_dropdowns
        current_song_dir = gr.State(None)

        (
            original_track_output,
            vocals_track_output,
            instrumentals_track_output,
            main_vocals_track_output,
            backup_vocals_track_output,
            dereverbed_vocals_track_output,
            reverb_track_output,
            converted_vocals_track_output,
            postprocessed_vocals_track_output,
            shifted_instrumentals_track_output,
            shifted_backup_vocals_track_output,
            song_cover_track,
        ) = [
            gr.Audio(label=label, type="filepath", interactive=False, render=False)
            for label in [
                "Input song",
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
        input_tracks = [
            gr.Audio(label=label, type="filepath", render=False)
            for label in [
                "Input song",
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
            original_track_input,
            vocals_track_input,
            main_vocals_track_input,
            dereverbed_vocals_track_input,
            converted_vocals_track_input,
            instrumentals_track_input,
            backup_vocals_track_input,
            postprocessed_vocals_track_input,
            shifted_instrumentals_track_input,
            shifted_backup_vocals_track_input,
        ) = input_tracks

        transfer_defaults = [
            ["Step 1: input song"],
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
            original_track_transfer_default,
            vocals_track_transfer_default,
            instrumentals_track_transfer_default,
            main_vocals_track_transfer_default,
            backup_vocals_track_transfer_default,
            dereverbed_vocals_track_transfer_default,
            reverb_track_transfer_default,
            converted_vocals_track_transfer_default,
            postprocessed_vocals_track_transfer_default,
            shifted_instrumentals_track_transfer_default,
            shifted_backup_vocals_track_transfer_default,
            song_cover_track_transfer_default,
        ) = transfer_defaults

        transfer_output_track_dropdowns = [
            gr.Dropdown(
                [
                    "Step 1: input song",
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
                info="Select the input track(s) to transfer the output track to once generation completes.",
                render=False,
                type="index",
                multiselect=True,
                value=value,
            )
            for value in transfer_defaults
        ]

        (
            original_track_transfer_dropdown,
            vocals_track_transfer_dropdown,
            instrumentals_track_transfer_dropdown,
            main_vocals_track_transfer_dropdown,
            backup_vocals_track_transfer_dropdown,
            dereverbed_vocals_track_transfer_dropdown,
            reverb_track_transfer_dropdown,
            converted_vocals_track_transfer_dropdown,
            postprocessed_vocals_track_transfer_dropdown,
            shifted_instrumentals_track_transfer_dropdown,
            shifted_backup_vocals_track_transfer_dropdown,
            song_cover_track_transfer_dropdown,
        ) = transfer_output_track_dropdowns

        clear_btns = [gr.Button(value="Reset settings", render=False) for _ in range(8)]
        (
            retrieve_song_clear_btn,
            separate_vocals_clear_btn,
            separate_main_vocals_clear_btn,
            dereverb_vocals_clear_btn,
            convert_vocals_clear_btn,
            postprocess_vocals_clear_btn,
            pitch_shift_background_clear_btn,
            mix_clear_btn,
        ) = clear_btns

        with gr.Accordion("Step 0: song retrieval", open=True):
            gr.Markdown("")
            gr.Markdown("**Inputs**")
            with gr.Row():
                with gr.Column():
                    song_input_type_dropdown = gr.Dropdown(
                        [
                            "YouTube link/local path",
                            "Local file/microphone",
                            "Cached song",
                        ],
                        value="YouTube link/local path",
                        label="Song input type",
                        type="index",
                    )
                with gr.Column():
                    song_input = gr.Text(
                        label="Song input",
                        info="Link to a song on YouTube or the full path of a local audio file.",
                    )
                    local_file = gr.Audio(
                        label="Song input",
                        type="filepath",
                        visible=False,
                    )
                    cached_input_songs_dropdown2.render()

                song_input_type_dropdown.input(
                    partial(toggle_visible_component, 3),
                    inputs=song_input_type_dropdown,
                    outputs=[song_input, local_file, cached_input_songs_dropdown2],
                    show_progress="hidden",
                )

                local_file.change(
                    update_value,
                    inputs=local_file,
                    outputs=song_input,
                    show_progress="hidden",
                )
                cached_input_songs_dropdown2.input(
                    update_value,
                    inputs=cached_input_songs_dropdown2,
                    outputs=song_input,
                    show_progress="hidden",
                )
            gr.Markdown("**Outputs**")
            original_track_output.render()
            original_track_transfer_dropdown.render()
            retrieve_song_clear_btn.render()
            retrieve_song_clear_btn.click(
                lambda: gr.Dropdown(value=original_track_transfer_default),
                outputs=[original_track_transfer_dropdown],
                show_progress="hidden",
            )

            retrieve_song_btn.render()

            retrieve_song_event_args_list = [
                EventArgs(
                    partial(
                        exception_harness(retrieve_song), progress_bar=PROGRESS_BAR
                    ),
                    inputs=[song_input],
                    outputs=[original_track_output, current_song_dir],
                ),
                EventArgs(
                    partial(
                        update_cached_input_songs,
                        len(song_dir_dropdowns) + 2,
                        value_indices=range(len(song_dir_dropdowns) + 1),
                    ),
                    inputs=[current_song_dir],
                    outputs=(
                        song_dir_dropdowns
                        + [
                            cached_input_songs_dropdown2,
                            cached_input_songs_dropdown,
                        ]
                    ),
                    name="then",
                    show_progress="hidden",
                ),
                EventArgs(
                    partial(update_cached_input_songs, 1, [], [0]),
                    outputs=[intermediate_audio_to_delete],
                    name="then",
                    show_progress="hidden",
                ),
                EventArgs(
                    partial(_transfer, len(input_tracks)),
                    inputs=[original_track_transfer_dropdown, original_track_output],
                    outputs=input_tracks,
                    name="then",
                    show_progress="hidden",
                ),
            ]
            setup_consecutive_event_listeners_with_toggled_interactivity(
                retrieve_song_btn,
                retrieve_song_event_args_list,
                generate_buttons,
            )
        with gr.Accordion("Step 1: vocals/instrumentals separation", open=False):
            gr.Markdown("")
            gr.Markdown("**Inputs**")
            original_track_input.render()
            separate_vocals_dir.render()
            gr.Markdown("**Outputs**")
            with gr.Row():
                with gr.Column():
                    vocals_track_output.render()
                    vocals_track_transfer_dropdown.render()

                with gr.Column():
                    instrumentals_track_output.render()
                    instrumentals_track_transfer_dropdown.render()

            separate_vocals_clear_btn.render()
            separate_vocals_clear_btn.click(
                lambda: tuple(
                    gr.Dropdown(value=value)
                    for value in [
                        vocals_track_transfer_default,
                        instrumentals_track_transfer_default,
                    ]
                ),
                outputs=[
                    vocals_track_transfer_dropdown,
                    instrumentals_track_transfer_dropdown,
                ],
                show_progress="hidden",
            )
            separate_vocals_btn.render()

            separate_vocals_event_args_list = [
                EventArgs(
                    partial(
                        exception_harness(separate_vocals), progress_bar=PROGRESS_BAR
                    ),
                    inputs=[original_track_input, separate_vocals_dir],
                    outputs=[vocals_track_output, instrumentals_track_output],
                )
            ] + [
                EventArgs(
                    partial(_transfer, len(input_tracks)),
                    inputs=[transfer_dropdown, output_track],
                    outputs=input_tracks,
                    name="then",
                    show_progress="hidden",
                )
                for transfer_dropdown, output_track in zip(
                    [
                        vocals_track_transfer_dropdown,
                        instrumentals_track_transfer_dropdown,
                    ],
                    [vocals_track_output, instrumentals_track_output],
                )
            ]
            setup_consecutive_event_listeners_with_toggled_interactivity(
                separate_vocals_btn,
                separate_vocals_event_args_list,
                generate_buttons,
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
                    main_vocals_track_transfer_dropdown.render()
                with gr.Column():
                    backup_vocals_track_output.render()
                    backup_vocals_track_transfer_dropdown.render()

            separate_main_vocals_clear_btn.render()
            separate_main_vocals_clear_btn.click(
                lambda: tuple(
                    gr.Dropdown(value=value)
                    for value in [
                        main_vocals_track_transfer_default,
                        backup_vocals_track_transfer_default,
                    ]
                ),
                outputs=[
                    main_vocals_track_transfer_dropdown,
                    backup_vocals_track_transfer_dropdown,
                ],
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
                )
            ] + [
                EventArgs(
                    partial(_transfer, len(input_tracks)),
                    inputs=[transfer_dropdown, output_track],
                    outputs=input_tracks,
                    name="then",
                    show_progress="hidden",
                )
                for transfer_dropdown, output_track in zip(
                    [
                        main_vocals_track_transfer_dropdown,
                        backup_vocals_track_transfer_dropdown,
                    ],
                    [main_vocals_track_output, backup_vocals_track_output],
                )
            ]

            setup_consecutive_event_listeners_with_toggled_interactivity(
                separate_main_vocals_btn,
                separate_main_vocals_event_args_list,
                generate_buttons,
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
                    dereverbed_vocals_track_transfer_dropdown.render()
                with gr.Column():
                    reverb_track_output.render()
                    reverb_track_transfer_dropdown.render()

            dereverb_vocals_clear_btn.render()
            dereverb_vocals_clear_btn.click(
                lambda: tuple(
                    gr.Dropdown(value=value)
                    for value in [
                        dereverbed_vocals_track_transfer_default,
                        reverb_track_transfer_default,
                    ]
                ),
                outputs=[
                    dereverbed_vocals_track_transfer_dropdown,
                    reverb_track_transfer_dropdown,
                ],
                show_progress="hidden",
            )
            dereverb_vocals_btn.render()
            dereverb_vocals_event_args_list = [
                EventArgs(
                    partial(
                        exception_harness(dereverb_vocals), progress_bar=PROGRESS_BAR
                    ),
                    inputs=[main_vocals_track_input, dereverb_vocals_dir],
                    outputs=[dereverbed_vocals_track_output, reverb_track_output],
                )
            ] + [
                EventArgs(
                    partial(_transfer, len(input_tracks)),
                    inputs=[transfer_dropdown, output_track],
                    outputs=input_tracks,
                    name="then",
                    show_progress="hidden",
                )
                for transfer_dropdown, output_track in zip(
                    [
                        dereverbed_vocals_track_transfer_dropdown,
                        reverb_track_transfer_dropdown,
                    ],
                    [dereverbed_vocals_track_output, reverb_track_output],
                )
            ]

            setup_consecutive_event_listeners_with_toggled_interactivity(
                dereverb_vocals_btn,
                dereverb_vocals_event_args_list,
                generate_buttons,
            )
        with gr.Accordion("Step 4: vocal conversion", open=False):
            gr.Markdown("")
            gr.Markdown("**Inputs**")
            dereverbed_vocals_track_input.render()
            convert_vocals_dir.render()
            with gr.Row():
                rvc_model.render()
                pitch_change_octaves = gr.Slider(
                    -3,
                    3,
                    value=0,
                    step=1,
                    label="Pitch shift (octaves)",
                    info="Shift pitch of converted vocals by number of octaves. Generally, use 1 for male-to-female conversions and -1 for vice-versa.",
                )
                pitch_change_semitones = gr.Slider(
                    -12,
                    12,
                    value=0,
                    step=1,
                    label="Pitch shift (semi-tones)",
                    info="Shift pitch of converted vocals by number of semi-tones. Altering this slightly reduces sound quality.",
                )
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

            gr.Markdown("**Outputs**")
            converted_vocals_track_output.render()
            converted_vocals_track_transfer_dropdown.render()
            convert_vocals_clear_btn.render()
            convert_vocals_clear_btn.click(
                lambda: [
                    0,
                    0,
                    0.5,
                    3,
                    0.25,
                    0.33,
                    "rmvpe",
                    128,
                    gr.Dropdown(value=converted_vocals_track_transfer_default),
                ],
                outputs=[
                    pitch_change_octaves,
                    pitch_change_semitones,
                    index_rate,
                    filter_radius,
                    rms_mix_rate,
                    protect,
                    f0_method,
                    crepe_hop_length,
                    converted_vocals_track_transfer_dropdown,
                ],
                show_progress="hidden",
            )
            convert_vocals_btn.render()
            convert_vocals_event_args_list = [
                EventArgs(
                    partial(
                        exception_harness(convert_vocals), progress_bar=PROGRESS_BAR
                    ),
                    inputs=[
                        dereverbed_vocals_track_input,
                        convert_vocals_dir,
                        rvc_model,
                        pitch_change_octaves,
                        pitch_change_semitones,
                        index_rate,
                        filter_radius,
                        rms_mix_rate,
                        protect,
                        f0_method,
                        crepe_hop_length,
                    ],
                    outputs=[converted_vocals_track_output],
                ),
                EventArgs(
                    partial(_transfer, len(input_tracks)),
                    inputs=[
                        converted_vocals_track_transfer_dropdown,
                        converted_vocals_track_output,
                    ],
                    outputs=input_tracks,
                    name="then",
                    show_progress="hidden",
                ),
            ]
            setup_consecutive_event_listeners_with_toggled_interactivity(
                convert_vocals_btn,
                convert_vocals_event_args_list,
                generate_buttons,
            )
        with gr.Accordion("Step 5: post-processing of vocals", open=False):
            gr.Markdown("")
            gr.Markdown("**Inputs**")
            converted_vocals_track_input.render()
            postprocess_vocals_dir.render()
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
            gr.Markdown("**Outputs**")

            postprocessed_vocals_track_output.render()
            postprocessed_vocals_track_transfer_dropdown.render()

            postprocess_vocals_clear_btn.render()
            postprocess_vocals_clear_btn.click(
                lambda: [
                    0.15,
                    0.2,
                    0.8,
                    0.7,
                    gr.Dropdown(value=postprocessed_vocals_track_transfer_default),
                ],
                outputs=[
                    reverb_rm_size,
                    reverb_wet,
                    reverb_dry,
                    reverb_damping,
                    postprocessed_vocals_track_transfer_dropdown,
                ],
                show_progress="hidden",
            )
            postprocess_vocals_btn.render()
            postprocess_vocals_event_args_list = [
                EventArgs(
                    partial(
                        exception_harness(postprocess_vocals),
                        progress_bar=PROGRESS_BAR,
                    ),
                    inputs=[
                        converted_vocals_track_input,
                        postprocess_vocals_dir,
                        reverb_rm_size,
                        reverb_wet,
                        reverb_dry,
                        reverb_damping,
                    ],
                    outputs=[postprocessed_vocals_track_output],
                ),
                EventArgs(
                    partial(_transfer, len(input_tracks)),
                    inputs=[
                        postprocessed_vocals_track_transfer_dropdown,
                        postprocessed_vocals_track_output,
                    ],
                    outputs=input_tracks,
                    name="then",
                    show_progress="hidden",
                ),
            ]
            setup_consecutive_event_listeners_with_toggled_interactivity(
                postprocess_vocals_btn,
                postprocess_vocals_event_args_list,
                generate_buttons,
            )
        with gr.Accordion("Step 6: pitch shift of background tracks", open=False):
            gr.Markdown("")
            gr.Markdown("**Inputs**")
            with gr.Row():
                instrumentals_track_input.render()
                backup_vocals_track_input.render()
            pitch_shift_background_dir.render()
            pitch_change_semitones_background = gr.Slider(
                -12,
                12,
                value=0,
                step=1,
                label="Pitch shift",
                info="Shift pitch of instrumentals and backup vocals. Measured in semi-tones.",
            )
            gr.Markdown("**Outputs**")
            with gr.Row():
                with gr.Column():
                    shifted_instrumentals_track_output.render()
                    shifted_instrumentals_track_transfer_dropdown.render()
                with gr.Column():
                    shifted_backup_vocals_track_output.render()
                    shifted_backup_vocals_track_transfer_dropdown.render()

            pitch_shift_background_clear_btn.render()
            pitch_shift_background_clear_btn.click(
                lambda: [
                    0,
                    gr.Dropdown(value=shifted_instrumentals_track_transfer_default),
                    gr.Dropdown(value=shifted_backup_vocals_track_transfer_default),
                ],
                outputs=[
                    pitch_change_semitones_background,
                    shifted_instrumentals_track_transfer_dropdown,
                    shifted_backup_vocals_track_transfer_dropdown,
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
                        pitch_change_semitones_background,
                    ],
                    outputs=[
                        shifted_instrumentals_track_output,
                        shifted_backup_vocals_track_output,
                    ],
                )
            ] + [
                EventArgs(
                    partial(_transfer, len(input_tracks)),
                    inputs=[dropdown, output_track],
                    outputs=input_tracks,
                    name="then",
                    show_progress="hidden",
                )
                for dropdown, output_track in zip(
                    [
                        shifted_instrumentals_track_transfer_dropdown,
                        shifted_backup_vocals_track_transfer_dropdown,
                    ],
                    [
                        shifted_instrumentals_track_output,
                        shifted_backup_vocals_track_output,
                    ],
                )
            ]

            setup_consecutive_event_listeners_with_toggled_interactivity(
                pitch_shift_background_btn,
                pitch_shift_background_event_args_list,
                generate_buttons,
            )
        with gr.Accordion("Step 7: song mixing", open=False):
            gr.Markdown("")
            gr.Markdown("**Inputs**")
            with gr.Row():
                postprocessed_vocals_track_input.render()
                shifted_instrumentals_track_input.render()
                shifted_backup_vocals_track_input.render()
            mix_dir.render()
            with gr.Row():
                main_gain = gr.Slider(-20, 20, value=0, step=1, label="Main vocals")
                inst_gain = gr.Slider(-20, 20, value=0, step=1, label="Instrumentals")
                backup_gain = gr.Slider(-20, 20, value=0, step=1, label="Backup vocals")
            with gr.Row():
                output_name = gr.Text(
                    label="Output file name",
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
                postprocessed_vocals_track_input.change(
                    get_song_cover_name_harness,
                    inputs=[postprocessed_vocals_track_input, mix_dir],
                    outputs=output_name,
                    show_progress="hidden",
                )
                mix_dir.change(
                    get_song_cover_name_harness,
                    inputs=[postprocessed_vocals_track_input, mix_dir],
                    outputs=output_name,
                    show_progress="hidden",
                )

            gr.Markdown("**Outputs**")
            song_cover_track.render()
            song_cover_track_transfer_dropdown.render()
            mix_clear_btn.render()
            mix_clear_btn.click(
                lambda: [
                    0,
                    0,
                    0,
                    44100,
                    "mp3",
                    gr.Dropdown(value=song_cover_track_transfer_default),
                ],
                outputs=[
                    main_gain,
                    inst_gain,
                    backup_gain,
                    output_sr,
                    output_format,
                    song_cover_track_transfer_dropdown,
                ],
                show_progress="hidden",
            )
            mix_btn.render()
            mix_btn_event_args_list = [
                EventArgs(
                    partial(
                        exception_harness(mix_song_cover), progress_bar=PROGRESS_BAR
                    ),
                    inputs=[
                        postprocessed_vocals_track_input,
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
                    outputs=[song_cover_track],
                ),
                EventArgs(
                    partial(update_output_audio, 1, [], [0]),
                    outputs=[output_audio_to_remove],
                    name="then",
                    show_progress="hidden",
                ),
                EventArgs(
                    partial(_transfer, len(input_tracks)),
                    inputs=[song_cover_track_transfer_dropdown, song_cover_track],
                    outputs=input_tracks,
                    name="then",
                    show_progress="hidden",
                ),
            ]

            setup_consecutive_event_listeners_with_toggled_interactivity(
                mix_btn,
                mix_btn_event_args_list,
                generate_buttons,
            )
