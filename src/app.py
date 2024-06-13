import os
from argparse import ArgumentParser

import gradio as gr
from gradio.components.base import Component
import asyncio

from functools import partial
from dataclasses import dataclass
from typing import Callable, Literal, Optional
from common import GRADIO_TEMP_DIR


from manage_voice_models import (
    get_current_models,
    load_public_models_table,
    load_public_model_tags,
    filter_public_models_table,
    download_online_model,
    upload_local_model,
    delete_models,
    delete_all_models,
)

from generate_song_cover import (
    get_named_song_dirs,
    delete_intermediate_audio,
    delete_all_intermediate_audio,
    get_song_cover_name,
    retrieve_song,
    separate_vocals,
    separate_main_vocals,
    dereverb_main_vocals,
    convert_main_vocals,
    postprocess_main_vocals,
    pitch_shift_background,
    mix_w_background,
    run_pipeline,
)

os.environ["GRADIO_TEMP_DIR"] = GRADIO_TEMP_DIR

if os.name == "nt":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

progress_bar = gr.Progress()


def confirmation_harness(fun, confirm, *args, **kwargs):
    if confirm:
        return exception_harness(fun, *args, **kwargs)
    else:
        raise gr.Error("Confirmation missing!")


def exception_harness(fun, *args, **kwargs):
    new_args = args + (progress_bar,)
    try:
        return fun(*new_args, **kwargs)
    except Exception as e:
        raise gr.Error(str(e))


def duplication_harness(fun, *args, **kwargs):

    res = exception_harness(fun, *args, **kwargs)
    if not isinstance(res, tuple):
        return (res, res)
    else:
        return (res[0],) + res


def update_audio_components(*args):
    res = run_pipeline(*args)
    if isinstance(res, tuple):
        return res
    else:
        return (None,) * 11 + (res,)


def mix_w_background_harness(
    vocals_path,
    instrumentals_path,
    backup_vocals_path,
    instrumentals_shifted_path,
    backup_vocals_shifted_path,
    *args,
):
    return mix_w_background(
        vocals_path,
        instrumentals_shifted_path or instrumentals_path,
        backup_vocals_shifted_path or backup_vocals_path,
        *args,
    )


def filter_public_models_table_harness(tags, query, progress):
    models_table = filter_public_models_table(tags, query, progress)
    return gr.DataFrame(value=models_table)


def confirm_box_js(msg):
    formatted_msg = f"'{msg}'"
    return f"(x) => confirm({formatted_msg})"


def update_dropdowns(fn, num_components, value=None, value_indices=[], **kwargs):
    if len(value_indices) > num_components:
        raise ValueError(
            "Number of components to update value for exceeds number of components."
        )
    if value_indices and max(value_indices) >= num_components:
        raise ValueError(
            "Index of component to update value for exceeds number of components."
        )
    updated_choices = fn(**kwargs)
    update_args = [{"choices": updated_choices} for _ in range(num_components)]
    for index in value_indices:
        update_args[index]["value"] = value
    return tuple(gr.Dropdown(**update_arg) for update_arg in update_args)


def update_model_lists(num_components, value=None, value_indices=[]):
    return update_dropdowns(
        get_current_models, num_components, value=value, value_indices=value_indices
    )


def update_cached_input_songs(num_components, value=None, value_indices=[]):
    return update_dropdowns(
        get_named_song_dirs,
        num_components,
        value=value,
        value_indices=value_indices,
    )


def pub_dl_autofill(pub_models, event: gr.SelectData):
    return gr.Text(value=pub_models.loc[event.index[0], "URL"]), gr.Text(
        value=pub_models.loc[event.index[0], "Model Name"]
    )


def toggle_visible_component(num_components, visible_index):
    update_args = [{"visible": False, "value": None} for _ in range(num_components)]
    update_args[visible_index]["visible"] = True
    return tuple(gr.update(**update_arg) for update_arg in update_args)


def show_hop_slider(pitch_detection_algo):
    if pitch_detection_algo == "mangio-crepe":
        return gr.update(visible=True)
    else:
        return gr.update(visible=False)


def toggle_intermediate_files_accordion(visible):
    audio_components = (None,) * 11
    accordions = (gr.update(open=False),) * 7
    return (gr.update(visible=visible, open=False),) + accordions + audio_components


def duplication_harness2(fun, *args):
    res = exception_harness(fun, *args)
    if not isinstance(res, tuple):
        return (res, res)
    else:
        return res + res


def mix_w_background_harness2(*args):
    new_args = args + (True,)
    return exception_harness(mix_w_background, *new_args)


def get_song_cover_name_harness(mixed_vocals, song_dir, voice_model=None):
    update_args = {}
    if mixed_vocals or song_dir or voice_model:
        name = exception_harness(
            get_song_cover_name, mixed_vocals, song_dir, voice_model
        )
        update_args["value"] = name
    else:
        update_args["value"] = None
    return gr.update(**update_args)


def transfer(num_components, output_indices, value):
    update_args = [{} for _ in range(num_components)]
    for index in output_indices:
        update_args[index]["value"] = value
    return tuple(gr.update(**update_arg) for update_arg in update_args)


def toggle_component_interactivity(num_components, interactive):
    return tuple(gr.update(interactive=interactive) for _ in range(num_components))


@dataclass
class EventArgs:
    fn: Callable
    inputs: Optional[list[Component]] = None
    outputs: Optional[list[Component]] = None
    name: Literal["success", "then"] = "success"
    show_progress: Literal["full", "minimal", "hidden"] = "full"


def setup_consecutive_event_listeners(
    component: Component, event_args_list: list[EventArgs]
):
    if len(event_args_list) == 0:
        raise ValueError("Event args list must not be empty.")
    dependency = component
    for event_args in event_args_list:
        event_listener = getattr(dependency, event_args.name)
        dependency = event_listener(
            event_args.fn,
            inputs=event_args.inputs,
            outputs=event_args.outputs,
            show_progress=event_args.show_progress,
        )
    return dependency


def setup_consecutive_event_listeners_with_toggled_interactivity(
    component: Component,
    event_args_list: list[EventArgs],
    toggled_components: list[Component],
):
    if len(event_args_list) == 0:
        raise ValueError("Event args list must not be empty.")

    disable_event_args = EventArgs(
        partial(toggle_component_interactivity, len(toggled_components), False),
        outputs=toggled_components,
        name="click",
        show_progress="hidden",
    )
    enable_event_args = EventArgs(
        partial(toggle_component_interactivity, len(toggled_components), True),
        outputs=toggled_components,
        name="then",
        show_progress="hidden",
    )
    event_args_list_augmented = (
        [disable_event_args] + event_args_list + [enable_event_args]
    )
    return setup_consecutive_event_listeners(component, event_args_list_augmented)


voice_models = get_current_models()
cached_input_songs = get_named_song_dirs()

with gr.Blocks(title="Ultimate RVC") as app:

    gr.Label("Ultimate RVC ❤️", show_label=False)

    dummy_deletion_checkbox = gr.Checkbox(visible=False)
    delete_confirmation = gr.State(False)
    current_song_dir, current_song_dir2 = [gr.State(None) for _ in range(2)]
    song_dir_dropdowns = [
        gr.Dropdown(
            cached_input_songs,
            label="Song directory",
            info="Directory where intermediate audio files are stored and loaded from locally. When a new song is retrieved, its directory is chosen by default.",
            render=False,
        )
        for _ in range(7)
    ]
    cached_input_songs_dropdown, cached_input_songs_dropdown2 = [
        gr.Dropdown(
            cached_input_songs,
            label="Song input",
            info="Select a song from the list of cached songs.",
            visible=False,
            render=False,
        )
        for _ in range(2)
    ]
    intermediate_audio_to_delete = gr.Dropdown(
        cached_input_songs,
        label="Songs with intermediate audio files",
        filterable=True,
        multiselect=True,
        info="Select one or more songs to delete their asssociated intermediate audio files.",
        render=False,
    )
    rvc_model, rvc_model2 = [
        gr.Dropdown(voice_models, label="Voice model", render=False) for _ in range(2)
    ]

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

    (
        retrieve_song_btn,
        separate_vocals_btn,
        separate_main_vocals_btn,
        dereverb_main_vocals_btn,
        convert_main_vocals_btn,
        postprocess_main_vocals_btn,
        pitch_shift_background_btn,
        mix_btn,
        generate_btn,
        generate_btn2,
    ) = generate_buttons

    # main tab
    with gr.Tab("Step-by-step generation"):

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
                "Pitch shifted instrumentals",
                "Pitch shifted backup vocals",
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

        steps = [
            "Step 0: Song retrieval",
            "Step 1: vocals/instrumentals separation",
            "Step 2: main vocals/ backup vocals separation",
            "Step 3: vocal cleanup",
            "Step 4: vocal conversion",
            "Step 5: post-processing of vocals",
            "Step 6: pitch shift of background",
            "Step 7: song mixing",
        ]

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

        clear_btns = [
            gr.Button(
                value="Reset settings",
                render=False,
            )
            for _ in range(8)
        ]

        with gr.Accordion(steps[0], open=False):
            gr.Markdown("")
            gr.Markdown("**Inputs**")
            with gr.Row():
                with gr.Column():
                    song_input_type_dropdown2 = gr.Dropdown(
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
                    song_input2 = gr.Text(
                        label="Song input",
                        info="Link to a song on YouTube or the full path of a local audio file.",
                    )
                    local_file2 = gr.Audio(
                        label="Song input",
                        type="filepath",
                        visible=False,
                    )
                    cached_input_songs_dropdown2.render()

                song_input_type_dropdown2.input(
                    partial(toggle_visible_component, 3),
                    inputs=song_input_type_dropdown2,
                    outputs=[song_input2, local_file2, cached_input_songs_dropdown2],
                    show_progress="hidden",
                )

                local_file2.change(
                    lambda x: gr.update(value=x),
                    inputs=local_file2,
                    outputs=song_input2,
                )
                cached_input_songs_dropdown2.input(
                    lambda x: gr.update(value=x),
                    inputs=cached_input_songs_dropdown2,
                    outputs=song_input2,
                    show_progress="hidden",
                )
            gr.Markdown("**Outputs**")
            original_track_output.render()
            transfer_output_track_dropdowns[0].render()
            clear_btns[0].render()
            clear_btns[0].click(
                lambda: gr.Dropdown(value=transfer_defaults[0]),
                outputs=[transfer_output_track_dropdowns[0]],
                show_progress="hidden",
            )

            retrieve_song_btn.render()

            retrieve_song_event_args_list = [
                EventArgs(
                    partial(exception_harness, retrieve_song),
                    inputs=song_input2,
                    outputs=[original_track_output, current_song_dir2],
                ),
                EventArgs(
                    partial(
                        update_cached_input_songs,
                        len(song_dir_dropdowns) + 3,
                        value_indices=range(len(song_dir_dropdowns) + 1),
                    ),
                    inputs=current_song_dir2,
                    outputs=(
                        song_dir_dropdowns
                        + [
                            cached_input_songs_dropdown2,
                            cached_input_songs_dropdown,
                            intermediate_audio_to_delete,
                        ]
                    ),
                    name="then",
                    show_progress="hidden",
                ),
                EventArgs(
                    partial(transfer, len(input_tracks)),
                    inputs=[transfer_output_track_dropdowns[0], original_track_output],
                    outputs=input_tracks,
                    name="then",
                    show_progress="hidden",
                ),
            ]
            retrieve_song_btn_click = (
                setup_consecutive_event_listeners_with_toggled_interactivity(
                    retrieve_song_btn,
                    retrieve_song_event_args_list,
                    generate_buttons,
                )
            )
        with gr.Accordion(steps[1], open=False):
            gr.Markdown("")
            gr.Markdown("**Inputs**")
            original_track_input.render()
            song_dir_dropdowns[0].render()
            gr.Markdown("**Outputs**")
            with gr.Row():
                with gr.Column():
                    vocals_track_output.render()
                    transfer_output_track_dropdowns[1].render()

                with gr.Column():
                    instrumentals_track_output.render()
                    transfer_output_track_dropdowns[2].render()

            clear_btns[1].render()
            clear_btns[1].click(
                lambda: tuple(
                    gr.Dropdown(value=value) for value in transfer_defaults[1:3]
                ),
                outputs=transfer_output_track_dropdowns[1:3],
                show_progress="hidden",
            )
            separate_vocals_btn.render()

            separate_vocals_event_args_list = [
                EventArgs(
                    partial(exception_harness, separate_vocals),
                    inputs=[original_track_input, song_dir_dropdowns[0]],
                    outputs=[vocals_track_output, instrumentals_track_output],
                )
            ] + [
                EventArgs(
                    partial(transfer, len(input_tracks)),
                    inputs=[transfer_dropdown, output_track],
                    outputs=input_tracks,
                    name="then",
                    show_progress="hidden",
                )
                for transfer_dropdown, output_track in zip(
                    transfer_output_track_dropdowns[1:3],
                    [vocals_track_output, instrumentals_track_output],
                )
            ]

            separate_vocals_btn_click = (
                setup_consecutive_event_listeners_with_toggled_interactivity(
                    separate_vocals_btn,
                    separate_vocals_event_args_list,
                    generate_buttons,
                )
            )

        with gr.Accordion(steps[2], open=False):
            gr.Markdown("")
            gr.Markdown("**Inputs**")
            vocals_track_input.render()
            song_dir_dropdowns[1].render()
            gr.Markdown("**Outputs**")
            with gr.Row():
                with gr.Column():
                    main_vocals_track_output.render()
                    transfer_output_track_dropdowns[3].render()
                with gr.Column():
                    backup_vocals_track_output.render()
                    transfer_output_track_dropdowns[4].render()

            clear_btns[2].render()
            clear_btns[2].click(
                lambda: tuple(
                    gr.Dropdown(value=value) for value in transfer_defaults[3:5]
                ),
                outputs=transfer_output_track_dropdowns[3:5],
                show_progress="hidden",
            )
            separate_main_vocals_btn.render()

            separate_main_vocals_event_args_list = [
                EventArgs(
                    partial(exception_harness, separate_main_vocals),
                    inputs=[vocals_track_input, song_dir_dropdowns[1]],
                    outputs=[main_vocals_track_output, backup_vocals_track_output],
                )
            ] + [
                EventArgs(
                    partial(transfer, len(input_tracks)),
                    inputs=[transfer_dropdown, output_track],
                    outputs=input_tracks,
                    name="then",
                    show_progress="hidden",
                )
                for transfer_dropdown, output_track in zip(
                    transfer_output_track_dropdowns[3:5],
                    [main_vocals_track_output, backup_vocals_track_output],
                )
            ]

            separate_main_vocals_btn_click = (
                setup_consecutive_event_listeners_with_toggled_interactivity(
                    separate_main_vocals_btn,
                    separate_main_vocals_event_args_list,
                    generate_buttons,
                )
            )

        with gr.Accordion(steps[3], open=False):
            gr.Markdown("")
            gr.Markdown("**Inputs**")
            main_vocals_track_input.render()
            song_dir_dropdowns[2].render()
            gr.Markdown("**Outputs**")
            with gr.Row():
                with gr.Column():
                    dereverbed_vocals_track_output.render()
                    transfer_output_track_dropdowns[5].render()
                with gr.Column():
                    reverb_track_output.render()
                    transfer_output_track_dropdowns[6].render()

            clear_btns[3].render()
            clear_btns[3].click(
                lambda: tuple(
                    gr.Dropdown(value=value) for value in transfer_defaults[5:7]
                ),
                outputs=transfer_output_track_dropdowns[5:7],
                show_progress="hidden",
            )
            dereverb_main_vocals_btn.render()
            dereverb_main_vocals_event_args_list = [
                EventArgs(
                    partial(exception_harness, dereverb_main_vocals),
                    inputs=[main_vocals_track_input, song_dir_dropdowns[2]],
                    outputs=[dereverbed_vocals_track_output, reverb_track_output],
                )
            ] + [
                EventArgs(
                    partial(transfer, len(input_tracks)),
                    inputs=[transfer_dropdown, output_track],
                    outputs=input_tracks,
                    name="then",
                    show_progress="hidden",
                )
                for transfer_dropdown, output_track in zip(
                    transfer_output_track_dropdowns[5:7],
                    [dereverbed_vocals_track_output, reverb_track_output],
                )
            ]

            dereverb_main_vocals_btn_click = (
                setup_consecutive_event_listeners_with_toggled_interactivity(
                    dereverb_main_vocals_btn,
                    dereverb_main_vocals_event_args_list,
                    generate_buttons,
                )
            )
        with gr.Accordion(steps[4], open=False):
            gr.Markdown("")
            gr.Markdown("**Inputs**")
            dereverbed_vocals_track_input.render()
            song_dir_dropdowns[3].render()
            with gr.Row():
                rvc_model2.render()
                pitch_change_octaves = gr.Slider(
                    -3,
                    3,
                    value=0,
                    step=1,
                    label="Pitch shift (octaves)",
                    info="Shift pitch of converted vocals by number of octaves. Generally, use 1 for male-to-female conversions and -1 for vice-versa.",
                )
                # TODO when changing this component the component in the step 6 accordion should be changed accordingly
                # alternatively, consider making this a component
                pitch_change_semitones = gr.Slider(
                    -12,
                    12,
                    value=0,
                    step=1,
                    label="Pitch shift (semi-tones)",
                    info="Shift pitch of converted vocals by number of semi-tones. Altering this slightly reduces sound quality.",
                )
            with gr.Row():
                index_rate2 = gr.Slider(
                    0,
                    1,
                    value=0.5,
                    label="Index rate",
                    info="Controls how much of the accent in the voice model to keep in the converted vocals",
                )
                filter_radius2 = gr.Slider(
                    0,
                    7,
                    value=3,
                    step=1,
                    label="Filter radius",
                    info="If >=3: apply median filtering to the harvested pitch results. Can reduce breathiness",
                )
                rms_mix_rate2 = gr.Slider(
                    0,
                    1,
                    value=0.25,
                    label="RMS mix rate",
                    info="Control how much to mimic the loudness (0) of the input vocals or a fixed loudness (1)",
                )
                protect2 = gr.Slider(
                    0,
                    0.5,
                    value=0.33,
                    label="Protect rate",
                    info="Protect voiceless consonants and breath sounds. Set to 0.5 to disable.",
                )
                with gr.Column():
                    f0_method2 = gr.Dropdown(
                        ["rmvpe", "mangio-crepe"],
                        value="rmvpe",
                        label="Pitch detection algorithm",
                        info="Best option is rmvpe (clarity in vocals), then mangio-crepe (smoother vocals)",
                    )
                    crepe_hop_length2 = gr.Slider(
                        32,
                        320,
                        value=128,
                        step=1,
                        visible=False,
                        label="Crepe hop length",
                        info="Lower values leads to longer conversions and higher risk of voice cracks, but better pitch accuracy.",
                    )
                    f0_method2.change(
                        show_hop_slider,
                        inputs=f0_method2,
                        outputs=crepe_hop_length2,
                        show_progress="hidden",
                    )

            gr.Markdown("**Outputs**")
            converted_vocals_track_output.render()
            transfer_output_track_dropdowns[7].render()
            clear_btns[4].render()
            clear_btns[4].click(
                lambda: [
                    0,
                    0,
                    0.5,
                    3,
                    0.25,
                    0.33,
                    "rmvpe",
                    128,
                    gr.Dropdown(value=transfer_defaults[7]),
                ],
                outputs=[
                    pitch_change_octaves,
                    pitch_change_semitones,
                    index_rate2,
                    filter_radius2,
                    rms_mix_rate2,
                    protect2,
                    f0_method2,
                    crepe_hop_length2,
                    transfer_output_track_dropdowns[7],
                ],
                show_progress="hidden",
            )
            convert_main_vocals_btn.render()
            convert_main_vocals_event_args_list = [
                EventArgs(
                    partial(exception_harness, convert_main_vocals),
                    inputs=[
                        dereverbed_vocals_track_input,
                        song_dir_dropdowns[3],
                        rvc_model2,
                        pitch_change_octaves,
                        pitch_change_semitones,
                        index_rate2,
                        filter_radius2,
                        rms_mix_rate2,
                        protect2,
                        f0_method2,
                        crepe_hop_length2,
                    ],
                    outputs=[converted_vocals_track_output],
                ),
                EventArgs(
                    partial(transfer, len(input_tracks)),
                    inputs=[
                        transfer_output_track_dropdowns[7],
                        converted_vocals_track_output,
                    ],
                    outputs=input_tracks,
                    name="then",
                    show_progress="hidden",
                ),
            ]
            convert_main_vocals_btn_click = (
                setup_consecutive_event_listeners_with_toggled_interactivity(
                    convert_main_vocals_btn,
                    convert_main_vocals_event_args_list,
                    generate_buttons,
                )
            )
        with gr.Accordion(steps[5], open=False):
            converted_vocals_track_input.render()
            song_dir_dropdowns[4].render()
            with gr.Row():
                reverb_rm_size2 = gr.Slider(
                    0,
                    1,
                    value=0.15,
                    label="Room size",
                    info="The larger the room, the longer the reverb time",
                )
                reverb_wet2 = gr.Slider(
                    0,
                    1,
                    value=0.2,
                    label="Wetness level",
                    info="Loudness level of converted vocals with reverb",
                )
                reverb_dry2 = gr.Slider(
                    0,
                    1,
                    value=0.8,
                    label="Dryness level",
                    info="Loudness level of converted vocals without reverb",
                )
                reverb_damping2 = gr.Slider(
                    0,
                    1,
                    value=0.7,
                    label="Damping level",
                    info="Absorption of high frequencies in the reverb",
                )

            postprocessed_vocals_track_output.render()
            transfer_output_track_dropdowns[8].render()

            clear_btns[5].render()
            clear_btns[5].click(
                lambda: [
                    0.15,
                    0.2,
                    0.8,
                    0.7,
                    gr.Dropdown(value=transfer_defaults[8]),
                ],
                outputs=[
                    reverb_rm_size2,
                    reverb_wet2,
                    reverb_dry2,
                    reverb_damping2,
                    transfer_output_track_dropdowns[8],
                ],
                show_progress="hidden",
            )
            postprocess_main_vocals_btn.render()
            postprocess_main_vocals_event_args_list = [
                EventArgs(
                    partial(exception_harness, postprocess_main_vocals),
                    inputs=[
                        converted_vocals_track_input,
                        song_dir_dropdowns[4],
                        reverb_rm_size2,
                        reverb_wet2,
                        reverb_dry2,
                        reverb_damping2,
                    ],
                    outputs=[postprocessed_vocals_track_output],
                ),
                EventArgs(
                    partial(transfer, len(input_tracks)),
                    inputs=[
                        transfer_output_track_dropdowns[8],
                        postprocessed_vocals_track_output,
                    ],
                    outputs=input_tracks,
                    name="then",
                    show_progress="hidden",
                ),
            ]
            postprocess_main_vocals_btn_click = (
                setup_consecutive_event_listeners_with_toggled_interactivity(
                    postprocess_main_vocals_btn,
                    postprocess_main_vocals_event_args_list,
                    generate_buttons,
                )
            )
        with gr.Accordion(steps[6], open=False):
            gr.Markdown("")
            gr.Markdown("**Inputs**")
            with gr.Row():
                instrumentals_track_input.render()
                backup_vocals_track_input.render()
            song_dir_dropdowns[5].render()
            # TODO changing this should change the pitch_change_semitones in the step 4 accordion
            # alternatively, consider making this a global component
            pitch_change_semitones2 = gr.Slider(
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
                    transfer_output_track_dropdowns[9].render()
                with gr.Column():
                    shifted_backup_vocals_track_output.render()
                    transfer_output_track_dropdowns[10].render()

            clear_btns[6].render()
            clear_btns[6].click(
                lambda: [
                    0,
                    gr.Dropdown(value=transfer_defaults[9]),
                    gr.Dropdown(value=transfer_defaults[10]),
                ],
                outputs=[
                    pitch_change_semitones2,
                    transfer_output_track_dropdowns[9],
                    transfer_output_track_dropdowns[10],
                ],
                show_progress="hidden",
            )
            pitch_shift_background_btn.render()
            pitch_shift_background_event_args_list = [
                EventArgs(
                    partial(exception_harness, pitch_shift_background),
                    inputs=[
                        instrumentals_track_input,
                        backup_vocals_track_input,
                        song_dir_dropdowns[5],
                        pitch_change_semitones2,
                    ],
                    outputs=[
                        shifted_instrumentals_track_output,
                        shifted_backup_vocals_track_output,
                    ],
                ),
            ] + [
                EventArgs(
                    partial(transfer, len(input_tracks)),
                    inputs=[dropdown, output_track],
                    outputs=input_tracks,
                    name="then",
                    show_progress="hidden",
                )
                for dropdown, output_track in zip(
                    transfer_output_track_dropdowns[9:11],
                    [
                        shifted_instrumentals_track_output,
                        shifted_backup_vocals_track_output,
                    ],
                )
            ]

            pitch_shift_background_btn_click = (
                setup_consecutive_event_listeners_with_toggled_interactivity(
                    pitch_shift_background_btn,
                    pitch_shift_background_event_args_list,
                    generate_buttons,
                )
            )
        with gr.Accordion(steps[7], open=False):
            gr.Markdown("")
            gr.Markdown("**Inputs**")
            with gr.Row():
                postprocessed_vocals_track_input.render()
                shifted_instrumentals_track_input.render()
                shifted_backup_vocals_track_input.render()
            song_dir_dropdowns[6].render()
            with gr.Row():
                main_gain2 = gr.Slider(-20, 20, value=0, step=1, label="Main vocals")
                inst_gain2 = gr.Slider(-20, 20, value=0, step=1, label="Instrumentals")
                backup_gain2 = gr.Slider(
                    -20, 20, value=0, step=1, label="Backup vocals"
                )
            with gr.Row():
                output_name2 = gr.Text(
                    label="Output file name",
                    placeholder="Ultimate RVC song cover",
                )
                output_sr2 = gr.Dropdown(
                    choices=[16000, 44100, 48000, 96000, 192000],
                    value=44100,
                    label="Output sample rate",
                )
                output_format2 = gr.Dropdown(
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
                    inputs=[postprocessed_vocals_track_input, song_dir_dropdowns[6]],
                    outputs=output_name2,
                    show_progress="hidden",
                )
                song_dir_dropdowns[6].change(
                    get_song_cover_name_harness,
                    inputs=[postprocessed_vocals_track_input, song_dir_dropdowns[6]],
                    outputs=output_name2,
                    show_progress="hidden",
                )

            gr.Markdown("**Outputs**")
            song_cover_track.render()
            transfer_output_track_dropdowns[11].render()
            clear_btns[7].render()
            clear_btns[7].click(
                lambda: [
                    0,
                    0,
                    0,
                    44100,
                    "mp3",
                    gr.Dropdown(value=transfer_defaults[11]),
                ],
                outputs=[
                    main_gain2,
                    backup_gain2,
                    inst_gain2,
                    output_sr2,
                    output_format2,
                    transfer_output_track_dropdowns[11],
                ],
                show_progress="hidden",
            )
            mix_btn.render()
            mix_btn_event_args_list = [
                EventArgs(
                    partial(mix_w_background_harness2),
                    inputs=[
                        postprocessed_vocals_track_input,
                        shifted_instrumentals_track_input,
                        shifted_backup_vocals_track_input,
                        song_dir_dropdowns[6],
                        main_gain2,
                        backup_gain2,
                        inst_gain2,
                        output_sr2,
                        output_format2,
                        output_name2,
                    ],
                    outputs=[song_cover_track],
                ),
                EventArgs(
                    partial(transfer, len(input_tracks)),
                    inputs=[transfer_output_track_dropdowns[11], song_cover_track],
                    outputs=input_tracks,
                    name="then",
                    show_progress="hidden",
                ),
            ]

            mix_btn_click = (
                setup_consecutive_event_listeners_with_toggled_interactivity(
                    mix_btn,
                    mix_btn_event_args_list,
                    generate_buttons,
                )
            )

    with gr.Tab("Generate song covers"):
        # with gr.Tab("One-click generation"):
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
                        sources="upload",
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
                        lambda x: gr.update(value=x),
                        inputs=local_file,
                        outputs=song_input,
                    )
                    cached_input_songs_dropdown.input(
                        lambda x: gr.update(value=x),
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
                backup_gain = gr.Slider(-20, 20, value=0, step=1, label="Backup vocals")
                inst_gain = gr.Slider(-20, 20, value=0, step=1, label="Instrumentals")
        with gr.Accordion("Audio output options", open=False):
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
            rvc_model.change(
                partial(get_song_cover_name_harness, None),
                inputs=[cached_input_songs_dropdown, rvc_model],
                outputs=output_name,
                show_progress="hidden",
            )
            cached_input_songs_dropdown.change(
                partial(get_song_cover_name_harness, None),
                inputs=[cached_input_songs_dropdown, rvc_model],
                outputs=output_name,
                show_progress="hidden",
            )
        with gr.Accordion("Intermediate audio options", open=False):
            with gr.Row():
                keep_files = gr.Checkbox(
                    label="Keep intermediate audio files",
                    value=True,
                    info="Keep intermediate audio files generated during song cover generation. Leave unchecked to save space.",
                )
                show_intermediate_files = gr.Checkbox(
                    label="Show intermediate audio files",
                    value=False,
                    info="Show generated intermediate audio files when song cover generation completes. Leave unchecked to optimize performance.",
                )
            with gr.Accordion("Delete intermediate audio files", open=False):
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
                        intermediate_audio_delete_msg = gr.Text(
                            label="Output message", interactive=False
                        )

                delete_intermediate_audio_click = delete_intermediate_audio_btn.click(
                    lambda x: x,
                    inputs=dummy_deletion_checkbox,
                    outputs=delete_confirmation,
                    js=confirm_box_js(
                        "Are you sure you want to delete intermediate audio files for the selected songs?"
                    ),
                ).then(
                    partial(confirmation_harness, delete_intermediate_audio),
                    inputs=[delete_confirmation, intermediate_audio_to_delete],
                    outputs=intermediate_audio_delete_msg,
                )

                delete_all_intermediate_audio_click = delete_all_intermediate_audio_btn.click(
                    lambda x: x,
                    inputs=dummy_deletion_checkbox,
                    outputs=delete_confirmation,
                    js=confirm_box_js(
                        "Are you sure you want to delete all intermediate audio files?"
                    ),
                ).then(
                    partial(confirmation_harness, delete_all_intermediate_audio),
                    inputs=delete_confirmation,
                    outputs=intermediate_audio_delete_msg,
                )
                for click_event in [
                    delete_intermediate_audio_click,
                    delete_all_intermediate_audio_click,
                ]:
                    click_event.success(
                        partial(
                            update_cached_input_songs,
                            3 + len(song_dir_dropdowns),
                            [],
                            [0],
                        ),
                        outputs=[
                            intermediate_audio_to_delete,
                            cached_input_songs_dropdown,
                            cached_input_songs_dropdown2,
                        ]
                        + song_dir_dropdowns,
                    )

        with gr.Accordion(
            "Access intermediate audio files", open=False, visible=False
        ) as intermediate_files_accordion:

            with gr.Accordion("Step 0: input", open=False) as original_accordion:
                original_track = gr.Audio(
                    label="Original song", type="filepath", interactive=False
                )

            with gr.Accordion(
                "Step 1: instrumentals/vocals separation", open=False
            ) as vocals_separation_accordion:
                with gr.Row():
                    vocals_track = gr.Audio(
                        label="Vocals", type="filepath", interactive=False
                    )
                    instrumentals_track = gr.Audio(
                        label="Instrumentals", type="filepath", interactive=False
                    )
            with gr.Accordion(
                "Step 2: main vocals/ backup vocals separation", open=False
            ) as main_vocals_separation_accordion:
                with gr.Row():
                    main_vocals_track = gr.Audio(
                        label="Main vocals", type="filepath", interactive=False
                    )
                    backup_vocals_track = gr.Audio(
                        label="Backup vocals", type="filepath", interactive=False
                    )
            with gr.Accordion(
                "Step 3: main vocals cleanup", open=False
            ) as main_vocals_cleanup_accordion:
                with gr.Row():
                    main_vocals_dereverbed_track = gr.Audio(
                        label="De-reverbed main vocals",
                        type="filepath",
                        interactive=False,
                    )
                    main_vocals_reverb_track = gr.Audio(
                        label="Main vocals reverb", type="filepath", interactive=False
                    )
            with gr.Accordion(
                "Step 4: conversion of main vocals", open=False
            ) as vocals_conversion_accordion:
                ai_vocals_track = gr.Audio(
                    label="Converted vocals", type="filepath", interactive=False
                )
            with gr.Accordion(
                "Step 5: post-processing of converted vocals", open=False
            ) as vocals_postprocessing_accordion:
                mixed_ai_vocals_track = gr.Audio(
                    label="Post-processed vocals",
                    type="filepath",
                    interactive=False,
                )
            with gr.Accordion(
                "Step 6: Pitch shift of instrumentals and backup vocals",
                open=False,
            ) as pitch_shift_accordion:
                with gr.Row():
                    instrumentals_shifted_track = gr.Audio(
                        label="Pitch shifted instrumentals",
                        type="filepath",
                        interactive=False,
                    )
                    backup_vocals_shifted_track = gr.Audio(
                        label="Pitch shifted backup vocals",
                        type="filepath",
                        interactive=False,
                    )

        with gr.Row():
            clear_btn = gr.Button(
                value="Reset settings",
                scale=2,
            )
            generate_btn2.render()
            generate_btn.render()
            ai_cover = gr.Audio(label="Song cover", scale=3)
        show_intermediate_files.change(
            toggle_intermediate_files_accordion,
            inputs=show_intermediate_files,
            outputs=[
                intermediate_files_accordion,
                original_accordion,
                vocals_separation_accordion,
                main_vocals_separation_accordion,
                main_vocals_cleanup_accordion,
                vocals_conversion_accordion,
                vocals_postprocessing_accordion,
                pitch_shift_accordion,
                original_track,
                vocals_track,
                instrumentals_track,
                main_vocals_track,
                backup_vocals_track,
                main_vocals_dereverbed_track,
                main_vocals_reverb_track,
                ai_vocals_track,
                mixed_ai_vocals_track,
                instrumentals_shifted_track,
                backup_vocals_shifted_track,
            ],
            show_progress="hidden",
        )
        generate_event_args_list = [
            EventArgs(
                partial(exception_harness, update_audio_components),
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
                    backup_gain,
                    inst_gain,
                    output_sr,
                    output_format,
                    output_name,
                    keep_files,
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
                    ai_vocals_track,
                    mixed_ai_vocals_track,
                    instrumentals_shifted_track,
                    backup_vocals_shifted_track,
                    ai_cover,
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
        generate_btn_click = (
            setup_consecutive_event_listeners_with_toggled_interactivity(
                generate_btn,
                generate_event_args_list,
                generate_buttons + [show_intermediate_files],
            )
        )
        percentages = [i / 38 for i in range(38)]

        generate_btn2_event_args_list = [
            EventArgs(
                partial(
                    duplication_harness,
                    retrieve_song,
                    percentages=percentages[:4],
                ),
                inputs=[song_input],
                outputs=[ai_cover, original_track, current_song_dir],
            ),
            EventArgs(
                partial(
                    duplication_harness, separate_vocals, percentages=percentages[4:8]
                ),
                inputs=[original_track, current_song_dir],
                outputs=[ai_cover, vocals_track, instrumentals_track],
            ),
            EventArgs(
                partial(
                    duplication_harness,
                    separate_main_vocals,
                    percentages=percentages[8:12],
                ),
                inputs=[vocals_track, current_song_dir],
                outputs=[ai_cover, main_vocals_track, backup_vocals_track],
            ),
            EventArgs(
                partial(
                    duplication_harness,
                    dereverb_main_vocals,
                    percentages=percentages[12:16],
                ),
                inputs=[main_vocals_track, current_song_dir],
                outputs=[
                    ai_cover,
                    main_vocals_dereverbed_track,
                    main_vocals_reverb_track,
                ],
            ),
            EventArgs(
                partial(
                    duplication_harness,
                    convert_main_vocals,
                    percentages=percentages[16:20],
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
                outputs=[ai_cover, ai_vocals_track],
            ),
            EventArgs(
                partial(
                    duplication_harness,
                    postprocess_main_vocals,
                    percentages=percentages[20:24],
                ),
                inputs=[
                    ai_vocals_track,
                    current_song_dir,
                    reverb_rm_size,
                    reverb_wet,
                    reverb_dry,
                    reverb_damping,
                ],
                outputs=[ai_cover, mixed_ai_vocals_track],
            ),
            EventArgs(
                partial(
                    duplication_harness,
                    pitch_shift_background,
                    percentages=percentages[24:32],
                ),
                inputs=[
                    instrumentals_track,
                    backup_vocals_track,
                    current_song_dir,
                    pitch_change_all,
                ],
                outputs=[
                    ai_cover,
                    instrumentals_shifted_track,
                    backup_vocals_shifted_track,
                ],
            ),
            EventArgs(
                partial(
                    exception_harness,
                    mix_w_background_harness,
                    percentages=percentages[32:],
                ),
                inputs=[
                    mixed_ai_vocals_track,
                    instrumentals_track,
                    backup_vocals_track,
                    instrumentals_shifted_track,
                    backup_vocals_shifted_track,
                    current_song_dir,
                    main_gain,
                    backup_gain,
                    inst_gain,
                    output_sr,
                    output_format,
                    output_name,
                    keep_files,
                ],
                outputs=[ai_cover],
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
        generate_btn2_click = (
            setup_consecutive_event_listeners_with_toggled_interactivity(
                generate_btn2,
                generate_btn2_event_args_list,
                generate_buttons + [show_intermediate_files, clear_btn],
            )
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
                True,
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
                backup_gain,
                inst_gain,
                output_sr,
                output_format,
                keep_files,
                show_intermediate_files,
            ],
            show_progress="hidden",
        )
    with gr.Tab("Manage models"):

        # Download tab
        with gr.Tab("Download model"):

            with gr.Accordion("View public models table", open=False):

                gr.Markdown("")
                gr.Markdown("HOW TO USE")
                gr.Markdown("- Filter models using tags or search bar")
                gr.Markdown(
                    "- Select a row to autofill the download link and model name"
                )

                filter_tags = gr.CheckboxGroup(
                    value=[],
                    label="Show voice models with tags",
                    choices=load_public_model_tags(),
                )
                search_query = gr.Text(label="Search")

                public_models_table = gr.DataFrame(
                    value=load_public_models_table([]),
                    headers=[
                        "Model Name",
                        "Description",
                        "Tags",
                        "Credit",
                        "Added",
                        "URL",
                    ],
                    label="Available Public Models",
                    interactive=False,
                )

            with gr.Row():
                model_zip_link = gr.Text(
                    label="Download link to model",
                    info="Should point to a zip file containing a .pth model file and an optional .index file.",
                )
                model_name = gr.Text(
                    label="Model name",
                    info="Enter a unique name for the model.",
                )

            with gr.Row():
                download_btn = gr.Button("Download 🌐", variant="primary", scale=19)
                dl_output_message = gr.Text(
                    label="Output message", interactive=False, scale=20
                )

            download_button_click = download_btn.click(
                partial(exception_harness, download_online_model),
                inputs=[model_zip_link, model_name],
                outputs=dl_output_message,
            )

            public_models_table.select(
                pub_dl_autofill,
                inputs=public_models_table,
                outputs=[model_zip_link, model_name],
                show_progress="hidden",
            )
            search_query.change(
                partial(exception_harness, filter_public_models_table_harness),
                inputs=[filter_tags, search_query],
                outputs=public_models_table,
            )
            filter_tags.select(
                partial(exception_harness, filter_public_models_table_harness),
                inputs=[filter_tags, search_query],
                outputs=public_models_table,
            )

        # Upload tab
        with gr.Tab("Upload model"):
            with gr.Accordion("HOW TO USE"):
                gr.Markdown(
                    "- Find locally trained RVC v2 model file (weights folder) and optional index file (logs/[name] folder)"
                )
                gr.Markdown(
                    "- Upload model file and optional index file directly or compress into a zip file and upload that"
                )
                gr.Markdown("- Enter a unique name for the model")
                gr.Markdown("- Click 'Upload model'")

            with gr.Row():
                with gr.Column():
                    model_files = gr.File(label="Files", file_count="multiple")

                local_model_name = gr.Text(label="Model name")

            with gr.Row():
                model_upload_button = gr.Button(
                    "Upload model", variant="primary", scale=19
                )
                local_upload_output_message = gr.Text(
                    label="Output message", interactive=False, scale=20
                )
                model_upload_button_click = model_upload_button.click(
                    partial(exception_harness, upload_local_model),
                    inputs=[model_files, local_model_name],
                    outputs=local_upload_output_message,
                )

        with gr.Tab("Delete models"):
            with gr.Row():
                with gr.Column():
                    rvc_models_to_delete = gr.Dropdown(
                        voice_models,
                        label="Voice models",
                        filterable=True,
                        multiselect=True,
                    )
                with gr.Column():
                    rvc_models_deleted_message = gr.Text(
                        label="Output message", interactive=False
                    )

            with gr.Row():
                with gr.Column():
                    delete_models_button = gr.Button(
                        "Delete selected models", variant="secondary"
                    )
                    delete_all_models_button = gr.Button(
                        "Delete all models", variant="primary"
                    )
                with gr.Column():
                    pass
            delete_models_button_click = delete_models_button.click(
                # NOTE not sure why, but in order for subsequent event listener
                # to trigger, changes coming from the js code
                # have to be routed through an identity function which takes as
                # input some dummy component of type bool.
                lambda x: x,
                inputs=dummy_deletion_checkbox,
                outputs=delete_confirmation,
                js=confirm_box_js(
                    "Are you sure you want to delete the selected models?"
                ),
            ).then(
                partial(confirmation_harness, delete_models),
                inputs=[delete_confirmation, rvc_models_to_delete],
                outputs=rvc_models_deleted_message,
            )

            delete_all_models_btn_click = delete_all_models_button.click(
                lambda x: x,
                inputs=dummy_deletion_checkbox,
                outputs=delete_confirmation,
                js=confirm_box_js("Are you sure you want to delete all models?"),
            ).then(
                partial(confirmation_harness, delete_all_models),
                inputs=delete_confirmation,
                outputs=rvc_models_deleted_message,
            )

        for click_event in [
            download_button_click,
            model_upload_button_click,
            delete_models_button_click,
            delete_all_models_btn_click,
        ]:
            click_event.success(
                partial(update_model_lists, 3, [], [2]),
                outputs=[rvc_model, rvc_model2, rvc_models_to_delete],
            )


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
