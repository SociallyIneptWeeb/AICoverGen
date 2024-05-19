import os
from argparse import ArgumentParser

import gradio as gr
import asyncio

from functools import partial

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
    get_cached_input_paths_named,
    make_song_dir,
    retrieve_song,
    separate_vocals,
    separate_main_vocals,
    dereverb_main_vocals,
    convert_main_vocals,
    postprocess_main_vocals,
    pitch_shift_background,
    combine_w_background,
    run_pipeline,
)

if os.name == "nt":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

progress_bar = gr.Progress()


def confirmation_harness(fun, confirm, *args):
    if confirm:
        return exception_harness(fun, *args)
    else:
        raise gr.Error("Confirmation missing!")


def exception_harness(fun, *args):
    new_args = args + (progress_bar,)
    try:
        return fun(*new_args)
    except Exception as e:
        raise gr.Error(str(e))


def duplication_harness(fun, *args):

    res = exception_harness(fun, *args)
    if not isinstance(res, tuple):
        return (res, res)
    else:
        return (res[0],) + res


def update_audio_components(*args):
    res = run_pipeline(*args)
    if isinstance(res, tuple):
        return res
    else:
        return (None,) * 10 + (res,)


def combine_w_background_harness(
    instrumentals_path,
    backup_vocals_path,
    instrumentals_shifted_path,
    backup_vocals_shifted_path,
    *args,
):
    return combine_w_background(
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


def update_model_lists():
    models_l = get_current_models()
    return gr.Dropdown(choices=models_l), gr.Dropdown(choices=models_l, value=[])


def update_cached_input_songs():
    songs_l = get_cached_input_paths_named()
    return gr.Dropdown(choices=songs_l)


def pub_dl_autofill(pub_models, event: gr.SelectData):
    return gr.Text(value=pub_models.loc[event.index[0], "URL"]), gr.Text(
        value=pub_models.loc[event.index[0], "Model Name"]
    )


def swap_visibility():
    return (
        gr.update(visible=True),
        gr.update(visible=False),
        gr.update(value=""),
        gr.update(value=None),
        gr.update(value=None),
    )


def show_hop_slider(pitch_detection_algo):
    if pitch_detection_algo == "mangio-crepe":
        return gr.update(visible=True)
    else:
        return gr.update(visible=False)


def toggle_intermediate_files_accordion(visible):
    audio_components = (None,) * 10
    accordions = (gr.update(open=False),) * 7
    return (gr.update(visible=visible, open=False),) + accordions + audio_components


voice_models = get_current_models()
cached_input_songs = get_cached_input_paths_named()

with gr.Blocks(title="Ultimate RVC") as app:

    gr.Label("Ultimate RVC ❤️", show_label=False)

    # main tab
    with gr.Tab("Generate song covers"):

        with gr.Accordion("Main options"):
            with gr.Row():
                with gr.Column() as yt_link_col:
                    song_input = gr.Text(
                        label="Song input",
                        info="Link to a song on YouTube or the full path of a local audio file. For file upload, click the button below.",
                    )
                    cached_input_songs_dropdown = gr.Dropdown(
                        cached_input_songs,
                        label="Cached input songs",
                    )
                    song_input.input(
                        lambda: gr.update(value=None),
                        outputs=cached_input_songs_dropdown,
                        show_progress="hidden",
                    )
                    cached_input_songs_dropdown.input(
                        lambda x: gr.update(value=x),
                        inputs=cached_input_songs_dropdown,
                        outputs=song_input,
                        show_progress="hidden",
                    )
                    show_file_upload_button = gr.Button("Upload local file instead")

                with gr.Column(visible=False) as file_upload_col:
                    local_file = gr.Audio(
                        label="Song input", sources="upload", type="filepath"
                    )
                    show_yt_link_button = gr.Button(
                        "Paste YouTube link/path to local file instead"
                    )
                    local_file.change(
                        lambda x: gr.update(value=x),
                        inputs=local_file,
                        outputs=song_input,
                    )

                with gr.Column():
                    rvc_model = gr.Dropdown(
                        voice_models,
                        label="Voice model",
                    )

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
                show_file_upload_button.click(
                    swap_visibility,
                    outputs=[
                        file_upload_col,
                        yt_link_col,
                        song_input,
                        local_file,
                        cached_input_songs_dropdown,
                    ],
                    show_progress="hidden",
                )
                show_yt_link_button.click(
                    swap_visibility,
                    outputs=[
                        yt_link_col,
                        file_upload_col,
                        song_input,
                        local_file,
                        cached_input_songs_dropdown,
                    ],
                    show_progress="hidden",
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
            output_sr = gr.Radio(
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
        with gr.Accordion("Intermediate audio options", open=False):
            with gr.Row():
                keep_files = gr.Checkbox(
                    label="Keep intermediate audio files",
                    value=True,
                    info="Keep all intermediate audio files. Leave unchecked to save space",
                )
                show_intermediate_files = gr.Checkbox(
                    label="Show intermediate audio files",
                    value=False,
                    info="Show available intermediate audio files when audio generation completes. Leave unchecked to optimize performance.",
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
                    instrumentals_track = gr.Audio(
                        label="Instrumentals", type="filepath", interactive=False
                    )
                    vocals_track = gr.Audio(
                        label="Vocals", type="filepath", interactive=False
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
                main_vocals_dereverbed_track = gr.Audio(
                    label="De-reverbed main vocals", type="filepath", interactive=False
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
            generate_btn2 = gr.Button(
                "Generate step-by-step", variant="primary", scale=1, visible=False
            )
            generate_btn = gr.Button("Generate", variant="primary", scale=2)
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
                ai_vocals_track,
                mixed_ai_vocals_track,
                instrumentals_shifted_track,
                backup_vocals_shifted_track,
            ],
            show_progress="hidden",
        )
        song_dir = gr.State()
        input_type = gr.State()
        generate_btn.click(
            lambda: (gr.update(interactive=False),) * 2,
            outputs=[show_intermediate_files, generate_btn2],
            show_progress="hidden",
        ).success(
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
                ai_vocals_track,
                mixed_ai_vocals_track,
                instrumentals_shifted_track,
                backup_vocals_shifted_track,
                ai_cover,
            ],
        ).then(
            update_cached_input_songs,
            outputs=cached_input_songs_dropdown,
        ).then(
            lambda: (gr.update(interactive=True),) * 2,
            outputs=[show_intermediate_files, generate_btn2],
            show_progress="hidden",
        )

        generate_btn2.click(
            lambda: (gr.update(interactive=False),) * 4,
            outputs=[show_intermediate_files, generate_btn, generate_btn2, clear_btn],
            show_progress="hidden",
        ).success(
            partial(exception_harness, make_song_dir),
            inputs=[song_input, rvc_model],
            outputs=[song_dir, input_type],
        ).success(
            partial(duplication_harness, retrieve_song),
            inputs=[song_input, input_type, song_dir],
            outputs=[ai_cover, original_track],
        ).success(
            partial(duplication_harness, separate_vocals),
            inputs=[original_track, song_dir],
            outputs=[ai_cover, vocals_track, instrumentals_track],
        ).success(
            partial(duplication_harness, separate_main_vocals),
            inputs=[vocals_track, song_dir],
            outputs=[ai_cover, backup_vocals_track, main_vocals_track],
        ).success(
            partial(duplication_harness, dereverb_main_vocals),
            inputs=[main_vocals_track, song_dir],
            outputs=[ai_cover, main_vocals_dereverbed_track],
        ).success(
            partial(duplication_harness, convert_main_vocals),
            inputs=[
                main_vocals_dereverbed_track,
                song_dir,
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
        ).success(
            partial(duplication_harness, postprocess_main_vocals),
            inputs=[
                ai_vocals_track,
                song_dir,
                reverb_rm_size,
                reverb_wet,
                reverb_dry,
                reverb_damping,
            ],
            outputs=[ai_cover, mixed_ai_vocals_track],
        ).success(
            partial(duplication_harness, pitch_shift_background),
            inputs=[
                instrumentals_track,
                backup_vocals_track,
                song_dir,
                pitch_change_all,
            ],
            outputs=[
                ai_cover,
                instrumentals_shifted_track,
                backup_vocals_shifted_track,
            ],
        ).success(
            partial(exception_harness, combine_w_background_harness),
            inputs=[
                instrumentals_track,
                backup_vocals_track,
                instrumentals_shifted_track,
                backup_vocals_shifted_track,
                original_track,
                mixed_ai_vocals_track,
                song_dir,
                rvc_model,
                main_gain,
                backup_gain,
                inst_gain,
                output_sr,
                output_format,
                keep_files,
            ],
            outputs=ai_cover,
        ).then(
            update_cached_input_songs,
            outputs=cached_input_songs_dropdown,
        ).then(
            lambda: (gr.update(interactive=True),) * 4,
            outputs=[show_intermediate_files, generate_btn, generate_btn2, clear_btn],
            show_progress="hidden",
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
                    value=load_public_models_table([], progress_bar),
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
            model_delete_confirmation = gr.State(False)
            dummy_deletion_checkbox = gr.Checkbox(visible=False)
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
                        "Delete selected models", variant="primary"
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
                outputs=model_delete_confirmation,
                js=confirm_box_js(
                    "Are you sure you want to delete the selected models?"
                ),
            ).then(
                partial(confirmation_harness, delete_models),
                inputs=[model_delete_confirmation, rvc_models_to_delete],
                outputs=rvc_models_deleted_message,
            )

            delete_all_models_btn_click = delete_all_models_button.click(
                lambda x: x,
                inputs=dummy_deletion_checkbox,
                outputs=model_delete_confirmation,
                js=confirm_box_js("Are you sure you want to delete all models?"),
            ).then(
                partial(confirmation_harness, delete_all_models),
                inputs=model_delete_confirmation,
                outputs=rvc_models_deleted_message,
            )

        for click_event in [
            download_button_click,
            model_upload_button_click,
            delete_models_button_click,
            delete_all_models_btn_click,
        ]:
            click_event.success(
                update_model_lists,
                outputs=[rvc_model, rvc_models_to_delete],
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
