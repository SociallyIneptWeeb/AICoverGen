import json
import os
import shutil
import urllib.request
import zipfile
from argparse import ArgumentParser

import gradio as gr
import asyncio

from functools import partial

from main import (
    make_song_dir,
    retrieve_song,
    separate_vocals,
    separate_main_vocals,
    dereverb_main_vocals,
    convert_main_vocals,
    postprocess_main_vocals,
    pitch_shift_background,
    combine_w_background,
    song_cover_pipeline,
)

if os.name == "nt":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

progress_bar = gr.Progress()


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
    res = song_cover_pipeline(*args)
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


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

mdxnet_models_dir = os.path.join(BASE_DIR, "mdxnet_models")
rvc_models_dir = os.path.join(BASE_DIR, "rvc_models")
output_dir = os.path.join(BASE_DIR, "song_output")


def get_current_models(models_dir):
    models_list = os.listdir(models_dir)
    items_to_remove = ["hubert_base.pt", "MODELS.txt", "public_models.json", "rmvpe.pt"]
    return [item for item in models_list if item not in items_to_remove]


def update_models_list():
    models_l = get_current_models(rvc_models_dir)
    return gr.Dropdown(choices=models_l)


def load_public_models():
    models_table = []
    for model in public_models["voice_models"]:
        if not model["name"] in voice_models:
            model = [
                model["name"],
                model["description"],
                model["credit"],
                model["url"],
                ", ".join(model["tags"]),
            ]
            models_table.append(model)

    tags = list(public_models["tags"].keys())
    return gr.DataFrame(value=models_table), gr.CheckboxGroup(choices=tags)


def extract_zip(extraction_folder, zip_name, remove_zip):
    try:
        os.makedirs(extraction_folder)
        with zipfile.ZipFile(zip_name, "r") as zip_ref:
            zip_ref.extractall(extraction_folder)

        index_filepath, model_filepath = None, None
        for root, dirs, files in os.walk(extraction_folder):
            for name in files:
                if (
                    name.endswith(".index")
                    and os.stat(os.path.join(root, name)).st_size > 1024 * 100
                ):
                    index_filepath = os.path.join(root, name)

                if (
                    name.endswith(".pth")
                    and os.stat(os.path.join(root, name)).st_size > 1024 * 1024 * 40
                ):
                    model_filepath = os.path.join(root, name)

        if not model_filepath:
            raise Exception(f"No .pth model file was found in the extracted zip.")
        # move model and index file to extraction folder

        os.rename(
            model_filepath,
            os.path.join(extraction_folder, os.path.basename(model_filepath)),
        )
        if index_filepath:
            os.rename(
                index_filepath,
                os.path.join(extraction_folder, os.path.basename(index_filepath)),
            )
    except Exception as e:
        if os.path.exists(extraction_folder):
            shutil.rmtree(extraction_folder)
        raise e
    finally:
        if remove_zip and os.path.exists(zip_name):
            os.remove(zip_name)

    # remove any unnecessary nested folders
    for filepath in os.listdir(extraction_folder):
        if os.path.isdir(os.path.join(extraction_folder, filepath)):
            shutil.rmtree(os.path.join(extraction_folder, filepath))


def remove_suffix_after(text: str, occurrence: str):
    location = text.rfind(occurrence)
    if location == -1:
        return text
    else:
        return text[: location + len(occurrence)]


def download_online_model(url, dir_name, progress=gr.Progress()):
    try:
        progress(0, desc=f"[~] Downloading voice model with name {dir_name}...")
        if not url:
            raise Exception("Voice model link missing!")
        zip_name = remove_suffix_after(url.split("/")[-1], ".zip")
        if not zip_name.endswith(".zip"):
            raise Exception("Link does not point to a valid zip file!")
        if not dir_name:
            raise Exception("Voice model name missing!")
        extraction_folder = os.path.join(rvc_models_dir, dir_name)
        if os.path.exists(extraction_folder):
            raise Exception(
                f"Voice model directory {dir_name} already exists! Choose a different name for your voice model."
            )

        if "pixeldrain.com" in url:
            url = f"https://pixeldrain.com/api/file/{zip_name}"

        urllib.request.urlretrieve(url, zip_name)

        progress(0.5, desc="[~] Extracting zip...")
        extract_zip(extraction_folder, zip_name, remove_zip=True)
        return f"[+] {dir_name} Model successfully downloaded!"

    except Exception as e:
        raise gr.Error(str(e))


def upload_local_model(zip_path, dir_name, progress=gr.Progress()):
    try:
        if not zip_path:
            raise Exception("No voice model selected.")
        if not dir_name:
            raise Exception("No name given for voice model.")
        extraction_folder = os.path.join(rvc_models_dir, dir_name)
        if os.path.exists(extraction_folder):
            raise Exception(
                f"Voice model directory {dir_name} already exists! Choose a different name for your voice model."
            )

        zip_name = zip_path.name
        progress(0.5, desc="[~] Extracting zip...")
        extract_zip(extraction_folder, zip_name, remove_zip=False)
        return f"[+] {dir_name} Model successfully uploaded!"

    except Exception as e:
        raise gr.Error(str(e))


def filter_models(tags, query):
    models_table = []

    # no filter
    if len(tags) == 0 and len(query) == 0:
        for model in public_models["voice_models"]:
            models_table.append(
                [
                    model["name"],
                    model["description"],
                    model["credit"],
                    model["url"],
                    model["tags"],
                ]
            )

    # filter based on tags and query
    elif len(tags) > 0 and len(query) > 0:
        for model in public_models["voice_models"]:
            if all(tag in model["tags"] for tag in tags):
                model_attributes = f"{model['name']} {model['description']} {model['credit']} {' '.join(model['tags'])}".lower()
                if query.lower() in model_attributes:
                    models_table.append(
                        [
                            model["name"],
                            model["description"],
                            model["credit"],
                            model["url"],
                            model["tags"],
                        ]
                    )

    # filter based on only tags
    elif len(tags) > 0:
        for model in public_models["voice_models"]:
            if all(tag in model["tags"] for tag in tags):
                models_table.append(
                    [
                        model["name"],
                        model["description"],
                        model["credit"],
                        model["url"],
                        model["tags"],
                    ]
                )

    # filter based on only query
    else:
        for model in public_models["voice_models"]:
            model_attributes = f"{model['name']} {model['description']} {model['credit']} {' '.join(model['tags'])}".lower()
            if query.lower() in model_attributes:
                models_table.append(
                    [
                        model["name"],
                        model["description"],
                        model["credit"],
                        model["url"],
                        model["tags"],
                    ]
                )

    return gr.DataFrame(value=models_table)


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
    )


def process_file_upload(file):
    return file.name, gr.update(value=file.name)


def show_hop_slider(pitch_detection_algo):
    if pitch_detection_algo == "mangio-crepe":
        return gr.update(visible=True)
    else:
        return gr.update(visible=False)


def toggle_intermediate_files_accordion(visible):
    audio_components = (None,) * 10
    accordions = (gr.update(open=False),) * 7
    return (gr.update(visible=visible, open=False),) + accordions + audio_components


voice_models = get_current_models(rvc_models_dir)
with open(
    os.path.join(rvc_models_dir, "public_models.json"), encoding="utf8"
) as infile:
    public_models = json.load(infile)

with gr.Blocks(title="AICoverGenWebUI") as app:

    gr.Label("AICoverGen WebUI created with ❤️", show_label=False)

    # main tab
    with gr.Tab("Generate"):

        with gr.Accordion("Main Options"):
            with gr.Row():
                with gr.Column():
                    rvc_model = gr.Dropdown(
                        voice_models,
                        label="Voice Models",
                        info='Models folder "AICoverGen --> rvc_models". After new models are added into this folder, click the refresh button',
                    )
                    ref_btn = gr.Button("Refresh Models 🔁", variant="primary")

                with gr.Column() as yt_link_col:
                    song_input = gr.Text(
                        label="Song input",
                        info="Link to a song on YouTube or full path to a local file. For file upload, click the button below.",
                    )
                    show_file_upload_button = gr.Button("Upload file instead")

                with gr.Column(visible=False) as file_upload_col:
                    local_file = gr.File(label="Audio file")
                    song_input_file = gr.UploadButton(
                        "Upload 📂", file_types=["audio"], variant="primary"
                    )
                    show_yt_link_button = gr.Button(
                        "Paste YouTube link/Path to local file instead"
                    )
                    song_input_file.upload(
                        process_file_upload,
                        inputs=[song_input_file],
                        outputs=[local_file, song_input],
                    )

                with gr.Column():
                    pitch = gr.Slider(
                        -3,
                        3,
                        value=0,
                        step=1,
                        label="Pitch Change (Vocals ONLY)",
                        info="Generally, use 1 for male to female conversions and -1 for vice-versa. (Octaves)",
                    )
                    pitch_all = gr.Slider(
                        -12,
                        12,
                        value=0,
                        step=1,
                        label="Overall Pitch Change",
                        info="Changes pitch/key of vocals and instrumentals together. Altering this slightly reduces sound quality. (Semitones)",
                    )
                show_file_upload_button.click(
                    swap_visibility,
                    outputs=[file_upload_col, yt_link_col, song_input, local_file],
                )
                show_yt_link_button.click(
                    swap_visibility,
                    outputs=[yt_link_col, file_upload_col, song_input, local_file],
                )

        with gr.Accordion("Voice conversion options", open=False):
            with gr.Row():
                index_rate = gr.Slider(
                    0,
                    1,
                    value=0.5,
                    label="Index Rate",
                    info="Controls how much of the AI voice's accent to keep in the vocals",
                )
                filter_radius = gr.Slider(
                    0,
                    7,
                    value=3,
                    step=1,
                    label="Filter radius",
                    info="If >=3: apply median filtering median filtering to the harvested pitch results. Can reduce breathiness",
                )
                rms_mix_rate = gr.Slider(
                    0,
                    1,
                    value=0.25,
                    label="RMS mix rate",
                    info="Control how much to mimic the original vocal's loudness (0) or a fixed loudness (1)",
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
                        show_hop_slider, inputs=f0_method, outputs=crepe_hop_length
                    )
        with gr.Accordion("Audio mixing options", open=False):
            gr.Markdown("### Volume Change (decibels)")
            with gr.Row():
                main_gain = gr.Slider(-20, 20, value=0, step=1, label="Main Vocals")
                backup_gain = gr.Slider(-20, 20, value=0, step=1, label="Backup Vocals")
                inst_gain = gr.Slider(-20, 20, value=0, step=1, label="Music")

            gr.Markdown("### Reverb Control on AI Vocals")
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
                    info="Level of AI vocals with reverb",
                )
                reverb_dry = gr.Slider(
                    0,
                    1,
                    value=0.8,
                    label="Dryness level",
                    info="Level of AI vocals without reverb",
                )
                reverb_damping = gr.Slider(
                    0,
                    1,
                    value=0.7,
                    label="Damping level",
                    info="Absorption of high frequencies in the reverb",
                )
        with gr.Accordion("Audio output options", open=False):
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
                label="Output file type",
            )
            output_sr = gr.Radio(
                choices=[16000, 44100, 48000, 96000, 192000],
                value=44100,
                label="Output sample rate",
            )
        with gr.Accordion("Intermediate file options", open=False):
            with gr.Row():
                keep_files = gr.Checkbox(
                    label="Keep intermediate files",
                    value=True,
                    info="Keep all intermediate audio files generated in the song_output/id directory. Leave unchecked to save space",
                )
                show_intermediate_files = gr.Checkbox(
                    label="Show intermediate files",
                    value=False,
                    info="Show available intermediate audio files when audio generation completes. Leave unchecked to optimize performance.",
                )
        intermediate_files_accordion = gr.Accordion(
            "Access intermediate files", open=False, visible=False
        )
        with intermediate_files_accordion:
            original_accordion = gr.Accordion(
                "Step 0: Input",
                open=False,
            )
            with original_accordion:
                original_track = gr.Audio(
                    label="Original song", type="filepath", interactive=False
                )
            vocals_separation_accordion = gr.Accordion(
                "Step 1: Instruments/vocals separation",
                open=False,
            )
            with vocals_separation_accordion:
                with gr.Row():
                    instrumentals_track = gr.Audio(
                        label="Instruments", type="filepath", interactive=False
                    )
                    vocals_track = gr.Audio(
                        label="Vocals", type="filepath", interactive=False
                    )
            main_vocals_separation_accordion = gr.Accordion(
                "Step 2: Main vocals/ background vocals separation",
                open=False,
            )
            with main_vocals_separation_accordion:
                with gr.Row():
                    main_vocals_track = gr.Audio(
                        label="Main vocals", type="filepath", interactive=False
                    )
                    background_vocals_track = gr.Audio(
                        label="Background vocals", type="filepath", interactive=False
                    )
            main_vocals_cleanup_accordion = gr.Accordion(
                "Step 3: Main vocals cleanup",
                open=False,
            )
            with main_vocals_cleanup_accordion:
                main_vocals_dereverbed_track = gr.Audio(
                    label="Vocals dereverbed", type="filepath", interactive=False
                )
            voice_conversion_accordion = gr.Accordion(
                "Step 4: Voice conversion",
                open=False,
            )
            with voice_conversion_accordion:
                ai_vocals_track = gr.Audio(
                    label="AI vocals", type="filepath", interactive=False
                )
            voice_postprocessing_accordion = gr.Accordion(
                "Step 5: Voice post-processing",
                open=False,
            )
            with voice_postprocessing_accordion:
                mixed_ai_vocals_track = gr.Audio(
                    label="Post-processed AI vocals", type="filepath", interactive=False
                )
            pitch_shift_accordion = gr.Accordion(
                "Step 6: Pitch shift of instrumentals and background vocals",
                open=False,
            )
            with pitch_shift_accordion:
                with gr.Row():
                    instrumentals_shifted_track = gr.Audio(
                        label="Pitch-shifted instrumentals",
                        type="filepath",
                        interactive=False,
                    )
                    background_vocals_shifted_track = gr.Audio(
                        label="Pitch-shifted background vocals",
                        type="filepath",
                        interactive=False,
                    )

        with gr.Row():
            clear_btn = gr.ClearButton(
                value="Clear",
                components=[song_input, rvc_model, local_file],
                scale=2,
            )
            generate_btn2 = gr.Button(
                "Generate step-by-step", variant="primary", scale=1, visible=False
            )
            generate_btn = gr.Button("Generate", variant="primary", scale=2)
            ai_cover = gr.Audio(label="AI Cover", scale=3)
        show_intermediate_files.change(
            toggle_intermediate_files_accordion,
            inputs=show_intermediate_files,
            outputs=[
                intermediate_files_accordion,
                original_accordion,
                vocals_separation_accordion,
                main_vocals_separation_accordion,
                main_vocals_cleanup_accordion,
                voice_conversion_accordion,
                voice_postprocessing_accordion,
                pitch_shift_accordion,
                original_track,
                vocals_track,
                instrumentals_track,
                main_vocals_track,
                background_vocals_track,
                main_vocals_dereverbed_track,
                ai_vocals_track,
                mixed_ai_vocals_track,
                instrumentals_shifted_track,
                background_vocals_shifted_track,
            ],
        )

        ref_btn.click(update_models_list, None, outputs=rvc_model)
        song_dir = gr.State()
        input_type = gr.State()
        generate_btn.click(
            lambda: (gr.update(interactive=False),) * 2,
            inputs=[],
            outputs=[show_intermediate_files, generate_btn2],
            show_progress=False,
        ).success(
            partial(exception_harness, update_audio_components),
            inputs=[
                song_input,
                rvc_model,
                pitch,
                keep_files,
                show_intermediate_files,
                main_gain,
                backup_gain,
                inst_gain,
                index_rate,
                filter_radius,
                rms_mix_rate,
                f0_method,
                crepe_hop_length,
                protect,
                pitch_all,
                reverb_rm_size,
                reverb_wet,
                reverb_dry,
                reverb_damping,
                output_format,
                output_sr,
            ],
            outputs=[
                original_track,
                vocals_track,
                instrumentals_track,
                main_vocals_track,
                background_vocals_track,
                main_vocals_dereverbed_track,
                ai_vocals_track,
                mixed_ai_vocals_track,
                instrumentals_shifted_track,
                background_vocals_shifted_track,
                ai_cover,
            ],
        ).then(
            lambda: (gr.update(interactive=True),) * 2,
            inputs=[],
            outputs=[show_intermediate_files, generate_btn2],
            show_progress=False,
        )

        generate_btn2.click(
            lambda: (gr.update(interactive=False),) * 4,
            inputs=[],
            outputs=[show_intermediate_files, generate_btn, generate_btn2, clear_btn],
            show_progress=False,
        ).success(
            partial(exception_harness, make_song_dir),
            inputs=[
                song_input,
                rvc_model,
            ],
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
            outputs=[ai_cover, background_vocals_track, main_vocals_track],
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
                pitch,
                pitch_all,
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
                background_vocals_track,
                song_dir,
                pitch_all,
            ],
            outputs=[
                ai_cover,
                instrumentals_shifted_track,
                background_vocals_shifted_track,
            ],
        ).success(
            partial(exception_harness, combine_w_background_harness),
            inputs=[
                instrumentals_track,
                background_vocals_track,
                instrumentals_shifted_track,
                background_vocals_shifted_track,
                original_track,
                mixed_ai_vocals_track,
                rvc_model,
                song_dir,
                main_gain,
                backup_gain,
                inst_gain,
                output_format,
                output_sr,
                keep_files,
            ],
            outputs=[ai_cover],
        ).then(
            lambda: (gr.update(interactive=True),) * 4,
            inputs=[],
            outputs=[show_intermediate_files, generate_btn, generate_btn2, clear_btn],
            show_progress=False,
        )
        clear_btn.click(
            lambda: [
                0,
                0,
                0,
                0,
                0.5,
                3,
                0.25,
                0.33,
                "rmvpe",
                128,
                0,
                0.15,
                0.2,
                0.8,
                0.7,
                "mp3",
                44100,
                True,
                False,
            ],
            outputs=[
                pitch,
                main_gain,
                backup_gain,
                inst_gain,
                index_rate,
                filter_radius,
                rms_mix_rate,
                protect,
                f0_method,
                crepe_hop_length,
                pitch_all,
                reverb_rm_size,
                reverb_wet,
                reverb_dry,
                reverb_damping,
                output_format,
                output_sr,
                keep_files,
                show_intermediate_files,
            ],
        )

    # Download tab
    with gr.Tab("Download model"):

        with gr.Tab("From HuggingFace/Pixeldrain URL"):
            with gr.Row():
                model_zip_link = gr.Text(
                    label="Download link to model",
                    info="Should be a zip file containing a .pth model file and an optional .index file.",
                )
                model_name = gr.Text(
                    label="Name your model",
                    info="Give your new model a unique name from your other voice models.",
                )

            with gr.Row():
                download_btn = gr.Button("Download 🌐", variant="primary", scale=19)
                dl_output_message = gr.Text(
                    label="Output Message", interactive=False, scale=20
                )

            download_btn.click(
                download_online_model,
                inputs=[model_zip_link, model_name],
                outputs=dl_output_message,
            )

            gr.Markdown("## Input Examples")
            gr.Examples(
                [
                    [
                        "https://huggingface.co/phant0m4r/LiSA/resolve/main/LiSA.zip",
                        "Lisa",
                    ],
                    ["https://pixeldrain.com/u/3tJmABXA", "Gura"],
                    [
                        "https://huggingface.co/Kit-Lemonfoot/kitlemonfoot_rvc_models/resolve/main/AZKi%20(Hybrid).zip",
                        "Azki",
                    ],
                ],
                [model_zip_link, model_name],
                [],
                download_online_model,
            )

        with gr.Tab("From Public Index"):

            gr.Markdown("## How to use")
            gr.Markdown("- Click Initialize public models table")
            gr.Markdown("- Filter models using tags or search bar")
            gr.Markdown("- Select a row to autofill the download link and model name")
            gr.Markdown("- Click Download")

            with gr.Row():
                pub_zip_link = gr.Text(label="Download link to model")
                pub_model_name = gr.Text(label="Model name")

            with gr.Row():
                download_pub_btn = gr.Button("Download 🌐", variant="primary", scale=19)
                pub_dl_output_message = gr.Text(
                    label="Output Message", interactive=False, scale=20
                )

            filter_tags = gr.CheckboxGroup(
                value=[], label="Show voice models with tags", choices=[]
            )
            search_query = gr.Text(label="Search")
            load_public_models_button = gr.Button(
                value="Initialize public models table", variant="primary"
            )

            public_models_table = gr.DataFrame(
                value=[],
                headers=["Model Name", "Description", "Credit", "URL", "Tags"],
                label="Available Public Models",
                interactive=False,
            )
            public_models_table.select(
                pub_dl_autofill,
                inputs=[public_models_table],
                outputs=[pub_zip_link, pub_model_name],
            )
            load_public_models_button.click(
                load_public_models, outputs=[public_models_table, filter_tags]
            )
            search_query.change(
                filter_models,
                inputs=[filter_tags, search_query],
                outputs=public_models_table,
            )
            filter_tags.select(
                filter_models,
                inputs=[filter_tags, search_query],
                outputs=public_models_table,
            )
            download_pub_btn.click(
                download_online_model,
                inputs=[pub_zip_link, pub_model_name],
                outputs=pub_dl_output_message,
            )

    # Upload tab
    with gr.Tab("Upload model"):
        gr.Markdown("## Upload locally trained RVC v2 model and index file")
        gr.Markdown(
            "- Find model file (weights folder) and optional index file (logs/[name] folder)"
        )
        gr.Markdown("- Compress files into zip file")
        gr.Markdown("- Upload zip file and give unique name for voice")
        gr.Markdown("- Click Upload model")

        with gr.Row():
            with gr.Column():
                zip_file = gr.File(label="Zip file")

            local_model_name = gr.Text(label="Model name")

        with gr.Row():
            model_upload_button = gr.Button("Upload model", variant="primary", scale=19)
            local_upload_output_message = gr.Text(
                label="Output Message", interactive=False, scale=20
            )
            model_upload_button.click(
                upload_local_model,
                inputs=[zip_file, local_model_name],
                outputs=local_upload_output_message,
            )


if __name__ == "__main__":

    parser = ArgumentParser(
        description="Generate a AI cover song in the song_output/id directory.",
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
