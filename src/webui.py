import json
import os
import urllib.request
import zipfile
import shutil
from argparse import ArgumentParser


import gradio as gr

from main import song_cover_pipeline

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

mdxnet_models_dir = os.path.join(BASE_DIR, 'mdxnet_models')
rvc_models_dir = os.path.join(BASE_DIR, 'rvc_models')
output_dir = os.path.join(BASE_DIR, 'song_output')


def get_models_list(models_dir):
    models_list = os.listdir(models_dir)
    items_to_remove = ['hubert_base.pt', 'MODELS.txt']
    models_clear_list = [item for item in models_list if item not in items_to_remove]
    return models_clear_list


def update_models_list():
    models_l = get_models_list(rvc_models_dir)
    return gr.Dropdown.update(choices=models_l)


def download_and_extract_zip(url, dir_name, progress=gr.Progress()):
    try:
        progress(0, desc=f'[~] Downloading voice model with name {dir_name}...')
        zip_name = url.split('/')[-1]
        extraction_folder = os.path.join(rvc_models_dir, dir_name)
        if os.path.exists(extraction_folder):
            raise gr.Error(f'Voice model directory {dir_name} already exists! Choose a different name for your voice model.')
        urllib.request.urlretrieve(url, zip_name)

        progress(0.5, desc='[~] Extracting zip...')
        os.makedirs(extraction_folder)
        with zipfile.ZipFile(zip_name, 'r') as zip_ref:
            zip_ref.extractall(extraction_folder)
        os.remove(zip_name)

        index_filepath, model_filepath = None, None
        for root, dirs, files in os.walk(extraction_folder):
            for name in files:
                if name.endswith('.index'):
                    index_filepath = os.path.join(root, name)

                if name.endswith('.pth'):
                    model_filepath = os.path.join(root, name)

        if not model_filepath:
            raise gr.Error(f'No .pth model file was found in the extracted zip. Please check {extraction_folder}.')

        # move model and index file to extraction folder
        os.rename(model_filepath, os.path.join(extraction_folder, os.path.basename(model_filepath)))
        if index_filepath:
            os.rename(index_filepath, os.path.join(extraction_folder, os.path.basename(index_filepath)))

        # remove any unnecessary nested folders
        for filepath in os.listdir(extraction_folder):
            if os.path.isdir(os.path.join(extraction_folder, filepath)):
                shutil.rmtree(os.path.join(extraction_folder, filepath))

        update_models_list()
        return f'[+] {dir_name} Model successfully downloaded!'

    except Exception as e:
        raise gr.Error(str(e))


if __name__ == '__main__':
    parser = ArgumentParser(description='Generate a AI cover song in the song_output/id directory.', add_help=True)
    parser.add_argument("--share", action="store_true", dest="share_enabled", default=False, help="Enable sharing")
    args = parser.parse_args()
    share_enabled = args.share_enabled

    voice_models = get_models_list(rvc_models_dir)
    with gr.Blocks(title='AICoverGenWebUI') as app:
        
        gr.Label('AICoverGen WebUI created with ❤️', show_label=False)
       
        # main tab
        with gr.Tab("Generate"):
            with gr.Row():
                with gr.Column():

                    with gr.Row():
                        rvc_model = gr.Dropdown(voice_models, label='Voice Models', scale=10, info='Models folder "AICoverGen --> rvc_models". After new models are added into this folder, click the refresh button')
                        ref_btn = gr.Button('Refresh 🔁', variant='primary', scale=9)

                    with gr.Row():
                        video_link = gr.Text(label='YouTube link')
                        pitch = gr.Slider(-12, 12, value=0, step=1, label='Pitch', info='Pitch should be set to either -12, 0, or 12 to ensure the vocals are not out of tune.')

                    with gr.Row():
                        clear_btn = gr.ClearButton(value='Clear', components=[video_link, rvc_model, pitch])
                        generate_btn = gr.Button("Generate", variant='primary')

                song_cover = gr.Audio(label='Song Cover', show_share_button=False)
                ref_btn.click(update_models_list, None, outputs=rvc_model)
                is_webui = gr.Number(value=1, visible=False)
                generate_btn.click(song_cover_pipeline, inputs=[video_link, rvc_model, pitch, is_webui], outputs=[song_cover])
        
        # Download tab
        with gr.Tab("Download model"):
            with gr.Row():
                model_zip_link = gr.Text(label='Download link to model', info='Should be a zip file containing a .pth model file and an optional .index file.')
                model_name = gr.Text(label='Name your model', info='Give your new model a unique name from your other voice models.')

            with gr.Row():
                download_btn = gr.Button('Download 🌐', variant='primary', show_progress=True, scale=19)
                dl_output_message = gr.Text(label='Output Message', interactive=False, scale=20)

            download_btn.click(download_and_extract_zip, inputs=[model_zip_link, model_name], outputs=dl_output_message)

            gr.Markdown('## Input Examples')
            gr.Examples(
                [
                    ["https://huggingface.co/phant0m4r/LiSA/resolve/main/LiSA.zip", "Lisa"],
                    ["https://huggingface.co/Kit-Lemonfoot/kitlemonfoot_rvc_models/resolve/main/AZKi%20(Hybrid).zip", "Azki"]
                ],
                [model_zip_link, model_name],
                [],
                download_and_extract_zip,
            )

    app.launch(share=share_enabled, enable_queue=True)
