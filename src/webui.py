import json
import os
import shutil
import urllib.request
import zipfile
from argparse import ArgumentParser

import gradio as gr

from main import song_cover_pipeline

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

mdxnet_models_dir = os.path.join(BASE_DIR, 'mdxnet_models')
rvc_models_dir = os.path.join(BASE_DIR, 'rvc_models')
output_dir = os.path.join(BASE_DIR, 'song_output')


def get_current_models(models_dir):
    models_list = os.listdir(models_dir)
    items_to_remove = ['hubert_base.pt', 'MODELS.txt', 'public_models.json', 'rmvpe.pt']
    return [item for item in models_list if item not in items_to_remove]


def update_models_list():
    models_l = get_current_models(rvc_models_dir)
    return gr.Dropdown.update(choices=models_l)


def load_public_models():
    models_table = []
    for model in public_models['voice_models']:
        if not model['name'] in voice_models:
            model = [model['name'], model['description'], model['credit'], model['url'], ', '.join(model['tags'])]
            models_table.append(model)

    tags = list(public_models['tags'].keys())
    return gr.DataFrame.update(value=models_table), gr.CheckboxGroup.update(choices=tags)


def download_and_extract_zip(url, dir_name, progress=gr.Progress()):
    try:
        progress(0, desc=f'[~] Downloading voice model with name {dir_name}...')
        zip_name = url.split('/')[-1]
        extraction_folder = os.path.join(rvc_models_dir, dir_name)
        if os.path.exists(extraction_folder):
            raise gr.Error(f'Voice model directory {dir_name} already exists! Choose a different name for your voice model.')

        if 'pixeldrain.com' in url:
            url = f'https://pixeldrain.com/api/file/{zip_name}'

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


def filter_models(tags, query):
    models_table = []

    # no filter
    if len(tags) == 0 and len(query) == 0:
        for model in public_models['voice_models']:
            models_table.append([model['name'], model['description'], model['credit'], model['url'], model['tags']])

    # filter based on tags and query
    elif len(tags) > 0 and len(query) > 0:
        for model in public_models['voice_models']:
            if all(tag in model['tags'] for tag in tags):
                model_attributes = f"{model['name']} {model['description']} {model['credit']} {' '.join(model['tags'])}".lower()
                if query.lower() in model_attributes:
                    models_table.append([model['name'], model['description'], model['credit'], model['url'], model['tags']])

    # filter based on only tags
    elif len(tags) > 0:
        for model in public_models['voice_models']:
            if all(tag in model['tags'] for tag in tags):
                models_table.append([model['name'], model['description'], model['credit'], model['url'], model['tags']])

    # filter based on only query
    else:
        for model in public_models['voice_models']:
            model_attributes = f"{model['name']} {model['description']} {model['credit']} {' '.join(model['tags'])}".lower()
            if query.lower() in model_attributes:
                models_table.append([model['name'], model['description'], model['credit'], model['url'], model['tags']])

    return gr.DataFrame.update(value=models_table)


def pub_dl_autofill(pub_models, event: gr.SelectData):
    return gr.Text.update(value=pub_models.loc[event.index[0], 'URL']), gr.Text.update(value=pub_models.loc[event.index[0], 'Model Name'])


if __name__ == '__main__':
    parser = ArgumentParser(description='Generate a AI cover song in the song_output/id directory.', add_help=True)
    parser.add_argument("--share", action="store_true", dest="share_enabled", default=False, help="Enable sharing")
    args = parser.parse_args()
    share_enabled = args.share_enabled

    voice_models = get_current_models(rvc_models_dir)
    with open(os.path.join(rvc_models_dir, 'public_models.json')) as infile:
        public_models = json.load(infile)

    with gr.Blocks(title='AICoverGenWebUI') as app:

        gr.Label('AICoverGen WebUI created with ❤️', show_label=False)

        # main tab
        with gr.Tab("Generate"):

            with gr.Accordion('Main Options'):
                with gr.Row():
                    rvc_model = gr.Dropdown(voice_models, label='Voice Models', info='Models folder "AICoverGen --> rvc_models". After new models are added into this folder, click the refresh button')
                    ref_btn = gr.Button('Refresh Models 🔁', variant='primary')
                    song_input = gr.Text(label='Song Input', info='Link to a YouTube video or the full path to a local audio file')
                    pitch = gr.Slider(-24, 24, value=0, step=1, label='Pitch Change', info='Pitch Change should be set to either -12, 0, or 12 (multiples of 12) to ensure the vocals are not out of tune')

            with gr.Accordion('Voice conversion options', open=False):
                with gr.Row():
                    keep_files = gr.Checkbox(label='Keep intermediate files', info='Keep all audio files generated in the song_output/id directory, e.g. Isolated Vocals/Instrumentals. Leave unchecked to save space')
                    index_rate = gr.Slider(0, 1, value=0.5, label='Index Rate', info="Controls how much of the AI voice's accent to keep in the vocals")

            with gr.Accordion('Audio mixing options', open=False):
                gr.Markdown('### Volume Change (decibels)')
                with gr.Row():
                    main_gain = gr.Slider(-20, 20, value=0, step=1, label='Main Vocals')
                    backup_gain = gr.Slider(-20, 20, value=0, step=1, label='Backup Vocals')
                    inst_gain = gr.Slider(-20, 20, value=0, step=1, label='Music')

                gr.Markdown('### Reverb Control on AI Vocals')
                with gr.Row():
                    reverb_rm_size = gr.Slider(0, 1, value=0.15, label='Room size', info='The larger the room, the longer the reverb time')
                    reverb_wet = gr.Slider(0, 1, value=0.2, label='Wetness level', info='Level of AI vocals with reverb')
                    reverb_dry = gr.Slider(0, 1, value=0.8, label='Dryness level', info='Level of AI vocals without reverb')
                    reverb_damping = gr.Slider(0, 1, value=0.7, label='Damping level', info='Absorption of high frequencies in the reverb')

            with gr.Row():
                clear_btn = gr.ClearButton(value='Clear', components=[song_input, rvc_model, keep_files])
                generate_btn = gr.Button("Generate", variant='primary')
                ai_cover = gr.Audio(label='AI Cover', show_share_button=False)

            ref_btn.click(update_models_list, None, outputs=rvc_model)
            is_webui = gr.Number(value=1, visible=False)
            generate_btn.click(song_cover_pipeline,
                               inputs=[song_input, rvc_model, pitch, keep_files, is_webui, main_gain, backup_gain,
                                       inst_gain, index_rate, reverb_rm_size, reverb_wet, reverb_dry, reverb_damping],
                               outputs=[ai_cover])
            clear_btn.click(lambda: [0, 0, 0, 0, 0.5, 0.15, 0.2, 0.8, 0.7, None],
                            outputs=[pitch, main_gain, backup_gain, inst_gain, index_rate, reverb_rm_size, reverb_wet,
                                     reverb_dry, reverb_damping, ai_cover])

        # Download tab
        with gr.Tab("Download model"):

            with gr.Tab('From HuggingFace/Pixeldrain URL'):
                with gr.Row():
                    model_zip_link = gr.Text(label='Download link to model', info='Should be a zip file containing a .pth model file and an optional .index file.')
                    model_name = gr.Text(label='Name your model', info='Give your new model a unique name from your other voice models.')

                with gr.Row():
                    download_btn = gr.Button('Download 🌐', variant='primary', scale=19)
                    dl_output_message = gr.Text(label='Output Message', interactive=False, scale=20)

                download_btn.click(download_and_extract_zip, inputs=[model_zip_link, model_name], outputs=dl_output_message)

                gr.Markdown('## Input Examples')
                gr.Examples(
                    [
                        ['https://huggingface.co/phant0m4r/LiSA/resolve/main/LiSA.zip', 'Lisa'],
                        ['https://pixeldrain.com/u/3tJmABXA', 'Gura'],
                        ['https://huggingface.co/Kit-Lemonfoot/kitlemonfoot_rvc_models/resolve/main/AZKi%20(Hybrid).zip', 'Azki']
                    ],
                    [model_zip_link, model_name],
                    [],
                    download_and_extract_zip,
                )

            with gr.Tab('From Public Index'):

                gr.Markdown('## How to use')
                gr.Markdown('- Click Initialize public models table')
                gr.Markdown('- Filter models using tags or search bar')
                gr.Markdown('- Select a row to autofill the download link and model name')
                gr.Markdown('- Click Download')

                with gr.Row():
                    pub_zip_link = gr.Text(label='Download link to model')
                    pub_model_name = gr.Text(label='Model name')

                with gr.Row():
                    download_pub_btn = gr.Button('Download 🌐', variant='primary', scale=19)
                    pub_dl_output_message = gr.Text(label='Output Message', interactive=False, scale=20)

                filter_tags = gr.CheckboxGroup(value=[], label='Show voice models with tags', choices=[])
                search_query = gr.Text(label='Search')
                load_public_models_button = gr.Button(value='Initialize public models table', variant='primary')

                public_models_table = gr.DataFrame(value=[], headers=['Model Name', 'Description', 'Credit', 'URL', 'Tags'], label='Available Public Models', interactive=False)
                public_models_table.select(pub_dl_autofill, inputs=[public_models_table], outputs=[pub_zip_link, pub_model_name])
                load_public_models_button.click(load_public_models, outputs=[public_models_table, filter_tags])
                search_query.change(filter_models, inputs=[filter_tags, search_query], outputs=public_models_table)
                filter_tags.change(filter_models, inputs=[filter_tags, search_query], outputs=public_models_table)
                download_pub_btn.click(download_and_extract_zip, inputs=[pub_zip_link, pub_model_name], outputs=pub_dl_output_message)

    app.launch(share=share_enabled, enable_queue=True)
