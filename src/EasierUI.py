import json
import os
import shutil
import urllib.request
import zipfile
import gdown
from argparse import ArgumentParser

import gradio as gr

from main import song_cover_pipeline

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

mdxnet_models_dir = os.path.join(BASE_DIR, 'mdxnet_models')
rvc_models_dir = os.path.join(BASE_DIR, 'rvc_models')
output_dir = os.path.join(BASE_DIR, 'song_output')


def get_current_models(models_dir):
    models_list = os.listdir(models_dir)
    items_to_remove = ['hubert_base.pt', 'MODELS.txt', 'public_models.json', 'rmvpe.pt', 'fcpe.pt']
    return [item for item in models_list if item not in items_to_remove]


def update_models_list():
    models_l = get_current_models(rvc_models_dir)
    return gr.Dropdown(choices=models_l)


def load_public_models():
    models_table = []
    for model in public_models['voice_models']:
        if not model['name'] in voice_models:
            model = [model['name'], model['description'], model['credit'], model['url'], ', '.join(model['tags'])]
            models_table.append(model)

    tags = list(public_models['tags'].keys())
    return gr.DataFrame.update(value=models_table), gr.CheckboxGroup.update(choices=tags)


def extract_zip(extraction_folder, zip_name):
    os.makedirs(extraction_folder)
    with zipfile.ZipFile(zip_name, 'r') as zip_ref:
        zip_ref.extractall(extraction_folder)
    os.remove(zip_name)

    index_filepath, model_filepath = None, None
    for root, dirs, files in os.walk(extraction_folder):
        for name in files:
            if name.endswith('.index') and os.stat(os.path.join(root, name)).st_size > 1024 * 100:
                index_filepath = os.path.join(root, name)

            if name.endswith('.pth') and os.stat(os.path.join(root, name)).st_size > 1024 * 1024 * 40:
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


def download_online_model(url, dir_name, progress=gr.Progress()):
    try:
        progress(0, desc=f'[~] Downloading voice model with name {dir_name}...')
        zip_name = url.split('/')[-1]
        extraction_folder = os.path.join(rvc_models_dir, dir_name)
        if os.path.exists(extraction_folder):
            raise gr.Error(f'Voice model directory {dir_name} already exists! Choose a different name for your voice model.')

        if 'huggingface.co' in url:
            urllib.request.urlretrieve(url, zip_name)

        if 'pixeldrain.com' in url:
            zip_name = dir_name + '.zip'
            url = f'https://pixeldrain.com/api/file/{zip_name}'
            urllib.request.urlretrieve(url, zip_name)

        elif 'drive.google.com' in url:
            # Extract the Google Drive file ID
            zip_name = dir_name + '.zip'
            file_id = url.split('/')[-2]
            output = os.path.join('.', f'{dir_name}.zip')  # Adjust the output path if needed
            gdown.download(id=file_id, output=output, quiet=False)

        progress(0.5, desc='[~] Extracting zip...')
        extract_zip(extraction_folder, zip_name)
        return f'[+] {dir_name} Model successfully downloaded!'

    except Exception as e:
        raise gr.Error(str(e))


def upload_local_model(zip_path, dir_name, progress=gr.Progress()):
    try:
        extraction_folder = os.path.join(rvc_models_dir, dir_name)
        if os.path.exists(extraction_folder):
            raise gr.Error(f'Voice model directory {dir_name} already exists! Choose a different name for your voice model.')

        zip_name = zip_path.name
        progress(0.5, desc='[~] Extracting zip...')
        extract_zip(extraction_folder, zip_name)
        return f'[+] {dir_name} Model successfully uploaded!'

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


def swap_visibility():
    return gr.update(visible=True), gr.update(visible=False), gr.update(value=''), gr.update(value=None)


def process_file_upload(file):
    return file.name, gr.update(value=file.name)


def show_hop_slider(pitch_detection_algo):
    if pitch_detection_algo == 'mangio-crepe':
        return gr.update(visible=True)
    else:
        return gr.update(visible=False)

def clear_components():
    return "", "", False  # Replace with default values of your components




if __name__ == '__main__':
    parser = ArgumentParser(description='EasyGUI for AICoverGen.', add_help=True)
    parser.add_argument("--share", action="store_true", dest="share_enabled", default=False, help="Enable sharing")
    parser.add_argument("--listen", action="store_true", default=False, help="Make the WebUI reachable from your local network.")
    parser.add_argument('--listen-host', type=str, help='The hostname that the server will use.')
    parser.add_argument('--listen-port', type=int, help='The listening port that the server will use.')
    args = parser.parse_args()

    voice_models = get_current_models(rvc_models_dir)
    with open(os.path.join(rvc_models_dir, 'public_models.json'), encoding='utf8') as infile:
        public_models = json.load(infile)
with gr.Blocks(theme=gr.themes.Base(), title=" Easy GUI v2 (rejekts) - adapted to AICoverGen 💻") as app:
    with gr.Tabs():
        with gr.TabItem("Inference"):
            gr.HTML("<h1> Easy GUI v2 (rejekts) - adapted to AICoverGen 💻 </h1>")
         
            # Other RVC stuff
            with gr.Row():
                rvc_model = gr.Dropdown(voice_models,label='Voice models')
                ref_btn = gr.Button('Refresh Models 🔁', variant='primary')
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        pitch = gr.Slider(-12, 12, value=0, step=1, label='Pitch Change (Vocals ONLY)')
                        pitch_all = gr.Slider(-12, 12, value=0, step=1, label='Overall Pitch Change')
            with gr.Row():
                generate_btn = gr.Button("Generate", variant='primary')
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        song_input = gr.Text(label='Song input', info='Link to a song on YouTube or full path to a local file.')
            with gr.Row():
                with gr.Column():
                    with gr.Accordion("Index Settings", open=False):
                        index_rate = gr.Slider(0, 1, value=0.5, label='Index Rate', info="Controls how much of the AI voice's accent to keep in the vocals")
                    with gr.Accordion("More Settings", open=False):
                        f0_method = gr.Dropdown(['rmvpe', 'mangio-crepe'], value='rmvpe', label='Pitch detection algorithm', info='Best option is rmvpe (clarity in vocals), then mangio-crepe (smoother vocals)')
                        crepe_hop_length = gr.Slider(32, 320, value=128, step=1, visible=False, label='Crepe hop length', info='Lower values leads to longer conversions and higher risk of voice cracks, but better pitch accuracy.')
                        f0_method.change(show_hop_slider, inputs=f0_method, outputs=crepe_hop_length)
                        keep_files = gr.Checkbox(label='Keep intermediate files', info='Keep all audio files generated in the song_output/id directory, e.g. Isolated Vocals/Instrumentals. Leave unchecked to save space')
                        filter_radius = gr.Slider(0, 7, value=3, step=1, label='Filter radius', info='If >=3: apply median filtering median filtering to the harvested pitch results. Can reduce breathiness')
                        rms_mix_rate = gr.Slider(0, 1, value=0.25, label='RMS mix rate', info="Control how much to mimic the original vocal's loudness (0) or a fixed loudness (1)")
                        protect = gr.Slider(0, 0.5, value=0.33, label='Protect rate', info='Protect voiceless consonants and breath sounds. Set to 0.5 to disable.')
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
                        gr.Markdown('### Audio Output Format')
                        output_format = gr.Dropdown(['mp3', 'wav'], value='mp3', label='Output file type', info='mp3: small file size, decent quality. wav: Large file size, best quality')

                    ai_cover = gr.Audio(label='AI Cover | Output Audio (Click on the Three Dots in the Right Corner to Download)', show_share_button=False)
                   
                    clear_btn = gr.ClearButton(value='Clear', components=[song_input, rvc_model, keep_files])
                
                ref_btn.click(update_models_list, None, outputs=rvc_model)
                is_webui = gr.Number(value=1, visible=False)
                generate_btn.click(song_cover_pipeline,
                                   inputs=[song_input, rvc_model, pitch, keep_files, is_webui, main_gain, backup_gain,
                                           inst_gain, index_rate, filter_radius, rms_mix_rate, f0_method, crepe_hop_length,
                                           protect, pitch_all, reverb_rm_size, reverb_wet, reverb_dry, reverb_damping,
                                           output_format],
                                   outputs=[ai_cover])
                clear_btn.click(lambda: [0, 0, 0, 0, 0.5, 3, 0.25, 0.33, 'rmvpe', 128, 0, 0.15, 0.2, 0.8, 0.7, 'mp3', None],
                                outputs=[pitch, main_gain, backup_gain, inst_gain, index_rate, filter_radius, rms_mix_rate,
                                         protect, f0_method, crepe_hop_length, pitch_all, reverb_rm_size, reverb_wet,
                                         reverb_dry, reverb_damping, output_format, ai_cover])
        with gr.TabItem("Download Model"):
            with gr.Row():
                url=gr.Textbox(label="Enter the URL to the Model:")
            with gr.Row():
                model = gr.Textbox(label="Name your model:")
                download_button=gr.Button("Download")
            with gr.Row():
                status_bar=gr.Textbox(label="")
                download_button.click(download_online_model, inputs=[url, model], outputs=status_bar)
            with gr.Row():
                gr.Markdown(
                """
                AICoverGen: https://github.com/SociallyIneptWeeb/AICoverGen\n
                Made by TheNeodev
                """
                )

                
    app.queue().launch(
        share=args.share_enabled,
        server_name=None if not args.listen else (args.listen_host or '0.0.0.0'),
        server_port=args.listen_port,
    )
