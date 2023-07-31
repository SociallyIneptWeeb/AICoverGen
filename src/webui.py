import json
import os
import urllib.request
import zipfile
from argparse import ArgumentParser
from urllib.parse import urlparse, parse_qs
from contextlib import suppress

import gradio as gr

from main import preprocess_song, get_audio_paths, voice_change, combine_audio, add_audio_effects

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

mdxnet_models_dir = os.path.join(BASE_DIR, 'mdxnet_models')
rvc_models_dir = os.path.join(BASE_DIR, 'rvc_models')
output_dir = os.path.join(BASE_DIR, 'song_output')


def get_youtube_video_id(url, ignore_playlist=True):
    """
    Examples:
    http://youtu.be/SA2iWivDJiE
    http://www.youtube.com/watch?v=_oPAwA_Udwc&feature=feedu
    http://www.youtube.com/embed/SA2iWivDJiE
    http://www.youtube.com/v/SA2iWivDJiE?version=3&amp;hl=en_US
    """
    query = urlparse(url)
    if query.hostname == 'youtu.be':
        if query.path[1:] == 'watch':
            return query.query[2:]
        return query.path[1:]

    if query.hostname in {'www.youtube.com', 'youtube.com', 'music.youtube.com'}:
        if not ignore_playlist:
            # use case: get playlist id not current video in playlist
            with suppress(KeyError):
                return parse_qs(query.query)['list'][0]
        if query.path == '/watch': 
            return parse_qs(query.query)['v'][0]
        if query.path[:7] == '/watch/': 
            return query.path.split('/')[1]
        if query.path[:7] == '/embed/':
            return query.path.split('/')[2]
        if query.path[:3] == '/v/':
            return query.path.split('/')[2]

    # returns None for invalid YouTube url
    return None


def song_cover_pipeline(yt_link, voice_model, pitch_change):
    with open(os.path.join(mdxnet_models_dir, 'model_data.json')) as infile:
        mdx_model_params = json.load(infile)

    song_id = get_youtube_video_id(yt_link)
    song_dir = os.path.join(output_dir, song_id)

    if not os.path.exists(song_dir):
        os.makedirs(song_dir)
        orig_song_path, vocals_path, instrumentals_path, main_vocals_path, backup_vocals_path, main_vocals_dereverb_path = preprocess_song(yt_link, mdx_model_params, song_id)

    else:
        vocals_path, main_vocals_path = None, None
        paths = get_audio_paths(song_dir)

        # if any of the audio files aren't available, rerun preprocess
        if any(path is None for path in paths):
            orig_song_path, vocals_path, instrumentals_path, main_vocals_path, backup_vocals_path, main_vocals_dereverb_path = preprocess_song(yt_link, mdx_model_params, song_id)
        else:
            orig_song_path, instrumentals_path, main_vocals_dereverb_path, backup_vocals_path = paths

    ai_vocals_path, ai_vocals_mixed_path = None, None
    ai_cover_path = os.path.join(song_dir, f'{os.path.splitext(orig_song_path)[0]} ({voice_model} Ver).wav')

    if not os.path.exists(ai_cover_path):
        print('Converting voice using RVC...')
        ai_vocals_path = voice_change(voice_model, main_vocals_dereverb_path, pitch_change)

        print('Applying audio effects to vocals...')
        ai_vocals_mixed_path = add_audio_effects(ai_vocals_path)

        print('Combining AI Vocals and Instrumentals...')
        combine_audio([ai_vocals_mixed_path, backup_vocals_path, instrumentals_path], ai_cover_path)

    print('Removing intermediate audio files...')
    intermediate_files = [vocals_path, main_vocals_path, ai_vocals_path, ai_vocals_mixed_path]
    for file in intermediate_files:
        if file and os.path.exists(file):
            os.remove(file)

    return ai_cover_path


def get_models_list(models_dir):
    models_list = os.listdir(models_dir)
    items_to_remove = ['hubert_base.pt', 'MODELS.txt']
    models_clear_list = [item for item in models_list if item not in items_to_remove]
    return models_clear_list


def update_models_list():
    models_l = get_models_list(rvc_models_dir)
    return gr.Dropdown.update(choices=models_l)


def download_and_extract_zip(url, dir_name, progress=gr.Progress()):
    progress(0, desc=f'[~] Downloading voice model with name {dir_name}...')
    zip_name = url.split('/')[-1]
    extraction_folder = os.path.join(rvc_models_dir, dir_name)
    if os.path.exists(extraction_folder):
        raise gr.Error(f'Voice model directory {dir_name} already exists! Choose a different name for your voice model.')

    os.makedirs(extraction_folder)
    urllib.request.urlretrieve(url, zip_name)

    progress(0.5, desc='[~] Extracting zip...')

    with zipfile.ZipFile(zip_name, 'r') as zip_ref:
        zip_ref.extractall(extraction_folder)
    os.remove(zip_name)
    update_models_list()

    return f'[+] {dir_name} Model successfully downloaded!'


if __name__ == '__main__':
    parser = ArgumentParser(description='Generate a AI cover song in the song_output/id directory.', add_help=True)
    parser.add_argument("--share", action="store_true", dest="share_enabled", default=False, help="Enable sharing")
    args = parser.parse_args()
    share_enabled = args.share_enabled

    # new web ui
    voice_models = get_models_list(rvc_models_dir)
    with gr.Blocks(title='AICoverGenWebUI') as app:
        
        gr.Label('AICoverGen WebUI created with ❤️', show_label=False)
       
        # main tab
        with gr.Tab("Generate"):
            with gr.Row():
                with gr.Column():

                    with gr.Row():
                        rvc_model = gr.Dropdown(voice_models, label='Voice Models', scale=10, info='Models folder "AICoverGen --> rvc_models". After the models are added into this folder, click the update button')
                        ref_btn = gr.Button('Update 🔁', variant='primary', scale=9)

                    with gr.Row():
                        video_link = gr.Text(label='YouTube link')
                        pitch = gr.Slider(-12, 12, value=0, step=1, label='Pitch')

                    with gr.Row():
                        clear_btn = gr.ClearButton(value='Clear', components=[video_link, rvc_model, pitch])
                        generate_btn = gr.Button("Generate", variant='primary')

                audio = gr.Audio(label='Audio', show_share_button=False)
                ref_btn.click(update_models_list, None, outputs=rvc_model)
                generate_btn.click(song_cover_pipeline, inputs=[video_link, rvc_model, pitch], outputs=[audio])
        
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
                    ["https://huggingface.co/daibi1998/RVC_v2_Punishing_Gray_Raven/resolve/main/KareninaEN_RVC_v2_RMVPE_250e.zip", "Karenina"],
                    ["https://huggingface.co/RinkaEmina/RVC_Sharing/resolve/main/Emilia%20V2%2048000.zip", "Emilia"]
                ],
                [model_zip_link, model_name],
                [],
                download_and_extract_zip,
            )

    app.launch(share=share_enabled, enable_queue=True)
