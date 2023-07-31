import gradio as gr
from urllib.parse import urlparse
import urllib.request
import zipfile,os,json
from main import preprocess_song,get_audio_paths,voice_change,combine_audio,add_audio_effects
from argparse import ArgumentParser

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

mdxnet_models_dir = os.path.join(BASE_DIR, 'mdxnet_models')
rvc_models_dir = os.path.join(BASE_DIR, 'rvc_models')
output_dir = os.path.join(BASE_DIR, 'song_output')



def get_youtube_video_id(value):
    """
    Examples:
    - http://youtu.be/SA2iWivDJiE
    - http://www.youtube.com/watch?v=_oPAwA_Udwc&feature=feedu
    - http://www.youtube.com/embed/SA2iWivDJiE
    - http://www.youtube.com/v/SA2iWivDJiE?version=3&amp;hl=en_US
    """
    query = urlparse(value)
    if query.hostname == 'youtu.be':
        return query.path[1:]
    if query.hostname in ('www.youtube.com', 'youtube.com'):
        if query.path == '/watch':
            p = urlparse.parse_qs(query.query)
            return p['v'][0]
        if query.path[:7] == '/embed/':
            return query.path.split('/')[2]
        if query.path[:3] == '/v/':
            return query.path.split('/')[2]
    
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

def download_and_extract_zip(url, folder_name):
    print(f'[~] Download voice model with name {folder_name}')
    # getting name  
    file_name = url.split('/')[-1]
    # download zip
    urllib.request.urlretrieve(url, file_name)
    extraction_folder = os.path.join(rvc_models_dir, folder_name)
    os.makedirs(extraction_folder, exist_ok=True)
    print(f'[~] Start extracting zip')
    # unzip
    with zipfile.ZipFile(file_name, 'r') as zip_ref:
        zip_ref.extractall(extraction_folder)
    # remove downloaded zip
    os.remove(file_name)
    obj =update_models_list()
    print(f'[+] Model {folder_name} was downloaded!')
    return obj


if __name__ == '__main__':
    parser = ArgumentParser(description='Generate a AI cover song in the song_output/id directory.', add_help=True)
    parser.add_argument("--share", action="store_true", dest="share_enabled", default=False, help="Enable sharing")
    args = parser.parse_args()
    share_enabled = args.share_enabled

    
    # new web ui
    voice_models = get_models_list(rvc_models_dir)
    with gr.Blocks(title='WebAICoverGen') as app:
        
        gr.Label('Created with love ❤️',show_label=False)
       
        #main tab
        with gr.Tab("Generate"):
            
            with gr.Row():
                with gr.Column():
                    yt_link = gr.Text(label='YouTabe link')
                    with gr.Row():
                        voice_model = gr.Dropdown(voice_models,label='Voice Models',info='Models folder "AICoverGen --> rvc_models", after you paste models, press update button')
                        ref_btn = gr.Button("Update 🔁",variant='primary',size='sm',min_width=50)
                    pitch = gr.Slider(-12,12,value=0,step=1,label='Pitch')
                    with gr.Row():
                        clear_btn = gr.ClearButton(value='Clear',components=[yt_link,voice_model,pitch])
                        generate_btn = gr.Button("Generate",variant='primary')
                with gr.Column():
                    audio = gr.Audio(label='Audio',show_share_button=False)
                ref_btn.click(update_models_list,None,outputs=voice_model)
                generate_btn.click(song_cover_pipeline, inputs=[yt_link,voice_model,pitch], outputs=[audio])
        
        #Download tab       
        with gr.Tab("Download model"):     
            model_zip_link = gr.Text(label='Link for your model', info='Example link: https://example.com/voice_model.zip')
            model_name = gr.Text(label='Name of your model')
            download_btn = gr.Button("Download 🌐",variant='primary')
            
            download_btn.click(download_and_extract_zip,inputs=[model_zip_link,model_name],outputs=voice_model)
            

    

    app.launch(share=share_enabled)

