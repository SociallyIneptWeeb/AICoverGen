import gc
import json
import os
import random
import re
from argparse import ArgumentParser
import string

import yt_dlp
from pedalboard import Pedalboard, Reverb, Compressor, HighpassFilter
from pedalboard.io import AudioFile
from pydub import AudioSegment
import ntpath
import subprocess

from mdx import run_mdx
from rvc import Config, load_hubert, get_vc, rvc_infer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

mdxnet_models_dir = os.path.join(BASE_DIR, 'mdxnet_models')
rvc_models_dir = os.path.join(BASE_DIR, 'rvc_models')
output_dir = os.path.join(BASE_DIR, 'song_output')


def download_audio(link):
    ydl_opts = {
        'format': 'bestaudio',
        'outtmpl': '%(title)s.%(ext)s',
        'nocheckcertificate': True,
        'ignoreerrors': True,
        'no_warnings': True,
        'quiet': True,
        'extractaudio': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        result = ydl.extract_info(link, download=True)
        download_path = ydl.prepare_filename(result)

    return download_path


def get_rvc_models(voice_model):
    rvc_model_filename, rvc_index_filename = None, None
    model_dir = os.path.join(rvc_models_dir, voice_model)
    for file in os.listdir(model_dir):
        ext = os.path.splitext(file)[1]
        if ext == '.pth':
            rvc_model_filename = file
        if ext == '.index':
            rvc_index_filename = file

    if rvc_model_filename is None:
        raise Exception(f'No model file exists in {model_dir}.')

    return os.path.join(model_dir, rvc_model_filename), os.path.join(model_dir, rvc_index_filename) if rvc_index_filename else ''


def get_audio_paths(song_dir):
    orig_song_path = None
    instrumentals_path = None
    main_vocals_dereverb_path = None
    backup_vocals_path = None

    for file in os.listdir(song_dir):
        if file.endswith('_Instrumental.wav'):
            instrumentals_path = os.path.join(song_dir, file)
            orig_song_path = instrumentals_path.replace('_Instrumental', '')

        elif file.endswith('_Vocals_Main_DeReverb.wav'):
            main_vocals_dereverb_path = os.path.join(song_dir, file)

        elif file.endswith('_Vocals_Backup.wav'):
            backup_vocals_path = os.path.join(song_dir, file)

    return orig_song_path, instrumentals_path, main_vocals_dereverb_path, backup_vocals_path


def display_progress(message, percent, progress=None):
    if progress is not None:
        progress(percent, desc=message)
    else:
        print(message)


def preprocess_song(yt_link, mdx_model_params, song_id, progress=None):
    display_progress('[~] Downloading song...', 0, progress)
    orig_song_path = download_audio(yt_link)
    song_output_dir = os.path.join(output_dir, song_id)
    print(str(song_output_dir))

    display_progress('[~] Separating Vocals from Instrumental...', 0.1, progress)
    vocals_path, instrumentals_path = run_mdx(mdx_model_params, song_output_dir, os.path.join(mdxnet_models_dir, 'UVR-MDX-NET-Voc_FT.onnx'), orig_song_path, denoise=True, keep_orig=False)

    display_progress('[~] Separating Main Vocals from Backup Vocals...', 0.2, progress)
    backup_vocals_path, main_vocals_path = run_mdx(mdx_model_params, song_output_dir, os.path.join(mdxnet_models_dir, 'UVR_MDXNET_KARA_2.onnx'), vocals_path, suffix='Backup', invert_suffix='Main', denoise=True)

    display_progress('[~] Applying DeReverb to Vocals...', 0.3, progress)
    _, main_vocals_dereverb_path = run_mdx(mdx_model_params, song_output_dir, os.path.join(mdxnet_models_dir, 'Reverb_HQ_By_FoxJoy.onnx'), main_vocals_path, invert_suffix='DeReverb', exclude_main=True, denoise=True)

    return orig_song_path, vocals_path, instrumentals_path, main_vocals_path, backup_vocals_path, main_vocals_dereverb_path


def generate_random_name(length=8):
    characters = string.ascii_letters + string.digits
    random_name = ''.join(random.choice(characters) for _ in range(length))
    return random_name

def convert_to_aac(input_file, output_file):
    cmd = ["ffmpeg", "-i", input_file, "-c:a", "aac","-b:a","192k", output_file]
    print(cmd)
    subprocess.run(cmd)

def is_supported_format(file_path):
    supported_formats = ['.wav', '.mp3']
    _, file_extension = os.path.splitext(file_path)
    return file_extension in supported_formats

def clean_file_name(file_path):
    base_name = ntpath.basename(file_path)
    file_name, file_extension = ntpath.splitext(base_name)
    
    # Remove special characters using regex
    clean_name = re.sub(r'[^\w\s-]', '', file_name)
    
    # Replace spaces with underscores
    clean_name = clean_name.replace(' ', '_')
    
    return clean_name

def preprocess_song_local(input_audio, mdx_model_params,song_id, progress=None):
    song_output_dir = os.path.join(output_dir, song_id)
    
    
    filename = clean_file_name(input_audio)
    song_output_file = os.path.join(song_output_dir, filename)
 
    print(input_audio)
    print(song_output_file)

    song_output_file = str(song_output_file)+".m4a"
    orig_song_path = input_audio

    if is_supported_format(input_audio):
        convert_to_aac(input_audio,song_output_file)
        print(f"Audio converted to .m4a format: {filename}")
        orig_song_path = song_output_file

    
    print(orig_song_path)
    song_output_dir = os.path.join(output_dir, song_id)

    display_progress('[~] Separating Vocals from Instrumental...', 0.1, progress)
    vocals_path, instrumentals_path = run_mdx(mdx_model_params, song_output_dir, os.path.join(mdxnet_models_dir, 'UVR-MDX-NET-Voc_FT.onnx'), orig_song_path, denoise=True, keep_orig=False)

    display_progress('[~] Separating Main Vocals from Backup Vocals...', 0.2, progress)
    backup_vocals_path, main_vocals_path = run_mdx(mdx_model_params, song_output_dir, os.path.join(mdxnet_models_dir, 'UVR_MDXNET_KARA_2.onnx'), vocals_path, suffix='Backup', invert_suffix='Main', denoise=True)

    display_progress('[~] Applying DeReverb to Vocals...', 0.3, progress)
    _, main_vocals_dereverb_path = run_mdx(mdx_model_params, song_output_dir, os.path.join(mdxnet_models_dir, 'Reverb_HQ_By_FoxJoy.onnx'), main_vocals_path, invert_suffix='DeReverb', exclude_main=True, denoise=True)

    return orig_song_path, vocals_path, instrumentals_path, main_vocals_path, backup_vocals_path, main_vocals_dereverb_path



def voice_change(voice_model, vocals_path, pitch_change):
    # determine pitch to change if none provided
    rvc_model_path, rvc_index_path = get_rvc_models(voice_model)
    device = 'cuda:0'
    config = Config(device, True)
    hubert_model = load_hubert(device, config.is_half, os.path.join(rvc_models_dir, 'hubert_base.pt'))
    cpt, version, net_g, tgt_sr, vc = get_vc(device, config.is_half, config, rvc_model_path)

    output_path = f'{os.path.splitext(vocals_path)[0]}_{voice_model}.wav'
    # convert main vocals
    rvc_infer(rvc_index_path, vocals_path, output_path, pitch_change, cpt, version, net_g, tgt_sr, vc, hubert_model)
    del hubert_model, cpt
    gc.collect()
    return output_path


def add_audio_effects(audio_path):
    output_path = f'{os.path.splitext(audio_path)[0]}_mixed.wav'

    # Initialize audio effects plugins
    board = Pedalboard([HighpassFilter(), Compressor(ratio=4, threshold_db=-15), Reverb(room_size=0.15, dry_level=0.8, wet_level=0.2, damping=0.7)])

    with AudioFile(audio_path) as f:
        with AudioFile(output_path, 'w', f.samplerate, f.num_channels) as o:
            # Read one second of audio at a time, until the file is empty:
            while f.tell() < f.frames:
                chunk = f.read(int(f.samplerate))
                effected = board(chunk, f.samplerate, reset=False)
                o.write(effected)

    return output_path


def combine_audio(audio_paths, output_path):
    main_vocal_audio = AudioSegment.from_wav(audio_paths[0]) - 4
    backup_vocal_audio = AudioSegment.from_wav(audio_paths[1]) - 6
    instrumental_audio = AudioSegment.from_wav(audio_paths[2]) - 7
    main_vocal_audio.overlay(backup_vocal_audio).overlay(instrumental_audio).export(output_path, format='mp3')


def song_cover_pipeline(yt_link, voice_model, pitch_change):
    with open(os.path.join(mdxnet_models_dir, 'model_data.json')) as infile:
        mdx_model_params = json.load(infile)

    if '&' in yt_link:
        yt_link = yt_link.split('&')[0]

    match = re.search(r"v=([^&]+)", yt_link)
    song_id = match.group(1)
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
    ai_cover_path = os.path.join(song_dir, f'{os.path.splitext(orig_song_path)[0]} ({voice_model} Ver {pitch_change}).mp3')

    if not os.path.exists(ai_cover_path):
        print('[~] Converting voice using RVC...')
        ai_vocals_path = voice_change(voice_model, main_vocals_dereverb_path, pitch_change)

        print('[~] Applying audio effects to vocals...')
        ai_vocals_mixed_path = add_audio_effects(ai_vocals_path)

        print('[~] Combining AI Vocals and Instrumentals...')
        combine_audio([ai_vocals_mixed_path, backup_vocals_path, instrumentals_path], ai_cover_path)

    print('[~] Removing intermediate audio files...')
    intermediate_files = [vocals_path, main_vocals_path, ai_vocals_path, ai_vocals_mixed_path]
    for file in intermediate_files:
        if file and os.path.exists(file):
            os.remove(file)

    return ai_cover_path


if __name__ == '__main__':
    parser = ArgumentParser(description='Generate a AI cover song in the song_output/id directory.', add_help=True)
    parser.add_argument('-yt', '--youtube-link', type=str, required=True, help='Link to a youtube video to create an AI cover of')
    parser.add_argument('-dir', '--rvc-dirname', type=str, required=True, help='Name of the folder in the rvc_models directory containing the RVC model file and optional index file to use')
    parser.add_argument('-p', '--pitch-change', type=int, required=True, help='Change the pitch of the AI voice. Generally use 12 for male to female conversions and -12 for vice-versa. Use 0 for no change')
    args = parser.parse_args()

    rvc_dirname = args.rvc_dirname
    if not os.path.exists(os.path.join(rvc_models_dir, rvc_dirname)):
        raise Exception(f'The folder {os.path.join(rvc_models_dir, rvc_dirname)} does not exist.')

    cover_path = song_cover_pipeline(args.youtube_link, rvc_dirname, args.pitch_change)
    print(f'[+] Cover generated at {cover_path}')
