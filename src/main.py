import gc
import hashlib
import json
import os
import argparse
from contextlib import suppress
from urllib.parse import urlparse, parse_qs

import gradio as gr
import yt_dlp
from pedalboard import Pedalboard, Reverb, Compressor, HighpassFilter
from pedalboard.io import AudioFile
from pydub import AudioSegment

from mdx import run_mdx
from rvc import Config, load_hubert, get_vc, rvc_infer

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


def yt_download(link):
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


def raise_exception(error_msg, is_webui):
    if is_webui:
        raise gr.Error(error_msg)
    else:
        raise Exception(error_msg)


def get_rvc_model(voice_model, is_webui):
    rvc_model_filename, rvc_index_filename = None, None
    model_dir = os.path.join(rvc_models_dir, voice_model)
    for file in os.listdir(model_dir):
        ext = os.path.splitext(file)[1]
        if ext == '.pth':
            rvc_model_filename = file
        if ext == '.index':
            rvc_index_filename = file

    if rvc_model_filename is None:
        error_msg = f'No model file exists in {model_dir}.'
        raise_exception(error_msg, is_webui)

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


def get_hash(filepath):
    with open(filepath, 'rb') as f:
        file_hash = hashlib.blake2b()
        while chunk := f.read(8192):
            file_hash.update(chunk)

    return file_hash.hexdigest()[:11]


def display_progress(message, percent, is_webui, progress=None):
    if is_webui:
        progress(percent, desc=message)
    else:
        print(message)


def preprocess_song(song_input, mdx_model_params, song_id, is_webui, input_type, progress=None):
    keep_orig = False
    if input_type == 'yt':
        display_progress('[~] Downloading song...', 0, is_webui, progress)
        orig_song_path = yt_download(song_input)
    elif input_type == 'local':
        orig_song_path = song_input
        keep_orig = True
    else:
        orig_song_path = None

    song_output_dir = os.path.join(output_dir, song_id)

    display_progress('[~] Separating Vocals from Instrumental...', 0.1, is_webui, progress)
    vocals_path, instrumentals_path = run_mdx(mdx_model_params, song_output_dir, os.path.join(mdxnet_models_dir, 'UVR-MDX-NET-Voc_FT.onnx'), orig_song_path, denoise=True, keep_orig=keep_orig)

    display_progress('[~] Separating Main Vocals from Backup Vocals...', 0.2, is_webui, progress)
    backup_vocals_path, main_vocals_path = run_mdx(mdx_model_params, song_output_dir, os.path.join(mdxnet_models_dir, 'UVR_MDXNET_KARA_2.onnx'), vocals_path, suffix='Backup', invert_suffix='Main', denoise=True)

    display_progress('[~] Applying DeReverb to Vocals...', 0.3, is_webui, progress)
    _, main_vocals_dereverb_path = run_mdx(mdx_model_params, song_output_dir, os.path.join(mdxnet_models_dir, 'Reverb_HQ_By_FoxJoy.onnx'), main_vocals_path, invert_suffix='DeReverb', exclude_main=True, denoise=True)

    return orig_song_path, vocals_path, instrumentals_path, main_vocals_path, backup_vocals_path, main_vocals_dereverb_path


def voice_change(voice_model, vocals_path, pitch_change, is_webui):
    rvc_model_path, rvc_index_path = get_rvc_model(voice_model, is_webui)
    device = 'cuda:0'
    config = Config(device, True)
    hubert_model = load_hubert(device, config.is_half, os.path.join(rvc_models_dir, 'hubert_base.pt'))
    cpt, version, net_g, tgt_sr, vc = get_vc(device, config.is_half, config, rvc_model_path)

    # convert main vocals
    output_path = f'{os.path.splitext(vocals_path)[0]}_{voice_model}.mp3'
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


def song_cover_pipeline(song_input, voice_model, pitch_change, keep_files, is_webui=0, progress=gr.Progress()):
    try:
        if not song_input or not voice_model:
            raise_exception('Ensure that the song input field and voice model field is filled.', is_webui)

        display_progress('[~] Starting AI Cover Generation Pipeline...', 0, is_webui, progress)

        with open(os.path.join(mdxnet_models_dir, 'model_data.json')) as infile:
            mdx_model_params = json.load(infile)

        # if youtube url
        if urlparse(song_input).scheme == 'https':
            input_type = 'yt'
            song_id = get_youtube_video_id(song_input)
            if song_id is None:
                error_msg = 'Invalid YouTube url.'
                raise_exception(error_msg, is_webui)

        # local audio file
        else:
            input_type = 'local'
            song_input = song_input.strip('\"')
            if os.path.exists(song_input):
                song_id = get_hash(song_input)
            else:
                error_msg = f'{song_input} does not exist.'
                song_id = None
                raise_exception(error_msg, is_webui)

        song_dir = os.path.join(output_dir, song_id)

        if not os.path.exists(song_dir):
            os.makedirs(song_dir)
            orig_song_path, vocals_path, instrumentals_path, main_vocals_path, backup_vocals_path, main_vocals_dereverb_path = preprocess_song(song_input, mdx_model_params, song_id, is_webui, input_type, progress)

        else:
            vocals_path, main_vocals_path = None, None
            paths = get_audio_paths(song_dir)

            # if any of the audio files aren't available or keep intermediate files, rerun preprocess
            if any(path is None for path in paths) or keep_files:
                orig_song_path, vocals_path, instrumentals_path, main_vocals_path, backup_vocals_path, main_vocals_dereverb_path = preprocess_song(song_input, mdx_model_params, song_id, is_webui, input_type, progress)
            else:
                orig_song_path, instrumentals_path, main_vocals_dereverb_path, backup_vocals_path = paths

        ai_vocals_path, ai_vocals_mixed_path = None, None
        ai_cover_path = os.path.join(song_dir, f'{os.path.splitext(os.path.basename(orig_song_path))[0]} ({voice_model} Ver {pitch_change}).mp3')

        if not os.path.exists(ai_cover_path):
            display_progress('[~] Converting voice using RVC...', 0.5, is_webui, progress)
            ai_vocals_path = voice_change(voice_model, main_vocals_dereverb_path, pitch_change, is_webui)

            display_progress('[~] Applying audio effects to vocals...', 0.8, is_webui, progress)
            ai_vocals_mixed_path = add_audio_effects(ai_vocals_path)

            display_progress('[~] Combining AI Vocals and Instrumentals...', 0.9, is_webui, progress)
            combine_audio([ai_vocals_mixed_path, backup_vocals_path, instrumentals_path], ai_cover_path)

        if not keep_files:
            display_progress('[~] Removing intermediate audio files...', 0.95, is_webui, progress)
            intermediate_files = [vocals_path, main_vocals_path, ai_vocals_path, ai_vocals_mixed_path]
            for file in intermediate_files:
                if file and os.path.exists(file):
                    os.remove(file)

        return ai_cover_path

    except Exception as e:
        raise_exception(str(e), is_webui)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate a AI cover song in the song_output/id directory.', add_help=True)
    parser.add_argument('-i', '--song-input', type=str, required=True, help='Link to a YouTube video or the filepath to a local mp3/wav file to create an AI cover of')
    parser.add_argument('-dir', '--rvc-dirname', type=str, required=True, help='Name of the folder in the rvc_models directory containing the RVC model file and optional index file to use')
    parser.add_argument('-p', '--pitch-change', type=int, required=True, help='Change the pitch of the AI voice. Generally use 12 for male to female conversions and -12 for vice-versa. Use 0 for no change')
    parser.add_argument('-k', '--keep-files', action=argparse.BooleanOptionalAction, help='Whether to keep all intermediate audio files generated in the song_output/id directory, e.g. Isolated Vocals/Instrumentals')
    args = parser.parse_args()

    rvc_dirname = args.rvc_dirname
    if not os.path.exists(os.path.join(rvc_models_dir, rvc_dirname)):
        raise Exception(f'The folder {os.path.join(rvc_models_dir, rvc_dirname)} does not exist.')

    cover_path = song_cover_pipeline(args.song_input, rvc_dirname, args.pitch_change, args.keep_files)
    print(f'[+] Cover generated at {cover_path}')
