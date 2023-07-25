import gc
import json
import os
import re
from argparse import ArgumentParser

import yt_dlp
from pedalboard import Pedalboard, Reverb, Compressor, HighpassFilter
from pedalboard.io import AudioFile
from pydub import AudioSegment

from mdx import run_mdx
from rvc import Config, load_hubert, get_vc, rvc_infer

BASE_DIR = os.path.dirname(os.getcwd())

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


def preprocess_song(yt_link, mdx_model_params, song_id):
    print('Downloading song...')
    orig_song_path = download_audio(yt_link)

    song_output_dir = os.path.join(output_dir, song_id)

    print('Separating Vocals from Instrumental...')
    vocals_path, instrumentals_path = run_mdx(mdx_model_params, song_output_dir, os.path.join(mdxnet_models_dir, 'UVR-MDX-NET-Voc_FT.onnx'), orig_song_path, denoise=True, keep_orig=False)

    print('Separating Main Vocals from Backup Vocals...')
    backup_vocals_path, main_vocals_path = run_mdx(mdx_model_params, song_output_dir, os.path.join(mdxnet_models_dir, 'UVR_MDXNET_KARA_2.onnx'), vocals_path, suffix='Backup', invert_suffix='Main', denoise=True)

    print('Applying DeReverb to Vocals...')
    _, main_vocals_dereverb_path = run_mdx(mdx_model_params, song_output_dir, os.path.join(mdxnet_models_dir, 'Reverb_HQ_By_FoxJoy.onnx'), main_vocals_path, invert_suffix='DeReverb', exclude_main=True, denoise=True)

    return orig_song_path, vocals_path, instrumentals_path, main_vocals_path, backup_vocals_path, main_vocals_dereverb_path


def voice_change(voice_model, vocals_path, pitch_change):
    # determine pitch to change if none provided
    rvc_model_path, rvc_index_path = get_rvc_models(voice_model)
    device = 'cuda:0'
    is_half = True
    config = Config(device, is_half)
    hubert_model = load_hubert(device, is_half, os.path.join(rvc_models_dir, 'hubert_base.pt'))
    cpt, version, net_g, tgt_sr, vc = get_vc(device, is_half, config, rvc_model_path)

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
    main_vocal_audio.overlay(backup_vocal_audio).overlay(instrumental_audio).export(output_path, format='wav')


def song_cover_pipeline(yt_link, voice_model, pitch_change):
    with open(os.path.join(mdxnet_models_dir, 'model_data.json')) as infile:
        mdx_model_params = json.load(infile)

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
    print(f'Cover generated at {cover_path}')
