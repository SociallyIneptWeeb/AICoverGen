import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import shlex
import subprocess
from contextlib import suppress
from urllib.parse import urlparse, parse_qs

import soundfile as sf
import sox
import yt_dlp
from pedalboard import Pedalboard, Reverb, Compressor, HighpassFilter
from pedalboard.io import AudioFile
from pydub import AudioSegment, utils as pydub_utils

from mdx import run_mdx
from rvc import Config, load_hubert, get_vc, rvc_infer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

mdxnet_models_dir = os.path.join(BASE_DIR, "mdxnet_models")
rvc_models_dir = os.path.join(BASE_DIR, "rvc_models")
output_dir = os.path.join(BASE_DIR, "song_output")


def get_youtube_video_id(url, ignore_playlist=True):
    """
    Examples:
    http://youtu.be/SA2iWivDJiE
    http://www.youtube.com/watch?v=_oPAwA_Udwc&feature=feedu
    http://www.youtube.com/embed/SA2iWivDJiE
    http://www.youtube.com/v/SA2iWivDJiE?version=3&amp;hl=en_US
    """
    query = urlparse(url)
    if query.hostname == "youtu.be":
        if query.path[1:] == "watch":
            return query.query[2:]
        return query.path[1:]

    if query.hostname in {"www.youtube.com", "youtube.com", "music.youtube.com"}:
        if not ignore_playlist:
            # use case: get playlist id not current video in playlist
            with suppress(KeyError):
                return parse_qs(query.query)["list"][0]
        if query.path == "/watch":
            return parse_qs(query.query)["v"][0]
        if query.path[:7] == "/watch/":
            return query.path.split("/")[1]
        if query.path[:7] == "/embed/":
            return query.path.split("/")[2]
        if query.path[:3] == "/v/":
            return query.path.split("/")[2]

    # returns None for invalid YouTube url
    return None


def yt_download(link, song_output_dir):
    outtmpl = os.path.join(song_output_dir, "0_%(title)s_Original")
    ydl_opts = {
        "format": "bestaudio",
        "outtmpl": outtmpl,
        "nocheckcertificate": True,
        "ignoreerrors": True,
        "no_warnings": True,
        "quiet": True,
        "extractaudio": True,
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "mp3"}],
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        result = ydl.extract_info(link, download=True)
        download_path = ydl.prepare_filename(result, outtmpl=f"{outtmpl}.mp3")

    return download_path


def get_rvc_model(voice_model):
    rvc_model_filename, rvc_index_filename = None, None
    model_dir = os.path.join(rvc_models_dir, voice_model)
    for file in os.listdir(model_dir):
        ext = os.path.splitext(file)[1]
        if ext == ".pth":
            rvc_model_filename = file
        if ext == ".index":
            rvc_index_filename = file

    if rvc_model_filename is None:
        error_msg = f"No model file exists in {model_dir}."
        raise Exception(error_msg)

    return os.path.join(model_dir, rvc_model_filename), (
        os.path.join(model_dir, rvc_index_filename) if rvc_index_filename else ""
    )


def pitch_shift(audio_path, output_path, pitch_change):
    y, sr = sf.read(audio_path)
    tfm = sox.Transformer()
    tfm.pitch(pitch_change)
    y_shifted = tfm.build_array(input_array=y, sample_rate_in=sr)
    sf.write(output_path, y_shifted, sr)


def json_dumps(thing):
    return json.dumps(
        thing,
        ensure_ascii=False,
        sort_keys=True,
        indent=4,
        separators=(",", ": "),
    )


def json_dump(thing, file):
    return json.dump(
        thing,
        file,
        ensure_ascii=False,
        sort_keys=True,
        indent=4,
        separators=(",", ": "),
    )


def get_hash(thing, size=5):
    return hashlib.blake2b(
        json_dumps(thing).encode("utf-8"), digest_size=size
    ).hexdigest()


def get_unique_base_path(song_dir, prefix, arg_dict, progress, percent, hash_size=5):
    dict_hash = get_hash(arg_dict, size=hash_size)
    while True:
        base_path = os.path.join(song_dir, f"{prefix}_{dict_hash}")
        json_path = f"{base_path}.json"
        if os.path.exists(json_path):
            with open(json_path) as f:
                file_dict = json.load(f)
            if file_dict == arg_dict:
                return base_path
            display_progress("Rehashing...", percent, progress)
            dict_hash = get_hash(dict_hash, size=hash_size)
        else:
            return base_path


def get_file_hash(filepath, size=5):
    with open(filepath, "rb") as f:
        file_hash = hashlib.blake2b(digest_size=size)
        while chunk := f.read(8192):
            file_hash.update(chunk)

    return file_hash.hexdigest()


def display_progress(message, percent, progress=None):
    if progress is None:
        print(message)
    else:
        progress(percent, desc=message)


def voice_change(
    voice_model,
    vocals_path,
    output_path,
    pitch_change,
    f0_method,
    index_rate,
    filter_radius,
    rms_mix_rate,
    protect,
    crepe_hop_length,
    output_sr,
):
    rvc_model_path, rvc_index_path = get_rvc_model(voice_model)
    device = "cuda:0"
    config = Config(device, True)
    hubert_model = load_hubert(
        device, config.is_half, os.path.join(rvc_models_dir, "hubert_base.pt")
    )
    cpt, version, net_g, tgt_sr, vc = get_vc(
        device, config.is_half, config, rvc_model_path
    )

    # convert main vocals
    rvc_infer(
        rvc_index_path,
        index_rate,
        vocals_path,
        output_path,
        pitch_change,
        f0_method,
        cpt,
        version,
        net_g,
        filter_radius,
        tgt_sr,
        rms_mix_rate,
        protect,
        crepe_hop_length,
        vc,
        hubert_model,
        output_sr,
    )
    del hubert_model, cpt
    gc.collect()


def add_audio_effects(
    audio_path,
    output_path,
    reverb_rm_size,
    reverb_wet,
    reverb_dry,
    reverb_damping,
):

    # Initialize audio effects plugins
    board = Pedalboard(
        [
            HighpassFilter(),
            Compressor(ratio=4, threshold_db=-15),
            Reverb(
                room_size=reverb_rm_size,
                dry_level=reverb_dry,
                wet_level=reverb_wet,
                damping=reverb_damping,
            ),
        ]
    )

    with AudioFile(audio_path) as f:
        with AudioFile(output_path, "w", f.samplerate, f.num_channels) as o:
            # Read one second of audio at a time, until the file is empty:
            while f.tell() < f.frames:
                chunk = f.read(int(f.samplerate))
                effected = board(chunk, f.samplerate, reset=False)
                o.write(effected)


def combine_audio(
    audio_paths,
    output_path,
    main_gain,
    backup_gain,
    inst_gain,
    output_format,
    output_sr,
):
    if output_format == "m4a":
        output_format = "ipod"
    elif output_format == "aac":
        output_format = "adts"
    main_vocal_audio = AudioSegment.from_wav(audio_paths[0]) - 4 + main_gain
    backup_vocal_audio = AudioSegment.from_wav(audio_paths[1]) - 6 + backup_gain
    instrumental_audio = AudioSegment.from_wav(audio_paths[2]) - 7 + inst_gain
    combined_audio = main_vocal_audio.overlay(backup_vocal_audio).overlay(
        instrumental_audio
    )
    combined_audio_resampled = combined_audio.set_frame_rate(output_sr)
    combined_audio_resampled.export(output_path, format=output_format)


def make_song_dir(
    song_input,
    voice_model,
    progress,
):
    if not song_input or not voice_model:
        raise Exception(
            "Ensure that the song input field and voice model field is filled."
        )

    display_progress("[~] Starting AI Cover Generation Pipeline...", 0, progress)

    # if youtube url
    if urlparse(song_input).scheme == "https":
        input_type = "yt"
        song_id = get_youtube_video_id(song_input)
        if song_id is None:
            error_msg = "Invalid YouTube url."
            raise Exception(error_msg)
    # local audio file
    else:
        input_type = "local"
        # TODO can probably remove line below
        # filenames cant contain '"' on windows and on linux it should be fine to
        # song_input = song_input.strip('"')
        if os.path.exists(song_input):
            song_id = get_file_hash(song_input)
        else:
            error_msg = f"{song_input} does not exist."
            song_id = None
            raise Exception(error_msg)

    song_dir = os.path.join(output_dir, "temp", song_id)

    Path(song_dir).mkdir(parents=True, exist_ok=True)

    return song_dir, input_type


def retrieve_song(
    song_input,
    input_type,
    song_dir,
    progress,
):
    audio_paths = {
        audio_type: os.path.join(song_dir, file)
        for file in os.listdir(song_dir)
        for audio_type in ["Original", "Stereo"]
        if os.path.splitext(file)[0].endswith(f"_{audio_type}")
    }

    orig_song_path = audio_paths.get("Original")
    stereo_path = audio_paths.get("Stereo")

    if not orig_song_path:
        if input_type == "yt":
            display_progress(
                "[~] Downloading song...",
                0,
                progress,
            )
            song_link = song_input.split("&")[0]
            orig_song_path = yt_download(song_link, song_dir)
        else:
            song_input_base = os.path.basename(song_input)
            song_input_name, song_input_ext = os.path.splitext(song_input_base)
            orig_song_name = f"0_{song_input_name}_Original"
            orig_song_path = os.path.join(song_dir, orig_song_name + song_input_ext)
            shutil.copyfile(song_input, orig_song_path)

    if not stereo_path:
        orig_song_info = pydub_utils.mediainfo(orig_song_path)
        # check if mono
        if orig_song_info["channels"] == "1":
            display_progress(
                "[~] Converting Song to stereo...",
                0.05,
                progress,
            )
            stereo_path_base = os.path.splitext(orig_song_path)[0].removesuffix(
                "_Original"
            )
            stereo_path = f"{stereo_path_base}_Stereo.wav"
            command = shlex.split(
                f'ffmpeg -y -loglevel error -i "{orig_song_path}" -ac 2 -f wav "{stereo_path}"'
            )
            subprocess.run(command)
    return stereo_path or orig_song_path


def separate_vocals(
    song_path,
    song_dir,
    progress,
):
    if not song_path:
        error_msg = "Original song not available. Try and reset server."
        raise Exception(error_msg)

    arg_dict = {
        "input-files": [os.path.basename(song_path)],
    }

    vocals_path_base = get_unique_base_path(
        song_dir, "1_Vocals", arg_dict, progress, 0.075
    )

    instrumentals_path_base = get_unique_base_path(
        song_dir, "1_Instrumental", arg_dict, progress, 0.075
    )

    vocals_path = f"{vocals_path_base}.wav"
    vocals_json_path = f"{vocals_path_base}.json"
    instrumentals_path = f"{instrumentals_path_base}.wav"
    instrumentals_json_path = f"{instrumentals_path_base}.json"

    if not (
        os.path.exists(vocals_path)
        and os.path.exists(vocals_json_path)
        and os.path.exists(instrumentals_path)
        and os.path.exists(instrumentals_json_path)
    ):
        display_progress(
            "[~] Separating Vocals from Instrumentals...",
            0.1,
            progress,
        )
        vocals_path, instrumentals_path = run_mdx(
            mdxnet_models_dir,
            song_dir,
            "UVR-MDX-NET-Voc_FT.onnx",
            song_path,
            suffix=vocals_path_base,
            invert_suffix=instrumentals_path_base,
            denoise=True,
        )
        with (
            open(vocals_json_path, "w") as file1,
            open(instrumentals_json_path, "w") as file2,
        ):
            json_dump(arg_dict, file1)
            json_dump(arg_dict, file2)
    return vocals_path, instrumentals_path


def separate_main_vocals(
    vocals_path,
    song_dir,
    progress,
):

    if not vocals_path:
        error_msg = "Isolated Vocals not available. Try and reset server."
        raise Exception(error_msg)

    arg_dict = {
        "input-files": [os.path.basename(vocals_path)],
    }

    main_vocals_path_base = get_unique_base_path(
        song_dir, "2_Vocals_Main", arg_dict, progress, 0.15
    )

    backup_vocals_path_base = get_unique_base_path(
        song_dir, "2_Vocals_Backup", arg_dict, progress, 0.15
    )

    main_vocals_path = f"{main_vocals_path_base}.wav"
    main_vocals_json_path = f"{main_vocals_path_base}.json"
    backup_vocals_path = f"{backup_vocals_path_base}.wav"
    backup_vocals_json_path = f"{backup_vocals_path_base}.json"

    if not (
        os.path.exists(main_vocals_path)
        and os.path.exists(main_vocals_json_path)
        and os.path.exists(backup_vocals_path)
        and os.path.exists(backup_vocals_json_path)
    ):
        display_progress(
            "[~] Separating Main Vocals from Backup Vocals...",
            0.2,
            progress,
        )
        backup_vocals_path, main_vocals_path = run_mdx(
            mdxnet_models_dir,
            song_dir,
            "UVR_MDXNET_KARA_2.onnx",
            vocals_path,
            suffix=backup_vocals_path_base,
            invert_suffix=main_vocals_path_base,
            denoise=True,
        )
        with (
            open(main_vocals_json_path, "w") as file1,
            open(backup_vocals_json_path, "w") as file2,
        ):
            json_dump(arg_dict, file1)
            json_dump(arg_dict, file2)
    return backup_vocals_path, main_vocals_path


def dereverb_main_vocals(
    main_vocals_path,
    song_dir,
    progress,
):

    if not main_vocals_path:
        error_msg = "Isolated Main Vocals not available. Try and reset server."
        raise Exception(error_msg)

    arg_dict = {
        "input-files": [os.path.basename(main_vocals_path)],
    }

    main_vocals_dereverb_path_base = get_unique_base_path(
        song_dir, "3_Vocals_Main_DeReverb", arg_dict, progress, 0.25
    )

    main_vocals_dereverb_path = f"{main_vocals_dereverb_path_base}.wav"
    main_vocals_dereverb_json_path = f"{main_vocals_dereverb_path_base}.json"

    if not (
        os.path.exists(main_vocals_dereverb_path)
        and os.path.exists(main_vocals_dereverb_json_path)
    ):
        display_progress(
            "[~] Applying DeReverb to Vocals...",
            0.3,
            progress,
        )
        _, main_vocals_dereverb_path = run_mdx(
            mdxnet_models_dir,
            song_dir,
            "Reverb_HQ_By_FoxJoy.onnx",
            main_vocals_path,
            invert_suffix=main_vocals_dereverb_path_base,
            exclude_main=True,
            denoise=True,
        )
        with open(main_vocals_dereverb_json_path, "w") as file:
            json_dump(arg_dict, file)
    return main_vocals_dereverb_path


def convert_main_vocals(
    main_vocals_dereverb_path,
    song_dir,
    voice_model,
    pitch_change,
    pitch_change_all,
    index_rate,
    filter_radius,
    rms_mix_rate,
    protect,
    f0_method,
    crepe_hop_length,
    progress,
):

    main_vocals_dereverb_path_base = os.path.basename(main_vocals_dereverb_path)
    pitch_change = pitch_change * 12 + pitch_change_all
    hop_length_suffix = "" if f0_method != "mangio-crepe" else f"_{crepe_hop_length}"
    arg_dict = {
        "input-files": [main_vocals_dereverb_path_base],
        "voice-model": voice_model,
        "pitch-change": pitch_change,
        "index-rate": index_rate,
        "filter-radius": filter_radius,
        "rms-mix-rate": rms_mix_rate,
        "protect": protect,
        "f0-method": f"{f0_method}{hop_length_suffix}",
    }

    ai_vocals_path_base = get_unique_base_path(
        song_dir, "4_Vocals_Converted", arg_dict, progress, 0.40
    )
    ai_vocals_path = f"{ai_vocals_path_base}.wav"
    ai_vocals_json_path = f"{ai_vocals_path_base}.json"

    if not (os.path.exists(ai_vocals_path) and os.path.exists(ai_vocals_json_path)):
        display_progress("[~] Converting voice using RVC...", 0.5, progress)
        voice_change(
            voice_model,
            main_vocals_dereverb_path,
            ai_vocals_path,
            pitch_change,
            f0_method,
            index_rate,
            filter_radius,
            rms_mix_rate,
            protect,
            crepe_hop_length,
            44100,
        )
        with open(ai_vocals_json_path, "w") as file:
            json_dump(arg_dict, file)
    return ai_vocals_path


def postprocess_main_vocals(
    ai_vocals_path,
    song_dir,
    reverb_rm_size,
    reverb_wet,
    reverb_dry,
    reverb_damping,
    progress,
):

    ai_vocals_path_base = os.path.basename(ai_vocals_path)
    arg_dict = {
        "input-files": [ai_vocals_path_base],
        "reverb-room-size": reverb_rm_size,
        "reverb-wet": reverb_wet,
        "reverb-dry": reverb_dry,
        "reverb-damping": reverb_damping,
    }

    ai_vocals_mixed_path_base = get_unique_base_path(
        song_dir, "5_Vocals_Mixed", arg_dict, progress, 0.7
    )

    ai_vocals_mixed_path = f"{ai_vocals_mixed_path_base}.wav"
    ai_vocals_mixed_json_path = f"{ai_vocals_mixed_path_base}.json"

    if not (
        os.path.exists(ai_vocals_mixed_path)
        and os.path.exists(ai_vocals_mixed_json_path)
    ):
        display_progress("[~] Applying audio effects to Vocals...", 0.8, progress)
        add_audio_effects(
            ai_vocals_path,
            ai_vocals_mixed_path,
            reverb_rm_size,
            reverb_wet,
            reverb_dry,
            reverb_damping,
        )
        with open(ai_vocals_mixed_json_path, "w") as file:
            json_dump(arg_dict, file)
    return ai_vocals_mixed_path


def pitch_shift_background(
    instrumentals_path,
    backup_vocals_path,
    song_dir,
    pitch_change_all,
    progress,
):
    instrumentals_shifted_path = None
    backup_vocals_shifted_path = None
    if pitch_change_all != 0:

        instrumentals_path_base = os.path.basename(instrumentals_path)
        instrumentals_dict = {
            "input-files": [instrumentals_path_base],
            "pitch-shift": pitch_change_all,
        }

        instrumentals_shifted_path_base = get_unique_base_path(
            song_dir, "6_Instrumnetal_Shifted", instrumentals_dict, progress, 0.825
        )

        instrumentals_shifted_path = f"{instrumentals_shifted_path_base}.wav"
        instrumentals_shifted_json_path = f"{instrumentals_shifted_path_base}.json"

        if not (
            os.path.exists(instrumentals_shifted_path)
            and os.path.exists(instrumentals_shifted_json_path)
        ):
            display_progress(
                "[~] Applying pitch change to instrumentals",
                0.85,
                progress,
            )
            pitch_shift(
                instrumentals_path,
                instrumentals_shifted_path,
                pitch_change_all,
            )
            with open(instrumentals_shifted_json_path, "w") as file:
                json_dump(instrumentals_dict, file)

        backup_vocals_base = os.path.basename(backup_vocals_path)

        backup_vocals_dict = {
            "input-files": [backup_vocals_base],
            "pitch-shift": pitch_change_all,
        }

        backup_vocals_shifted_path_base = get_unique_base_path(
            song_dir, "6_Vocals_Backup_Shifted", backup_vocals_dict, progress, 0.865
        )
        backup_vocals_shifted_path = f"{backup_vocals_shifted_path_base}.wav"
        backup_vocals_shifted_json_path = f"{backup_vocals_shifted_path_base}.json"
        if not (
            os.path.exists(backup_vocals_shifted_path)
            and os.path.exists(backup_vocals_shifted_json_path)
        ):
            display_progress(
                "[~] Applying pitch change to backup vocals",
                0.88,
                progress,
            )
            pitch_shift(
                backup_vocals_path,
                backup_vocals_shifted_path,
                pitch_change_all,
            )
            with open(backup_vocals_shifted_json_path, "w") as file:
                json_dump(backup_vocals_dict, file)
    return instrumentals_shifted_path, backup_vocals_shifted_path


def combine_w_background(
    instrumentals_path,
    backup_vocals_path,
    orig_song_path,
    ai_vocals_mixed_path,
    voice_model,
    song_dir,
    main_gain,
    backup_gain,
    inst_gain,
    output_format,
    output_sr,
    keep_files,
    progress,
):
    ai_vocals_mixed_path_base = os.path.basename(ai_vocals_mixed_path)
    backup_vocals_path_base = os.path.basename(backup_vocals_path)
    instrumentals_path_base = os.path.basename(instrumentals_path)

    arg_dict = {
        "input-files": [
            ai_vocals_mixed_path_base,
            backup_vocals_path_base,
            instrumentals_path_base,
        ],
        "main-gain": main_gain,
        "instrument-gain": inst_gain,
        "background-gain": backup_gain,
        "sample-rate": output_sr,
    }

    combined_audio_path_base = get_unique_base_path(
        song_dir, "7_Vocals_Background_Combined", arg_dict, progress, 0.89
    )
    combined_audio_path = f"{combined_audio_path_base}.{output_format}"
    combined_audio_json_path = f"{combined_audio_path_base}.json"

    if not (
        os.path.exists(combined_audio_path) and os.path.exists(combined_audio_json_path)
    ):
        display_progress(
            "[~] Combining AI Vocals and Instrumentals...",
            0.9,
            progress,
        )

        combine_audio(
            [
                ai_vocals_mixed_path,
                backup_vocals_path,
                instrumentals_path,
            ],
            combined_audio_path,
            main_gain,
            backup_gain,
            inst_gain,
            output_format,
            output_sr,
        )
        with open(combined_audio_json_path, "w") as file:
            json_dump(arg_dict, file)

    orig_song_path_base = (
        os.path.splitext(os.path.basename(orig_song_path))[0]
        .removeprefix("0_")
        .removesuffix("_Original")
        .removesuffix("_Stereo")
    )
    ai_cover_path = os.path.join(
        output_dir,
        f"{orig_song_path_base} ({voice_model} Ver).{output_format}",
    )
    shutil.copyfile(combined_audio_path, ai_cover_path)

    if not keep_files:
        display_progress(
            "[~] Removing intermediate audio files...",
            0.95,
            progress,
        )
        shutil.rmtree(song_dir)
    return ai_cover_path


def song_cover_pipeline(
    song_input,
    voice_model,
    pitch_change=0,
    keep_files=True,
    return_files=False,
    main_gain=0,
    backup_gain=0,
    inst_gain=0,
    index_rate=0.5,
    filter_radius=3,
    rms_mix_rate=0.25,
    f0_method="rmvpe",
    crepe_hop_length=128,
    protect=0.33,
    pitch_change_all=0,
    reverb_rm_size=0.15,
    reverb_wet=0.2,
    reverb_dry=0.8,
    reverb_damping=0.7,
    output_format="mp3",
    output_sr=44100,
    progress=None,
):
    song_dir, input_type = make_song_dir(song_input, voice_model, progress)
    orig_song_path = retrieve_song(song_input, input_type, song_dir, progress)
    vocals_path, instrumentals_path = separate_vocals(
        orig_song_path, song_dir, progress
    )
    backup_vocals_path, main_vocals_path = separate_main_vocals(
        vocals_path, song_dir, progress
    )
    main_vocals_dereverb_path = dereverb_main_vocals(
        main_vocals_path, song_dir, progress
    )
    ai_vocals_path = convert_main_vocals(
        main_vocals_dereverb_path,
        song_dir,
        voice_model,
        pitch_change,
        pitch_change_all,
        index_rate,
        filter_radius,
        rms_mix_rate,
        protect,
        f0_method,
        crepe_hop_length,
        progress,
    )
    ai_vocals_mixed_path = postprocess_main_vocals(
        ai_vocals_path,
        song_dir,
        reverb_rm_size,
        reverb_wet,
        reverb_dry,
        reverb_damping,
        progress,
    )
    instrumentals_shifted_path, backup_vocals_shifted_path = pitch_shift_background(
        instrumentals_path,
        backup_vocals_path,
        song_dir,
        pitch_change_all,
        progress,
    )

    ai_cover_path = combine_w_background(
        instrumentals_shifted_path or instrumentals_path,
        backup_vocals_shifted_path or backup_vocals_path,
        orig_song_path,
        ai_vocals_mixed_path,
        voice_model,
        song_dir,
        main_gain,
        backup_gain,
        inst_gain,
        output_format,
        output_sr,
        keep_files,
        progress,
    )
    if keep_files and return_files:
        return (
            orig_song_path,
            vocals_path,
            instrumentals_path,
            main_vocals_path,
            backup_vocals_path,
            main_vocals_dereverb_path,
            ai_vocals_path,
            ai_vocals_mixed_path,
            instrumentals_shifted_path,
            backup_vocals_shifted_path,
            ai_cover_path,
        )
    else:
        return ai_cover_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a AI cover song in the song_output/id directory.",
        add_help=True,
    )
    parser.add_argument(
        "-i",
        "--song-input",
        type=str,
        required=True,
        help="Link to a YouTube video or the filepath to a local mp3/wav file to create an AI cover of",
    )
    parser.add_argument(
        "-dir",
        "--rvc-dirname",
        type=str,
        required=True,
        help="Name of the folder in the rvc_models directory containing the RVC model file and optional index file to use",
    )
    parser.add_argument(
        "-p",
        "--pitch-change",
        type=int,
        required=True,
        help="Change the pitch of AI Vocals only. Generally, use 1 for male to female and -1 for vice-versa. (Octaves)",
    )
    parser.add_argument(
        "-k",
        "--keep-files",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to keep all intermediate audio files generated in the song_output/id directory, e.g. Isolated Vocals/Instrumentals",
    )
    parser.add_argument(
        "-rf",
        "--return-files",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to return all intermediate audio files generated in the song_output/id directory, e.g. Isolated Vocals/Instrumentals",
    )
    parser.add_argument(
        "-ir",
        "--index-rate",
        type=float,
        default=0.5,
        help="A decimal number e.g. 0.5, used to reduce/resolve the timbre leakage problem. If set to 1, more biased towards the timbre quality of the training dataset",
    )
    parser.add_argument(
        "-fr",
        "--filter-radius",
        type=int,
        default=3,
        help="A number between 0 and 7. If >=3: apply median filtering to the harvested pitch results. The value represents the filter radius and can reduce breathiness.",
    )
    parser.add_argument(
        "-rms",
        "--rms-mix-rate",
        type=float,
        default=0.25,
        help="A decimal number e.g. 0.25. Control how much to use the original vocal's loudness (0) or a fixed loudness (1).",
    )
    parser.add_argument(
        "-palgo",
        "--pitch-detection-algo",
        type=str,
        default="rmvpe",
        help="Best option is rmvpe (clarity in vocals), then mangio-crepe (smoother vocals).",
    )
    parser.add_argument(
        "-hop",
        "--crepe-hop-length",
        type=int,
        default=128,
        help="If pitch detection algo is mangio-crepe, controls how often it checks for pitch changes in milliseconds. The higher the value, the faster the conversion and less risk of voice cracks, but there is less pitch accuracy. Recommended: 128.",
    )
    parser.add_argument(
        "-pro",
        "--protect",
        type=float,
        default=0.33,
        help="A decimal number e.g. 0.33. Protect voiceless consonants and breath sounds to prevent artifacts such as tearing in electronic music. Set to 0.5 to disable. Decrease the value to increase protection, but it may reduce indexing accuracy.",
    )
    parser.add_argument(
        "-mv",
        "--main-vol",
        type=int,
        default=0,
        help="Volume change for AI main vocals in decibels. Use -3 to decrease by 3 decibels and 3 to increase by 3 decibels",
    )
    parser.add_argument(
        "-bv",
        "--backup-vol",
        type=int,
        default=0,
        help="Volume change for backup vocals in decibels",
    )
    parser.add_argument(
        "-iv",
        "--inst-vol",
        type=int,
        default=0,
        help="Volume change for instrumentals in decibels",
    )
    parser.add_argument(
        "-pall",
        "--pitch-change-all",
        type=int,
        default=0,
        help="Change the pitch/key of vocals and instrumentals. Changing this slightly reduces sound quality",
    )
    parser.add_argument(
        "-rsize",
        "--reverb-size",
        type=float,
        default=0.15,
        help="Reverb room size between 0 and 1",
    )
    parser.add_argument(
        "-rwet",
        "--reverb-wetness",
        type=float,
        default=0.2,
        help="Reverb wet level between 0 and 1",
    )
    parser.add_argument(
        "-rdry",
        "--reverb-dryness",
        type=float,
        default=0.8,
        help="Reverb dry level between 0 and 1",
    )
    parser.add_argument(
        "-rdamp",
        "--reverb-damping",
        type=float,
        default=0.7,
        help="Reverb damping between 0 and 1",
    )
    parser.add_argument(
        "-oformat",
        "--output-format",
        type=str,
        default="mp3",
        help="format of output audio file",
    )
    parser.add_argument(
        "-osr",
        "--output-sr",
        type=int,
        default=44100,
        help="Sample rate of output audio file.",
    )
    args = parser.parse_args()

    rvc_dirname = args.rvc_dirname
    if not os.path.exists(os.path.join(rvc_models_dir, rvc_dirname)):
        raise Exception(
            f"The folder {os.path.join(rvc_models_dir, rvc_dirname)} does not exist."
        )

    cover_path = song_cover_pipeline(
        args.song_input,
        rvc_dirname,
        args.pitch_change,
        args.keep_files,
        args.return_files,
        main_gain=args.main_vol,
        backup_gain=args.backup_vol,
        inst_gain=args.inst_vol,
        index_rate=args.index_rate,
        filter_radius=args.filter_radius,
        rms_mix_rate=args.rms_mix_rate,
        f0_method=args.pitch_detection_algo,
        crepe_hop_length=args.crepe_hop_length,
        protect=args.protect,
        pitch_change_all=args.pitch_change_all,
        reverb_rm_size=args.reverb_size,
        reverb_wet=args.reverb_wetness,
        reverb_dry=args.reverb_dryness,
        reverb_damping=args.reverb_damping,
        output_format=args.output_format,
        output_sr=args.output_sr,
        progress=None,
    )
    print(f"[+] Cover generated at {cover_path}")
