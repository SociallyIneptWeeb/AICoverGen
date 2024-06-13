import argparse
import gc
import os
import glob
from pathlib import Path, PurePath
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

from common import MDXNET_MODELS_DIR, RVC_MODELS_DIR, SONGS_DIR, TEMP_AUDIO_DIR
from common import (
    display_progress,
    json_dump,
    json_load,
    get_hash,
    get_file_hash,
    get_rvc_model,
)
from mdx import run_mdx
from rvc import Config, load_hubert, get_vc, rvc_infer


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


def yt_download(link, song_dir):
    outtmpl = os.path.join(song_dir, "0_%(title)s_Original")
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "format": "bestaudio",
        "outtmpl": outtmpl,
        "ignoreerrors": True,
        "nocheckcertificate": True,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": 0,
            }
        ],
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        result = ydl.extract_info(link, download=True)
        download_path = ydl.prepare_filename(result, outtmpl=f"{outtmpl}.mp3")

    return download_path


def get_cached_input_paths():
    # TODO if we later add .json file for input then we need to exclude those here
    input_paths_pattern = os.path.join(TEMP_AUDIO_DIR, "*", "0_*_Original*")
    return glob.glob(input_paths_pattern)


def pitch_shift(audio_path, output_path, n_semi_tones):
    y, sr = sf.read(audio_path)
    tfm = sox.Transformer()
    tfm.pitch(n_semi_tones)
    y_shifted = tfm.build_array(input_array=y, sample_rate_in=sr)
    sf.write(output_path, y_shifted, sr)


# TODO consider increasing size to 16
# otherwise we might have problems with hash collisions
# when using app as CLI
def get_unique_base_path(song_dir, prefix, arg_dict, progress, percent, hash_size=5):
    dict_hash = get_hash(arg_dict, size=hash_size)
    while True:
        base_path = os.path.join(song_dir, f"{prefix}_{dict_hash}")
        json_path = f"{base_path}.json"
        if os.path.exists(json_path):
            file_dict = json_load(json_path)
            if file_dict == arg_dict:
                return base_path
            display_progress("[~] Rehashing...", percent, progress)
            dict_hash = get_hash(dict_hash, size=hash_size)
        else:
            return base_path


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
        device, config.is_half, os.path.join(RVC_MODELS_DIR, "hubert_base.pt")
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
    main_vocal_audio = AudioSegment.from_wav(audio_paths[0]) + main_gain
    backup_vocal_audio = AudioSegment.from_wav(audio_paths[1]) + backup_gain
    instrumental_audio = AudioSegment.from_wav(audio_paths[2]) + inst_gain
    combined_audio = main_vocal_audio.overlay(backup_vocal_audio).overlay(
        instrumental_audio
    )
    combined_audio_resampled = combined_audio.set_frame_rate(output_sr)
    combined_audio_resampled.export(output_path, format=output_format)


def get_named_song_dirs():
    input_paths = get_cached_input_paths()
    named_song_dirs = []

    for path in input_paths:
        song_dir, song_basename = os.path.split(path)
        song_name = (
            os.path.splitext(song_basename)[0]
            .removeprefix("0_")
            .removesuffix("_Original")
        )
        named_song_dirs.append((song_name, song_dir))
    return sorted(named_song_dirs, key=lambda x: x[0])


def delete_intermediate_audio(song_inputs, progress=None, percentages=[0.0]):
    if len(percentages) != 1:
        raise ValueError("Percentages must be a list of length 1.")
    if not song_inputs:
        raise Exception(
            "Song inputs missing! Please provide a non-empty list of song directories"
        )
    display_progress(
        "[~] Deleting intermediate audio files for selected songs...",
        percentages[0],
        progress,
    )
    for song_input in song_inputs:
        if not os.path.isdir(song_input):
            raise Exception(f"Song directory '{song_input}' does not exist.")

        if not PurePath(song_input).is_relative_to(TEMP_AUDIO_DIR):
            raise Exception(
                f"Song directory '{song_input}' is not located in the temporary audio directory."
            )
        shutil.rmtree(song_input)
    return "[+] Successfully deleted intermediate audio files for selected songs!"


def delete_all_intermediate_audio(progress=None, percentages=[0.0]):
    if len(percentages) != 1:
        raise ValueError("Percentages must be a list of length 1.")
    display_progress(
        "[~] Deleting all intermediate audio files...", percentages[0], progress
    )
    if os.path.isdir(TEMP_AUDIO_DIR):
        shutil.rmtree(TEMP_AUDIO_DIR)

    return "[+] All intermediate audio files successfully deleted!"


def convert_to_stereo(song_path, song_dir, progress=None, percentages=[0.0, 0.5]):
    if len(percentages) != 2:
        raise ValueError("Percentages must be a list of length 2.")
    if not song_path:
        raise Exception("Input song missing!")
    if not os.path.isfile(song_path):
        raise Exception("Input song does not exist!")
    if not song_dir:
        raise Exception("Song directory missing!")
    if not os.path.isdir(song_dir):
        raise Exception("Song directory does not exist!")

    stereo_path = song_path

    song_info = pydub_utils.mediainfo(song_path)
    if song_info["channels"] == "1":
        arg_dict = {
            "input-files": [
                {"name": os.path.basename(song_path), "hash": get_file_hash(song_path)}
            ],
        }
        stereo_path_base = get_unique_base_path(
            song_dir, "0_Stereo", arg_dict, progress, percentages[0]
        )
        stereo_path = f"{stereo_path_base}.wav"
        stereo_json_path = f"{stereo_path_base}.json"
        if not (os.path.exists(stereo_path) and os.path.exists(stereo_json_path)):
            display_progress(
                "[~] Converting song to stereo...",
                percentages[1],
                progress,
            )
            command = shlex.split(
                f'ffmpeg -y -loglevel error -i "{song_path}" -ac 2 -f wav "{stereo_path}"'
            )
            subprocess.run(command)
            json_dump(arg_dict, stereo_json_path)

    return stereo_path


def make_song_dir(song_input, progress=None, percentages=[0.0]):
    if len(percentages) != 1:
        raise ValueError("Percentages must be a list of length 1.")
    if os.path.isdir(song_input):
        if not PurePath(song_input).is_relative_to(TEMP_AUDIO_DIR):
            raise Exception("Song directory not located in temporary audio directory.")
        display_progress(
            "[~] Using existing song directory...", percentages[0], progress
        )
        input_type = "local"
        return song_input, input_type

    display_progress("[~] Creating song directory...", percentages[0], progress)
    # if youtube url
    if urlparse(song_input).scheme == "https":
        input_type = "yt"
        song_id = get_youtube_video_id(song_input)
        if song_id is None:
            raise Exception("Invalid YouTube url!")
    # local audio file
    else:
        input_type = "local"
        # TODO can probably remove line below
        # filenames cant contain '"' on windows and on linux it should be fine to
        # song_input = song_input.strip('"')
        if os.path.isfile(song_input):
            song_id = get_file_hash(song_input)
        else:
            raise Exception(f"File {song_input} does not exist.")

    song_dir = os.path.join(TEMP_AUDIO_DIR, song_id)

    Path(song_dir).mkdir(parents=True, exist_ok=True)

    return song_dir, input_type


def retrieve_song(
    song_input,
    progress=None,
    percentages=[i / 4 for i in range(4)],
):
    if len(percentages) != 4:
        raise ValueError("Percentages must be a list of length 4.")
    if not song_input:
        raise Exception(
            "Song input missing! Please provide a valid YouTube url or local audio file."
        )

    song_dir, input_type = make_song_dir(song_input, progress, percentages[:1])
    orig_song_path = next(
        iter(glob.glob(os.path.join(song_dir, "0_*_Original*"))), None
    )

    if not orig_song_path:
        if input_type == "yt":
            display_progress(
                "[~] Downloading song...",
                percentages[1],
                progress,
            )
            song_link = song_input.split("&")[0]
            orig_song_path = yt_download(song_link, song_dir)
        else:
            display_progress("[~] Copying song...", percentages[1], progress)
            song_input_base = os.path.basename(song_input)
            song_input_name, song_input_ext = os.path.splitext(song_input_base)
            orig_song_name = f"0_{song_input_name}_Original"
            orig_song_path = os.path.join(song_dir, orig_song_name + song_input_ext)
            shutil.copyfile(song_input, orig_song_path)

    stereo_path = convert_to_stereo(orig_song_path, song_dir, progress, percentages[2:])
    return stereo_path, song_dir


def separate_vocals(
    song_path,
    song_dir,
    progress=None,
    percentages=[i / 4 for i in range(4)],
):
    if len(percentages) != 4:
        raise ValueError("Percentages must be a list of length 4.")
    if not song_path:
        raise Exception("Input song missing!")
    if not os.path.isfile(song_path):
        raise Exception("Input song does not exist!")
    if not song_dir:
        raise Exception("Song directory missing!")
    if not os.path.isdir(song_dir):
        raise Exception("Song directory does not exist!")

    song_path = convert_to_stereo(song_path, song_dir, progress, percentages[:2])

    arg_dict = {
        "input-files": [
            {"name": os.path.basename(song_path), "hash": get_file_hash(song_path)}
        ],
    }

    vocals_path_base = get_unique_base_path(
        song_dir, "1_Vocals", arg_dict, progress, percentages[2]
    )

    instrumentals_path_base = get_unique_base_path(
        song_dir, "1_Instrumental", arg_dict, progress, percentages[2]
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
            "[~] Separating vocals from instrumentals...",
            percentages[3],
            progress,
        )
        run_mdx(
            MDXNET_MODELS_DIR,
            song_dir,
            "UVR-MDX-NET-Voc_FT.onnx",
            song_path,
            suffix=vocals_path_base,
            invert_suffix=instrumentals_path_base,
            denoise=True,
        )
        json_dump(arg_dict, vocals_json_path)
        json_dump(arg_dict, instrumentals_json_path)
    return vocals_path, instrumentals_path


def separate_main_vocals(
    vocals_path,
    song_dir,
    progress=None,
    percentages=[i / 4 for i in range(4)],
):
    if len(percentages) != 4:
        raise ValueError("Percentages must be a list of length 4.")

    if not vocals_path:
        raise Exception("Vocals missing!")
    if not os.path.isfile(vocals_path):
        raise Exception("Vocals do not exist!")
    if not song_dir:
        raise Exception("Song directory missing!")
    if not os.path.isdir(song_dir):
        raise Exception("song directory does not exist!")

    vocals_path = convert_to_stereo(vocals_path, song_dir, progress, percentages[:2])

    arg_dict = {
        "input-files": [
            {
                "name": os.path.basename(vocals_path),
                "hash": get_file_hash(vocals_path),
            }
        ],
    }

    main_vocals_path_base = get_unique_base_path(
        song_dir, "2_Vocals_Main", arg_dict, progress, percentages[2]
    )

    backup_vocals_path_base = get_unique_base_path(
        song_dir, "2_Vocals_Backup", arg_dict, progress, percentages[2]
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
            "[~] Separating main vocals from backup vocals...",
            percentages[3],
            progress,
        )
        run_mdx(
            MDXNET_MODELS_DIR,
            song_dir,
            "UVR_MDXNET_KARA_2.onnx",
            vocals_path,
            suffix=backup_vocals_path_base,
            invert_suffix=main_vocals_path_base,
            denoise=True,
        )
        json_dump(arg_dict, main_vocals_json_path)
        json_dump(arg_dict, backup_vocals_json_path)
    return main_vocals_path, backup_vocals_path


def dereverb_main_vocals(
    main_vocals_path, song_dir, progress=None, percentages=[i / 4 for i in range(4)]
):
    if len(percentages) != 4:
        raise ValueError("Percentages must be a list of length 4.")

    if not main_vocals_path:
        raise Exception("Vocals missing!")
    if not os.path.isfile(main_vocals_path):
        raise Exception("Vocals do not exist!")
    if not song_dir:
        raise Exception("Song directory missing!")
    if not os.path.isdir(song_dir):
        raise Exception("song directory does not exist!")

    main_vocals_path = convert_to_stereo(
        main_vocals_path, song_dir, progress, percentages[:2]
    )

    arg_dict = {
        "input-files": [
            {
                "name": os.path.basename(main_vocals_path),
                "hash": get_file_hash(main_vocals_path),
            }
        ],
    }

    main_vocals_dereverb_path_base = get_unique_base_path(
        song_dir, "3_Vocals_Main_DeReverb", arg_dict, progress, percentages[2]
    )
    main_vocals_reverb_path_base = get_unique_base_path(
        song_dir, "3_Vocals_Main_Reverb", arg_dict, progress, percentages[2]
    )

    main_vocals_dereverb_path = f"{main_vocals_dereverb_path_base}.wav"
    main_vocals_dereverb_json_path = f"{main_vocals_dereverb_path_base}.json"

    main_vocals_reverb_path = f"{main_vocals_reverb_path_base}.wav"
    main_vocals_reverb_json_path = f"{main_vocals_reverb_path_base}.json"

    if not (
        os.path.exists(main_vocals_dereverb_path)
        and os.path.exists(main_vocals_dereverb_json_path)
        and os.path.exists(main_vocals_reverb_path)
        and os.path.exists(main_vocals_reverb_json_path)
    ):
        display_progress(
            "[~] De-reverbing main vocals...",
            percentages[3],
            progress,
        )
        run_mdx(
            MDXNET_MODELS_DIR,
            song_dir,
            "Reverb_HQ_By_FoxJoy.onnx",
            main_vocals_path,
            suffix=main_vocals_reverb_path_base,
            invert_suffix=main_vocals_dereverb_path_base,
            denoise=True,
        )
        json_dump(arg_dict, main_vocals_dereverb_json_path)
        json_dump(arg_dict, main_vocals_reverb_json_path)
    return main_vocals_dereverb_path, main_vocals_reverb_path


def convert_main_vocals(
    main_vocals_dereverb_path,
    song_dir,
    voice_model,
    pitch_change_vocals,
    pitch_change_all,
    index_rate,
    filter_radius,
    rms_mix_rate,
    protect,
    f0_method,
    crepe_hop_length,
    progress=None,
    percentages=[i / 4 for i in range(4)],
):
    if len(percentages) != 4:
        raise ValueError("Percentages must be a list of length 4.")
    if not main_vocals_dereverb_path:
        raise Exception("Vocals missing!")
    if not os.path.isfile(main_vocals_dereverb_path):
        raise Exception("Vocals do not exist!")
    if not song_dir:
        raise Exception("Song directory missing!")
    if not os.path.isdir(song_dir):
        raise Exception("song directory does not exist!")
    if not voice_model:
        raise Exception("Voice model missing!")
    if not os.path.isdir(os.path.join(RVC_MODELS_DIR, voice_model)):
        raise Exception("Voice model does not exist!")

    main_vocals_dereverb_path = convert_to_stereo(
        main_vocals_dereverb_path, song_dir, progress, percentages[:2]
    )

    main_vocals_dereverb_path_base = os.path.basename(main_vocals_dereverb_path)
    pitch_change = pitch_change_vocals * 12 + pitch_change_all
    hop_length_suffix = "" if f0_method != "mangio-crepe" else f"_{crepe_hop_length}"
    arg_dict = {
        "input-files": [
            {
                "name": main_vocals_dereverb_path_base,
                "hash": get_file_hash(main_vocals_dereverb_path),
            }
        ],
        "voice-model": voice_model,
        "pitch-shift": pitch_change,
        "index-rate": index_rate,
        "filter-radius": filter_radius,
        "rms-mix-rate": rms_mix_rate,
        "protect": protect,
        "f0-method": f"{f0_method}{hop_length_suffix}",
    }

    ai_vocals_path_base = get_unique_base_path(
        song_dir, "4_Vocals_Converted", arg_dict, progress, percentages[2]
    )
    ai_vocals_path = f"{ai_vocals_path_base}.wav"
    ai_vocals_json_path = f"{ai_vocals_path_base}.json"

    if not (os.path.exists(ai_vocals_path) and os.path.exists(ai_vocals_json_path)):
        display_progress(
            "[~] Converting main vocals using RVC...", percentages[3], progress
        )
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
        json_dump(arg_dict, ai_vocals_json_path)
    return ai_vocals_path


def postprocess_main_vocals(
    ai_vocals_path,
    song_dir,
    reverb_rm_size,
    reverb_wet,
    reverb_dry,
    reverb_damping,
    progress=None,
    percentages=[i / 4 for i in range(4)],
):

    if len(percentages) != 4:
        raise ValueError("Percentages must be a list of length 4.")
    if not ai_vocals_path:
        raise Exception("Vocals missing!")
    if not os.path.isfile(ai_vocals_path):
        raise Exception("Vocals do not exist!")
    if not song_dir:
        raise Exception("Song directory missing!")
    if not os.path.isdir(song_dir):
        raise Exception("song directory does not exist!")

    ai_vocals_path = convert_to_stereo(
        ai_vocals_path, song_dir, progress, percentages[:2]
    )

    ai_vocals_path_base = os.path.basename(ai_vocals_path)
    arg_dict = {
        "input-files": [
            {"name": ai_vocals_path_base, "hash": get_file_hash(ai_vocals_path)}
        ],
        "reverb-room-size": reverb_rm_size,
        "reverb-wet": reverb_wet,
        "reverb-dry": reverb_dry,
        "reverb-damping": reverb_damping,
    }

    ai_vocals_mixed_path_base = get_unique_base_path(
        song_dir, "5_Vocals_Mixed", arg_dict, progress, percentages[2]
    )

    ai_vocals_mixed_path = f"{ai_vocals_mixed_path_base}.wav"
    ai_vocals_mixed_json_path = f"{ai_vocals_mixed_path_base}.json"

    if not (
        os.path.exists(ai_vocals_mixed_path)
        and os.path.exists(ai_vocals_mixed_json_path)
    ):
        display_progress(
            "[~] Applying audio effects to converted vocals...",
            percentages[3],
            progress,
        )
        add_audio_effects(
            ai_vocals_path,
            ai_vocals_mixed_path,
            reverb_rm_size,
            reverb_wet,
            reverb_dry,
            reverb_damping,
        )
        json_dump(arg_dict, ai_vocals_mixed_json_path)
    return ai_vocals_mixed_path


def pitch_shift_background(
    instrumentals_path,
    backup_vocals_path,
    song_dir,
    pitch_change_all,
    progress=None,
    percentages=[i / 8 for i in range(8)],
):
    if len(percentages) != 8:
        raise ValueError("Percentages must be a list of length 8.")

    if not instrumentals_path:
        raise Exception("Instrumentals missing!")
    if not os.path.isfile(instrumentals_path):
        raise Exception("Instrumentals do not exist!")
    if not backup_vocals_path:
        raise Exception("Backup vocals missing!")
    if not os.path.isfile(backup_vocals_path):
        raise Exception("Backup vocals do not exist!")
    if not song_dir:
        raise Exception("Song directory missing!")
    if not os.path.isdir(song_dir):
        raise Exception("song directory does not exist!")

    instrumentals_shifted_path = instrumentals_path
    backup_vocals_shifted_path = backup_vocals_path

    if pitch_change_all != 0:
        instrumentals_path = convert_to_stereo(
            instrumentals_path, song_dir, progress, percentages[:2]
        )
        backup_vocals_path = convert_to_stereo(
            backup_vocals_path, song_dir, progress, percentages[2:4]
        )
        instrumentals_path_base = os.path.basename(instrumentals_path)
        instrumentals_dict = {
            "input-files": [
                {
                    "name": instrumentals_path_base,
                    "hash": get_file_hash(instrumentals_path),
                }
            ],
            "pitch-shift": pitch_change_all,
        }

        instrumentals_shifted_path_base = get_unique_base_path(
            song_dir,
            "6_Instrumnetal_Shifted",
            instrumentals_dict,
            progress,
            percentages[4],
        )

        instrumentals_shifted_path = f"{instrumentals_shifted_path_base}.wav"
        instrumentals_shifted_json_path = f"{instrumentals_shifted_path_base}.json"

        if not (
            os.path.exists(instrumentals_shifted_path)
            and os.path.exists(instrumentals_shifted_json_path)
        ):
            display_progress(
                "[~] Applying pitch shift to instrumentals",
                percentages[5],
                progress,
            )
            pitch_shift(
                instrumentals_path,
                instrumentals_shifted_path,
                pitch_change_all,
            )
            json_dump(instrumentals_dict, instrumentals_shifted_json_path)

        backup_vocals_base = os.path.basename(backup_vocals_path)

        backup_vocals_dict = {
            "input-files": [
                {"name": backup_vocals_base, "hash": get_file_hash(backup_vocals_path)}
            ],
            "pitch-shift": pitch_change_all,
        }

        backup_vocals_shifted_path_base = get_unique_base_path(
            song_dir,
            "6_Vocals_Backup_Shifted",
            backup_vocals_dict,
            progress,
            percentages[6],
        )
        backup_vocals_shifted_path = f"{backup_vocals_shifted_path_base}.wav"
        backup_vocals_shifted_json_path = f"{backup_vocals_shifted_path_base}.json"
        if not (
            os.path.exists(backup_vocals_shifted_path)
            and os.path.exists(backup_vocals_shifted_json_path)
        ):
            display_progress(
                "[~] Applying pitch shift to backup vocals",
                percentages[7],
                progress,
            )
            pitch_shift(
                backup_vocals_path,
                backup_vocals_shifted_path,
                pitch_change_all,
            )
            json_dump(backup_vocals_dict, backup_vocals_shifted_json_path)
    return instrumentals_shifted_path, backup_vocals_shifted_path


def get_song_cover_name(
    mixed_vocals_path, song_dir, voice_model, progress=None, percentages=[0.0]
):
    if len(percentages) != 1:
        raise ValueError("Percentages must be a list of length 1.")
    display_progress("[~] Getting song cover name...", percentages[0], progress)
    orig_song_prefix = "Unknown"
    # NOTE orig_song_paths should never contain more than one element
    orig_song_path = (
        next(iter(glob.glob(os.path.join(song_dir, "0_*_Original*"))), None)
        if song_dir
        else None
    )
    if orig_song_path:
        orig_song_base_path = os.path.splitext(orig_song_path)[0]
        orig_song_path_without_ext = os.path.basename(orig_song_base_path)

        orig_song_prefix = orig_song_path_without_ext.removeprefix("0_").removesuffix(
            "_Original"
        )

    if not voice_model:
        voice_model = "Unknown"
        if mixed_vocals_path and song_dir:
            mixed_vocals_path_no_ext = os.path.splitext(mixed_vocals_path)[0]
            mixed_vocals_path_base = os.path.basename(mixed_vocals_path_no_ext)
            mixed_vocals_json_path = os.path.join(
                song_dir, f"{mixed_vocals_path_base}.json"
            )
            if os.path.isfile(mixed_vocals_json_path):
                mixed_vocals_json_dict = json_load(mixed_vocals_json_path)
                input_files = mixed_vocals_json_dict.get("input-files")
                input_path = input_files[0].get("name") if input_files else None
                if input_path:
                    input_path_no_ext = os.path.splitext(input_path)[0]
                    ai_vocals_json_path = os.path.join(
                        song_dir, f"{input_path_no_ext}.json"
                    )
                    if os.path.isfile(ai_vocals_json_path):
                        ai_vocals_dict = json_load(ai_vocals_json_path)
                        voice_model = ai_vocals_dict.get("voice-model", voice_model)

    return f"{orig_song_prefix} ({voice_model} Ver)"


def mix_w_background(
    ai_vocals_mixed_path,
    instrumentals_path,
    backup_vocals_path,
    song_dir,
    main_gain,
    backup_gain,
    inst_gain,
    output_sr,
    output_format,
    output_name,
    keep_files,
    progress=None,
    percentages=[i / 6 for i in range(6)],
):
    if len(percentages) != 6:
        raise ValueError("Percentages must be a list of length 6.")
    if not ai_vocals_mixed_path:
        raise Exception("Main vocals missing!")
    if not os.path.isfile(ai_vocals_mixed_path):
        raise Exception("Main vocals do not exist!")
    if not instrumentals_path:
        raise Exception("Instrumentals missing!")
    if not os.path.isfile(instrumentals_path):
        raise Exception("Instrumentals do not exist!")
    if not backup_vocals_path:
        raise Exception("Backup vocals missing!")
    if not os.path.isfile(backup_vocals_path):
        raise Exception("Backup vocals do not exist!")
    if not song_dir:
        raise Exception("Song directory missing!")
    if not os.path.isdir(song_dir):
        raise Exception("song directory does not exist!")

    ai_vocals_mixed_path = convert_to_stereo(
        ai_vocals_mixed_path, song_dir, progress, percentages[:2]
    )

    ai_vocals_mixed_path_base = os.path.basename(ai_vocals_mixed_path)
    backup_vocals_path_base = os.path.basename(backup_vocals_path)
    instrumentals_path_base = os.path.basename(instrumentals_path)

    arg_dict = {
        "input-files": [
            {
                "name": ai_vocals_mixed_path_base,
                "hash": get_file_hash(ai_vocals_mixed_path),
            },
            {
                "name": backup_vocals_path_base,
                "hash": get_file_hash(backup_vocals_path),
            },
            {
                "name": instrumentals_path_base,
                "hash": get_file_hash(instrumentals_path),
            },
            backup_vocals_path_base,
            instrumentals_path_base,
        ],
        "main-gain": main_gain,
        "instrument-gain": inst_gain,
        "background-gain": backup_gain,
        "sample-rate": output_sr,
    }

    combined_audio_path_base = get_unique_base_path(
        song_dir, "7_Vocals_Background_Combined", arg_dict, progress, percentages[2]
    )
    combined_audio_path = f"{combined_audio_path_base}.{output_format}"
    combined_audio_json_path = f"{combined_audio_path_base}.json"

    if not (
        os.path.exists(combined_audio_path) and os.path.exists(combined_audio_json_path)
    ):
        display_progress(
            "[~] Combining post-processed vocals and background tracks...",
            percentages[3],
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
        json_dump(arg_dict, combined_audio_json_path)

    if not output_name:
        output_name = get_song_cover_name(
            ai_vocals_mixed_path, song_dir, None, progress, percentages[4:5]
        )

    ai_cover_path = os.path.join(
        SONGS_DIR,
        f"{output_name}.{output_format}",
    )
    shutil.copyfile(combined_audio_path, ai_cover_path)

    if not keep_files:
        display_progress(
            "[~] Removing intermediate audio files...",
            percentages[5],
            progress,
        )
        shutil.rmtree(song_dir)
    return ai_cover_path


def run_pipeline(
    song_input,
    voice_model,
    pitch_change_vocals=0,
    pitch_change_all=0,
    index_rate=0.5,
    filter_radius=3,
    rms_mix_rate=0.25,
    protect=0.33,
    f0_method="rmvpe",
    crepe_hop_length=128,
    reverb_rm_size=0.15,
    reverb_wet=0.2,
    reverb_dry=0.8,
    reverb_damping=0.7,
    main_gain=0,
    backup_gain=0,
    inst_gain=0,
    output_sr=44100,
    output_format="mp3",
    output_name=None,
    keep_files=True,
    return_files=False,
    progress=None,
):
    display_progress("[~] Starting song cover generation pipeline...", 0, progress)
    percentages = [i / 38 for i in range(38)]
    orig_song_path, song_dir = retrieve_song(song_input, progress, percentages[:4])
    vocals_path, instrumentals_path = separate_vocals(
        orig_song_path, song_dir, progress, percentages[4:8]
    )
    main_vocals_path, backup_vocals_path = separate_main_vocals(
        vocals_path, song_dir, progress, percentages[8:12]
    )
    main_vocals_dereverb_path, reverb_path = dereverb_main_vocals(
        main_vocals_path, song_dir, progress, percentages[12:16]
    )
    ai_vocals_path = convert_main_vocals(
        main_vocals_dereverb_path,
        song_dir,
        voice_model,
        pitch_change_vocals,
        pitch_change_all,
        index_rate,
        filter_radius,
        rms_mix_rate,
        protect,
        f0_method,
        crepe_hop_length,
        progress,
        percentages[16:20],
    )
    ai_vocals_mixed_path = postprocess_main_vocals(
        ai_vocals_path,
        song_dir,
        reverb_rm_size,
        reverb_wet,
        reverb_dry,
        reverb_damping,
        progress,
        percentages[20:24],
    )
    instrumentals_shifted_path, backup_vocals_shifted_path = pitch_shift_background(
        instrumentals_path,
        backup_vocals_path,
        song_dir,
        pitch_change_all,
        progress,
        percentages[24:32],
    )

    ai_cover_path = mix_w_background(
        ai_vocals_mixed_path,
        instrumentals_shifted_path or instrumentals_path,
        backup_vocals_shifted_path or backup_vocals_path,
        song_dir,
        main_gain,
        backup_gain,
        inst_gain,
        output_sr,
        output_format,
        output_name,
        keep_files,
        progress,
        percentages[32:],
    )
    if keep_files and return_files:
        return (
            orig_song_path,
            vocals_path,
            instrumentals_path,
            main_vocals_path,
            backup_vocals_path,
            main_vocals_dereverb_path,
            reverb_path,
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
        description="Generate a cover song in the song_output/id directory.",
        add_help=True,
    )
    parser.add_argument(
        "-i",
        "--song-input",
        type=str,
        required=True,
        help="Link to a song on YouTube or the full path of a local audio file",
    )
    parser.add_argument(
        "-dir",
        "--rvc-dirname",
        type=str,
        required=True,
        help="Name of the folder in the rvc_models directory containing the RVC model file and optional index file to use",
    )
    parser.add_argument(
        "-pv",
        "--pitch-change-vocals",
        type=int,
        required=True,
        help="Shift the pitch of converted vocals only. Measured in octaves. Generally, use 1 for male to female and -1 for vice-versa.",
    )
    parser.add_argument(
        "-pall",
        "--pitch-change-all",
        type=int,
        default=0,
        help="Shift pitch of converted vocals, backup vocals and instrumentals. Measured in semi-tones. Altering this slightly reduces sound quality",
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
        help="A decimal number e.g. 0.25. Control how much to use the loudness of the input vocals (0) or a fixed loudness (1).",
    )
    parser.add_argument(
        "-pro",
        "--protect",
        type=float,
        default=0.33,
        help="A decimal number e.g. 0.33. Protect voiceless consonants and breath sounds to prevent artifacts such as tearing in electronic music. Set to 0.5 to disable. Decrease the value to increase protection, but it may reduce indexing accuracy.",
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
        "-mv",
        "--main-vol",
        type=int,
        default=0,
        help="Volume change for converted main vocals. Measured in dB. Use -3 to decrease by 3 dB and 3 to increase by 3 dB",
    )
    parser.add_argument(
        "-bv",
        "--backup-vol",
        type=int,
        default=0,
        help="Volume change for backup vocals. Measured in dB",
    )
    parser.add_argument(
        "-iv",
        "--inst-vol",
        type=int,
        default=0,
        help="Volume change for instrumentals. Measured in dB",
    )
    parser.add_argument(
        "-osr",
        "--output-sr",
        type=int,
        default=44100,
        help="Sample rate of output audio file.",
    )
    parser.add_argument(
        "-oformat",
        "--output-format",
        type=str,
        default="mp3",
        help="format of output audio file",
    )
    parser.add_argument(
        "-k",
        "--keep-files",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to keep all intermediate audio files",
    )
    args = parser.parse_args()

    rvc_dirname = args.rvc_dirname
    if not os.path.isdir(os.path.join(RVC_MODELS_DIR, rvc_dirname)):
        raise Exception(
            f"The folder {os.path.join(RVC_MODELS_DIR, rvc_dirname)} does not exist."
        )

    song_cover_path = run_pipeline(
        song_input=args.song_input,
        voice_model=rvc_dirname,
        pitch_change_vocals=args.pitch_change_vocals,
        pitch_change_all=args.pitch_change_all,
        index_rate=args.index_rate,
        filter_radius=args.filter_radius,
        rms_mix_rate=args.rms_mix_rate,
        protect=args.protect,
        f0_method=args.pitch_detection_algo,
        crepe_hop_length=args.crepe_hop_length,
        reverb_rm_size=args.reverb_size,
        reverb_wet=args.reverb_wetness,
        reverb_dry=args.reverb_dryness,
        reverb_damping=args.reverb_damping,
        main_gain=args.main_vol,
        backup_gain=args.backup_vol,
        inst_gain=args.inst_vol,
        output_sr=args.output_sr,
        output_format=args.output_format,
        keep_files=args.keep_files,
        return_files=False,
        progress=None,
    )
    print(f"[+] Cover generated at {song_cover_path}")
