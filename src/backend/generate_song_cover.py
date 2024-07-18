from typing import Any
from extra_typing import InputType, F0Method, InputAudioExt, OutputAudioExt
import gradio as gr

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
from pedalboard import Reverb, Compressor, HighpassFilter
from pedalboard._pedalboard import Pedalboard
from pedalboard.io import AudioFile
from pydub import AudioSegment, utils as pydub_utils
from audio_separator.separator import Separator

from common import RVC_MODELS_DIR, SEPARATOR_MODELS_DIR

from backend.common import (
    INTERMEDIATE_AUDIO_DIR,
    OUTPUT_AUDIO_DIR,
    display_progress,
    get_path_stem,
    json_dump,
    json_load,
    get_hash,
    get_file_hash,
    get_rvc_model,
)
from backend.exceptions import (
    InputMissingError,
    PathNotFoundError,
    InvalidPathError,
)
from vc.rvc import Config, load_hubert, get_vc, rvc_infer
from logging import WARNING

SEPARATOR = Separator(
    log_level=WARNING,
    model_file_dir=SEPARATOR_MODELS_DIR,
    output_dir=INTERMEDIATE_AUDIO_DIR,
    mdx_params={
        "hop_length": 1024,
        "segment_size": 256,
        "overlap": 0.001,
        "batch_size": 1,
        "enable_denoise": False,
    },
    mdxc_params={
        "segment_size": 256,
        "batch_size": 1,
        "overlap": 2,
    },
)


def _get_youtube_video_id(url: str, ignore_playlist: bool = True) -> str | None:
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


def _yt_download(link: str, song_dir: str) -> str:
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
                "preferredcodec": "wav",
                "preferredquality": 0,
            }
        ],
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        result = ydl.extract_info(link, download=True)
        download_path = ydl.prepare_filename(result, outtmpl=f"{outtmpl}.wav")

    return download_path


def _get_cached_input_paths() -> list[str]:
    # TODO if we later add .json file for input then we need to exclude those here
    return glob.glob(os.path.join(INTERMEDIATE_AUDIO_DIR, "*", "0_*_Original*"))


def _get_orig_song_path(song_dir: str) -> str | None:
    # NOTE orig_song_paths should never contain more than one element
    return next(iter(glob.glob(os.path.join(song_dir, "0_*_Original*"))), None)


def _pitch_shift(audio_path: str, output_path: str, n_semi_tones: int) -> None:
    y, sr = sf.read(audio_path)
    tfm = sox.Transformer()
    tfm.pitch(n_semi_tones)
    y_shifted = tfm.build_array(input_array=y, sample_rate_in=sr)
    sf.write(output_path, y_shifted, sr)


# TODO consider increasing size to 16
# otherwise we might have problems with hash collisions
# when using app as CLI
def _get_unique_base_path(
    song_dir: str,
    prefix: str,
    arg_dict: dict[str, Any],
    progress_bar: gr.Progress | None = None,
    percentage: float = 0.0,
    hash_size: int = 5,
) -> str:
    dict_hash = get_hash(arg_dict, size=hash_size)
    while True:
        base_path = os.path.join(song_dir, f"{prefix}_{dict_hash}")
        json_path = f"{base_path}.json"
        if os.path.exists(json_path):
            file_dict = json_load(json_path)
            if file_dict == arg_dict:
                return base_path
            display_progress("[~] Rehashing...", percentage, progress_bar)
            dict_hash = get_hash(dict_hash, size=hash_size)
        else:
            return base_path


def _voice_change(
    voice_model: str,
    vocals_path: str,
    output_path: str,
    pitch_change: int,
    f0_method: F0Method,
    index_rate: float,
    filter_radius: int,
    rms_mix_rate: float,
    protect: float,
    crepe_hop_length: int,
    output_sr: int,
) -> None:
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


def _add_audio_effects(
    audio_path: str,
    output_path: str,
    reverb_rm_size: float,
    reverb_wet: float,
    reverb_dry: float,
    reverb_damping: float,
) -> None:

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


def _map_audio_ext(input_audio_ext: InputAudioExt) -> OutputAudioExt:
    if input_audio_ext == "m4a":
        return "ipod"
    elif input_audio_ext == "aac":
        return "ipod"
    return input_audio_ext


def _mix_audio(
    audio_paths: list[str],
    output_path: str,
    main_gain: int,
    backup_gain: int,
    inst_gain: int,
    output_format: InputAudioExt,
    output_sr: int,
) -> None:
    main_vocal_audio = AudioSegment.from_wav(audio_paths[0]) + main_gain
    backup_vocal_audio = AudioSegment.from_wav(audio_paths[1]) + backup_gain
    instrumental_audio = AudioSegment.from_wav(audio_paths[2]) + inst_gain
    combined_audio = main_vocal_audio.overlay(backup_vocal_audio).overlay(
        instrumental_audio
    )
    combined_audio_resampled = combined_audio.set_frame_rate(output_sr)
    mapped_output_format = _map_audio_ext(output_format)
    combined_audio_resampled.export(output_path, format=mapped_output_format)


def get_named_song_dirs() -> list[tuple[str, str]]:
    input_paths = _get_cached_input_paths()
    named_song_dirs: list[tuple[str, str]] = []

    for path in input_paths:
        song_dir, song_basename = os.path.split(path)
        song_name = (
            os.path.splitext(song_basename)[0]
            .removeprefix("0_")
            .removesuffix("_Original")
        )
        named_song_dirs.append((song_name, song_dir))
    return sorted(named_song_dirs, key=lambda x: x[0])


def convert_to_stereo(
    song_path: str,
    song_dir: str,
    progress_bar: gr.Progress | None = None,
    percentage: float = 0.0,
) -> str:
    if not song_path:
        raise InputMissingError("Input song missing!")
    if not os.path.isfile(song_path):
        raise PathNotFoundError("Input song does not exist!")
    if not song_dir:
        raise InputMissingError("Song directory missing!")
    if not os.path.isdir(song_dir):
        raise PathNotFoundError("Song directory does not exist!")

    stereo_path = song_path

    song_info = pydub_utils.mediainfo(song_path)
    if song_info["channels"] == "1":
        arg_dict = {
            "input-files": [
                {"name": os.path.basename(song_path), "hash": get_file_hash(song_path)}
            ],
        }
        stereo_path_base = _get_unique_base_path(
            song_dir, "0_Stereo", arg_dict, progress_bar, percentage
        )
        stereo_path = f"{stereo_path_base}.wav"
        stereo_json_path = f"{stereo_path_base}.json"
        if not (os.path.exists(stereo_path) and os.path.exists(stereo_json_path)):
            display_progress(
                "[~] Converting song to stereo...",
                percentage,
                progress_bar,
            )
            command = shlex.split(
                f'ffmpeg -y -loglevel error -i "{song_path}" -ac 2 -f wav "{stereo_path}"'
            )
            subprocess.run(command)
            json_dump(arg_dict, stereo_json_path)

    return stereo_path


def _make_song_dir(
    song_input: str,
    progress_bar: gr.Progress | None = None,
    percentage: float = 0.0,
) -> tuple[str, InputType]:
    # if song directory
    if os.path.isdir(song_input):
        if not PurePath(song_input).parent == PurePath(INTERMEDIATE_AUDIO_DIR):
            raise InvalidPathError(
                "Song directory not located in intermediate audio root directory."
            )
        display_progress(
            "[~] Using existing song directory...", percentage, progress_bar
        )
        input_type = "local"
        return song_input, input_type

    display_progress("[~] Creating song directory...", percentage, progress_bar)
    # if youtube url
    if urlparse(song_input).scheme == "https":
        input_type = "yt"
        song_id = _get_youtube_video_id(song_input)
        if song_id is None:
            raise InvalidPathError("Invalid YouTube url!")
    # local audio file
    else:
        input_type = "local"
        # TODO can probably remove line below
        # filenames cant contain '"' on windows and on linux it should be fine to
        # song_input = song_input.strip('"')
        if os.path.isfile(song_input):
            song_id = get_file_hash(song_input)
        else:
            raise PathNotFoundError(f"File {song_input} does not exist.")

    song_dir = os.path.join(INTERMEDIATE_AUDIO_DIR, song_id)

    Path(song_dir).mkdir(parents=True, exist_ok=True)

    return song_dir, input_type


def retrieve_song(
    song_input: str,
    progress_bar: gr.Progress | None = None,
    percentages: list[float] = [i / 3 for i in range(3)],
) -> tuple[str, str]:
    if len(percentages) != 3:
        raise ValueError("Percentages must be a list of length 3.")
    if not song_input:
        raise InputMissingError(
            "Song input missing! Please provide a valid YouTube url, local audio file or cached input song."
        )

    song_dir, input_type = _make_song_dir(song_input, progress_bar, percentages[0])
    orig_song_path = _get_orig_song_path(song_dir)

    if not orig_song_path:
        if input_type == "yt":
            display_progress(
                "[~] Downloading song...",
                percentages[1],
                progress_bar,
            )
            song_link = song_input.split("&")[0]
            orig_song_path = _yt_download(song_link, song_dir)
        else:
            display_progress("[~] Copying song...", percentages[1], progress_bar)
            song_input_base = os.path.basename(song_input)
            song_input_name, song_input_ext = os.path.splitext(song_input_base)
            orig_song_name = f"0_{song_input_name}_Original"
            orig_song_path = os.path.join(song_dir, orig_song_name + song_input_ext)
            shutil.copyfile(song_input, orig_song_path)

    stereo_path = convert_to_stereo(
        orig_song_path, song_dir, progress_bar, percentages[2]
    )
    return stereo_path, song_dir


def separate_vocals(
    song_path: str,
    song_dir: str,
    stereofy: bool = True,
    progress_bar: gr.Progress | None = None,
    percentages: list[float] = [i / 2 for i in range(2)],
) -> tuple[str, str]:
    if len(percentages) != 2:
        raise ValueError("Percentages must be a list of length 2.")
    if not song_path:
        raise InputMissingError("Input song missing!")
    if not os.path.isfile(song_path):
        raise PathNotFoundError("Input song does not exist!")
    if not song_dir:
        raise InputMissingError("Song directory missing!")
    if not os.path.isdir(song_dir):
        raise PathNotFoundError("Song directory does not exist!")

    song_path = (
        convert_to_stereo(song_path, song_dir, progress_bar, percentages[0])
        if stereofy
        else song_path
    )

    arg_dict = {
        "input-files": [
            {"name": os.path.basename(song_path), "hash": get_file_hash(song_path)}
        ],
    }

    vocals_path_base = _get_unique_base_path(
        song_dir, "1_Vocals", arg_dict, progress_bar, percentages[1]
    )

    instrumentals_path_base = _get_unique_base_path(
        song_dir, "1_Instrumental", arg_dict, progress_bar, percentages[1]
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
            percentages[1],
            progress_bar,
        )
        SEPARATOR.arch_specific_params["MDX"]["segment_size"] = 512
        SEPARATOR.load_model("UVR-MDX-NET-Voc_FT.onnx")
        temp_instrumentals_name, temp_vocals_name = SEPARATOR.separate(song_path)
        shutil.move(
            os.path.join(INTERMEDIATE_AUDIO_DIR, temp_instrumentals_name),
            instrumentals_path,
        )
        shutil.move(
            os.path.join(INTERMEDIATE_AUDIO_DIR, temp_vocals_name),
            vocals_path,
        )
        json_dump(arg_dict, vocals_json_path)
        json_dump(arg_dict, instrumentals_json_path)
    return vocals_path, instrumentals_path


def separate_main_vocals(
    vocals_path: str,
    song_dir: str,
    stereofy: bool = True,
    progress_bar: gr.Progress | None = None,
    percentages: list[float] = [i / 2 for i in range(2)],
) -> tuple[str, str]:
    if len(percentages) != 2:
        raise ValueError("Percentages must be a list of length 2.")

    if not vocals_path:
        raise InputMissingError("Vocals missing!")
    if not os.path.isfile(vocals_path):
        raise PathNotFoundError("Vocals do not exist!")
    if not song_dir:
        raise InputMissingError("Song directory missing!")
    if not os.path.isdir(song_dir):
        raise PathNotFoundError("song directory does not exist!")

    vocals_path = (
        convert_to_stereo(vocals_path, song_dir, progress_bar, percentages[0])
        if stereofy
        else vocals_path
    )

    arg_dict = {
        "input-files": [
            {
                "name": os.path.basename(vocals_path),
                "hash": get_file_hash(vocals_path),
            }
        ],
    }

    main_vocals_path_base = _get_unique_base_path(
        song_dir, "2_Vocals_Main", arg_dict, progress_bar, percentages[1]
    )

    backup_vocals_path_base = _get_unique_base_path(
        song_dir, "2_Vocals_Backup", arg_dict, progress_bar, percentages[1]
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
            percentages[1],
            progress_bar,
        )
        SEPARATOR.arch_specific_params["MDX"]["segment_size"] = 512
        SEPARATOR.load_model("UVR_MDXNET_KARA_2.onnx")
        temp_main_vocals_name, temp_backup_vocals_name = SEPARATOR.separate(vocals_path)
        shutil.move(
            os.path.join(INTERMEDIATE_AUDIO_DIR, temp_main_vocals_name),
            main_vocals_path,
        )
        shutil.move(
            os.path.join(INTERMEDIATE_AUDIO_DIR, temp_backup_vocals_name),
            backup_vocals_path,
        )
        json_dump(arg_dict, main_vocals_json_path)
        json_dump(arg_dict, backup_vocals_json_path)
    return main_vocals_path, backup_vocals_path


def dereverb_vocals(
    vocals_path: str,
    song_dir: str,
    stereofy: bool = True,
    progress_bar: gr.Progress | None = None,
    percentages: list[float] = [i / 2 for i in range(2)],
) -> tuple[str, str]:
    if len(percentages) != 2:
        raise ValueError("Percentages must be a list of length 2.")

    if not vocals_path:
        raise InputMissingError("Vocals missing!")
    if not os.path.isfile(vocals_path):
        raise PathNotFoundError("Vocals do not exist!")
    if not song_dir:
        raise InputMissingError("Song directory missing!")
    if not os.path.isdir(song_dir):
        raise PathNotFoundError("song directory does not exist!")

    vocals_path = (
        convert_to_stereo(vocals_path, song_dir, progress_bar, percentages[0])
        if stereofy
        else vocals_path
    )

    arg_dict = {
        "input-files": [
            {
                "name": os.path.basename(vocals_path),
                "hash": get_file_hash(vocals_path),
            }
        ],
    }

    vocals_dereverb_path_base = _get_unique_base_path(
        song_dir, "3_Vocals_DeReverb", arg_dict, progress_bar, percentages[1]
    )
    vocals_reverb_path_base = _get_unique_base_path(
        song_dir, "3_Vocals_Reverb", arg_dict, progress_bar, percentages[1]
    )

    vocals_dereverb_path = f"{vocals_dereverb_path_base}.wav"
    vocals_dereverb_json_path = f"{vocals_dereverb_path_base}.json"

    vocals_reverb_path = f"{vocals_reverb_path_base}.wav"
    vocals_reverb_json_path = f"{vocals_reverb_path_base}.json"

    if not (
        os.path.exists(vocals_dereverb_path)
        and os.path.exists(vocals_dereverb_json_path)
        and os.path.exists(vocals_reverb_path)
        and os.path.exists(vocals_reverb_json_path)
    ):
        display_progress(
            "[~] De-reverbing vocals...",
            percentages[1],
            progress_bar,
        )
        SEPARATOR.arch_specific_params["MDX"]["segment_size"] = 256
        SEPARATOR.load_model("Reverb_HQ_By_FoxJoy.onnx")
        temp_vocals_dereverb_name, temp_vocals_reverb_name = SEPARATOR.separate(
            vocals_path
        )
        shutil.move(
            os.path.join(INTERMEDIATE_AUDIO_DIR, temp_vocals_dereverb_name),
            vocals_dereverb_path,
        )
        shutil.move(
            os.path.join(INTERMEDIATE_AUDIO_DIR, temp_vocals_reverb_name),
            vocals_reverb_path,
        )
        json_dump(arg_dict, vocals_dereverb_json_path)
        json_dump(arg_dict, vocals_reverb_json_path)
    return vocals_dereverb_path, vocals_reverb_path


def convert_vocals(
    vocals_path: str,
    song_dir: str,
    voice_model: str,
    pitch_change_octaves: int = 0,
    pitch_change_semi_tones: int = 0,
    index_rate: float = 0.5,
    filter_radius: int = 3,
    rms_mix_rate: float = 0.25,
    protect: float = 0.33,
    f0_method: F0Method = "rmvpe",
    crepe_hop_length: int = 128,
    progress_bar: gr.Progress | None = None,
    percentage: float = 0.0,
) -> str:
    if not vocals_path:
        raise InputMissingError("Vocals missing!")
    if not os.path.isfile(vocals_path):
        raise PathNotFoundError("Vocals do not exist!")
    if not song_dir:
        raise InputMissingError("Song directory missing!")
    if not os.path.isdir(song_dir):
        raise PathNotFoundError("song directory does not exist!")
    if not voice_model:
        raise InputMissingError("Voice model missing!")
    if not os.path.isdir(os.path.join(RVC_MODELS_DIR, voice_model)):
        raise PathNotFoundError("Voice model does not exist!")

    pitch_change = pitch_change_octaves * 12 + pitch_change_semi_tones
    hop_length_suffix = "" if f0_method != "mangio-crepe" else f"_{crepe_hop_length}"
    arg_dict = {
        "input-files": [
            {
                "name": os.path.basename(vocals_path),
                "hash": get_file_hash(vocals_path),
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

    converted_vocals_path_base = _get_unique_base_path(
        song_dir, "4_Vocals_Converted", arg_dict, progress_bar, percentage
    )
    converted_vocals_path = f"{converted_vocals_path_base}.wav"
    converted_vocals_json_path = f"{converted_vocals_path_base}.json"

    if not (
        os.path.exists(converted_vocals_path)
        and os.path.exists(converted_vocals_json_path)
    ):
        display_progress("[~] Converting vocals using RVC...", percentage, progress_bar)
        _voice_change(
            voice_model,
            vocals_path,
            converted_vocals_path,
            pitch_change,
            f0_method,
            index_rate,
            filter_radius,
            rms_mix_rate,
            protect,
            crepe_hop_length,
            44100,
        )
        json_dump(arg_dict, converted_vocals_json_path)
    return converted_vocals_path


def postprocess_vocals(
    vocals_path: str,
    song_dir: str,
    reverb_rm_size: float = 0.15,
    reverb_wet: float = 0.2,
    reverb_dry: float = 0.8,
    reverb_damping: float = 0.7,
    progress_bar: gr.Progress | None = None,
    percentage: float = 0.0,
) -> str:

    if not vocals_path:
        raise InputMissingError("Vocals missing!")
    if not os.path.isfile(vocals_path):
        raise PathNotFoundError("Vocals do not exist!")
    if not song_dir:
        raise InputMissingError("Song directory missing!")
    if not os.path.isdir(song_dir):
        raise PathNotFoundError("song directory does not exist!")

    arg_dict = {
        "input-files": [
            {
                "name": os.path.basename(vocals_path),
                "hash": get_file_hash(vocals_path),
            }
        ],
        "reverb-room-size": reverb_rm_size,
        "reverb-wet": reverb_wet,
        "reverb-dry": reverb_dry,
        "reverb-damping": reverb_damping,
    }

    vocals_mixed_path_base = _get_unique_base_path(
        song_dir, "5_Vocals_Postprocessed", arg_dict, progress_bar, percentage
    )

    vocals_mixed_path = f"{vocals_mixed_path_base}.wav"
    vocals_mixed_json_path = f"{vocals_mixed_path_base}.json"

    if not (
        os.path.exists(vocals_mixed_path) and os.path.exists(vocals_mixed_json_path)
    ):
        display_progress(
            "[~] Applying audio effects to vocals...",
            percentage,
            progress_bar,
        )
        _add_audio_effects(
            vocals_path,
            vocals_mixed_path,
            reverb_rm_size,
            reverb_wet,
            reverb_dry,
            reverb_damping,
        )
        json_dump(arg_dict, vocals_mixed_json_path)
    return vocals_mixed_path


def pitch_shift_background(
    instrumentals_path: str,
    backup_vocals_path: str,
    song_dir: str,
    pitch_change: int = 0,
    progress_bar: gr.Progress | None = None,
    percentages: list[float] = [i / 2 for i in range(2)],
) -> tuple[str, str]:
    if len(percentages) != 2:
        raise ValueError("Percentages must be a list of length 2.")
    if not instrumentals_path:
        raise InputMissingError("Instrumentals missing!")
    if not os.path.isfile(instrumentals_path):
        raise PathNotFoundError("Instrumentals do not exist!")
    if not backup_vocals_path:
        raise InputMissingError("Backup vocals missing!")
    if not os.path.isfile(backup_vocals_path):
        raise PathNotFoundError("Backup vocals do not exist!")
    if not song_dir:
        raise InputMissingError("Song directory missing!")
    if not os.path.isdir(song_dir):
        raise PathNotFoundError("song directory does not exist!")

    instrumentals_shifted_path = instrumentals_path
    backup_vocals_shifted_path = backup_vocals_path

    if pitch_change != 0:
        instrumentals_dict = {
            "input-files": [
                {
                    "name": os.path.basename(instrumentals_path),
                    "hash": get_file_hash(instrumentals_path),
                }
            ],
            "pitch-shift": pitch_change,
        }

        instrumentals_shifted_path_base = _get_unique_base_path(
            song_dir,
            "6_Instrumental_Shifted",
            instrumentals_dict,
            progress_bar,
            percentages[0],
        )

        instrumentals_shifted_path = f"{instrumentals_shifted_path_base}.wav"
        instrumentals_shifted_json_path = f"{instrumentals_shifted_path_base}.json"

        if not (
            os.path.exists(instrumentals_shifted_path)
            and os.path.exists(instrumentals_shifted_json_path)
        ):
            display_progress(
                "[~] Applying pitch shift to instrumentals",
                percentages[0],
                progress_bar,
            )
            _pitch_shift(
                instrumentals_path,
                instrumentals_shifted_path,
                pitch_change,
            )
            json_dump(instrumentals_dict, instrumentals_shifted_json_path)

        backup_vocals_dict = {
            "input-files": [
                {
                    "name": os.path.basename(backup_vocals_path),
                    "hash": get_file_hash(backup_vocals_path),
                }
            ],
            "pitch-shift": pitch_change,
        }

        backup_vocals_shifted_path_base = _get_unique_base_path(
            song_dir,
            "6_Vocals_Backup_Shifted",
            backup_vocals_dict,
            progress_bar,
            percentages[1],
        )
        backup_vocals_shifted_path = f"{backup_vocals_shifted_path_base}.wav"
        backup_vocals_shifted_json_path = f"{backup_vocals_shifted_path_base}.json"
        if not (
            os.path.exists(backup_vocals_shifted_path)
            and os.path.exists(backup_vocals_shifted_json_path)
        ):
            display_progress(
                "[~] Applying pitch shift to backup vocals",
                percentages[1],
                progress_bar,
            )
            _pitch_shift(
                backup_vocals_path,
                backup_vocals_shifted_path,
                pitch_change,
            )
            json_dump(backup_vocals_dict, backup_vocals_shifted_json_path)
    return instrumentals_shifted_path, backup_vocals_shifted_path


def _get_voice_model(
    mixed_vocals_path: str | None = None, song_dir: str | None = None
) -> str:
    voice_model = "Unknown"
    if not (mixed_vocals_path and song_dir):
        return voice_model
    mixed_vocals_stem = get_path_stem(mixed_vocals_path)
    mixed_vocals_json_path = os.path.join(song_dir, f"{mixed_vocals_stem}.json")
    if not os.path.isfile(mixed_vocals_json_path):
        return voice_model
    mixed_vocals_json_dict = json_load(mixed_vocals_json_path)
    input_files = mixed_vocals_json_dict.get("input-files")
    input_path = input_files[0].get("name") if input_files else None
    if not input_path:
        return voice_model
    input_stem = get_path_stem(input_path)
    converted_vocals_json_path = os.path.join(song_dir, f"{input_stem}.json")
    if not os.path.isfile(converted_vocals_json_path):
        return voice_model
    converted_vocals_dict = json_load(converted_vocals_json_path)
    return converted_vocals_dict.get("voice-model", voice_model)


def get_song_cover_name(
    mixed_vocals_path: str | None = None,
    song_dir: str | None = None,
    voice_model: str | None = None,
    progress_bar: gr.Progress | None = None,
    percentage: float = 0.0,
) -> str:
    display_progress("[~] Getting song cover name...", percentage, progress_bar)

    orig_song_path = _get_orig_song_path(song_dir) if song_dir else None
    orig_song_name = (
        (get_path_stem(orig_song_path).removeprefix("0_").removesuffix("_Original"))
        if orig_song_path
        else "Unknown"
    )

    voice_model = voice_model or _get_voice_model(mixed_vocals_path, song_dir)

    return f"{orig_song_name} ({voice_model} Ver)"


def mix_song_cover(
    main_vocals_path: str,
    instrumentals_path: str,
    backup_vocals_path: str,
    song_dir: str,
    main_gain: int = 0,
    inst_gain: int = 0,
    backup_gain: int = 0,
    output_sr: int = 44100,
    output_format: InputAudioExt = "mp3",
    output_name: str | None = None,
    progress_bar: gr.Progress | None = None,
    percentages: list[float] = [i / 2 for i in range(2)],
) -> str:
    if len(percentages) != 2:
        raise ValueError("Percentages must be a list of length 2.")
    if not main_vocals_path:
        raise InputMissingError("Main vocals missing!")
    if not os.path.isfile(main_vocals_path):
        raise PathNotFoundError("Main vocals do not exist!")
    if not instrumentals_path:
        raise InputMissingError("Instrumentals missing!")
    if not os.path.isfile(instrumentals_path):
        raise PathNotFoundError("Instrumentals do not exist!")
    if not backup_vocals_path:
        raise InputMissingError("Backup vocals missing!")
    if not os.path.isfile(backup_vocals_path):
        raise PathNotFoundError("Backup vocals do not exist!")
    if not song_dir:
        raise InputMissingError("Song directory missing!")
    if not os.path.isdir(song_dir):
        raise PathNotFoundError("song directory does not exist!")

    arg_dict = {
        "input-files": [
            {
                "name": os.path.basename(main_vocals_path),
                "hash": get_file_hash(main_vocals_path),
            },
            {
                "name": os.path.basename(instrumentals_path),
                "hash": get_file_hash(instrumentals_path),
            },
            {
                "name": os.path.basename(backup_vocals_path),
                "hash": get_file_hash(backup_vocals_path),
            },
        ],
        "main-gain": main_gain,
        "instrument-gain": inst_gain,
        "backup-gain": backup_gain,
        "sample-rate": output_sr,
    }

    mixdown_path_base = _get_unique_base_path(
        song_dir, "7_Mixdown", arg_dict, progress_bar, percentages[0]
    )
    mixdown_path = f"{mixdown_path_base}.{output_format}"
    mixdown_json_path = f"{mixdown_path_base}.json"

    if not (os.path.exists(mixdown_path) and os.path.exists(mixdown_json_path)):
        display_progress(
            "[~] Mixing main vocals, instrumentals, and backup vocals...",
            percentages[0],
            progress_bar,
        )

        _mix_audio(
            [
                main_vocals_path,
                backup_vocals_path,
                instrumentals_path,
            ],
            mixdown_path,
            main_gain,
            backup_gain,
            inst_gain,
            output_format,
            output_sr,
        )
        json_dump(arg_dict, mixdown_json_path)

    output_name = output_name or get_song_cover_name(
        main_vocals_path, song_dir, None, progress_bar, percentages[1]
    )
    song_cover_path = os.path.join(OUTPUT_AUDIO_DIR, f"{output_name}.{output_format}")
    os.makedirs(OUTPUT_AUDIO_DIR, exist_ok=True)
    shutil.copyfile(mixdown_path, song_cover_path)

    return song_cover_path


def run_pipeline(
    song_input: str,
    voice_model: str,
    pitch_change_vocals: int = 0,
    pitch_change_all: int = 0,
    index_rate: float = 0.5,
    filter_radius: int = 3,
    rms_mix_rate: float = 0.25,
    protect: float = 0.33,
    f0_method: F0Method = "rmvpe",
    crepe_hop_length: int = 128,
    reverb_rm_size: float = 0.15,
    reverb_wet: float = 0.2,
    reverb_dry: float = 0.8,
    reverb_damping: float = 0.7,
    main_gain: int = 0,
    inst_gain: int = 0,
    backup_gain: int = 0,
    output_sr: int = 44100,
    output_format: InputAudioExt = "mp3",
    output_name: str | None = None,
    return_files: bool = False,
    progress_bar: gr.Progress | None = None,
) -> str | tuple[str, ...]:
    display_progress("[~] Starting song cover generation pipeline...", 0, progress_bar)
    percentages = [i / 15 for i in range(15)]
    orig_song_path, song_dir = retrieve_song(song_input, progress_bar, percentages[:3])
    vocals_path, instrumentals_path = separate_vocals(
        orig_song_path, song_dir, False, progress_bar, percentages[3:5]
    )
    main_vocals_path, backup_vocals_path = separate_main_vocals(
        vocals_path, song_dir, False, progress_bar, percentages[5:7]
    )
    vocals_dereverb_path, reverb_path = dereverb_vocals(
        main_vocals_path, song_dir, False, progress_bar, percentages[7:9]
    )
    converted_vocals_path = convert_vocals(
        vocals_dereverb_path,
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
        progress_bar,
        percentages[9],
    )
    vocals_mixed_path = postprocess_vocals(
        converted_vocals_path,
        song_dir,
        reverb_rm_size,
        reverb_wet,
        reverb_dry,
        reverb_damping,
        progress_bar,
        percentages[10],
    )
    instrumentals_shifted_path, backup_vocals_shifted_path = pitch_shift_background(
        instrumentals_path,
        backup_vocals_path,
        song_dir,
        pitch_change_all,
        progress_bar,
        percentages[11:13],
    )

    song_cover_path = mix_song_cover(
        vocals_mixed_path,
        instrumentals_shifted_path or instrumentals_path,
        backup_vocals_shifted_path or backup_vocals_path,
        song_dir,
        main_gain,
        inst_gain,
        backup_gain,
        output_sr,
        output_format,
        output_name,
        progress_bar,
        percentages[13:15],
    )
    if return_files:
        return (
            orig_song_path,
            vocals_path,
            instrumentals_path,
            main_vocals_path,
            backup_vocals_path,
            vocals_dereverb_path,
            reverb_path,
            converted_vocals_path,
            vocals_mixed_path,
            instrumentals_shifted_path,
            backup_vocals_shifted_path,
            song_cover_path,
        )
    else:
        return song_cover_path
