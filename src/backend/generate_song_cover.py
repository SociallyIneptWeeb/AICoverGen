"""
This module contains functions to generate song covers using RVC-based voice models.
"""

from typing import Any
from typings.extra import F0Method, InputAudioExt, InputType, OutputAudioExt

import gc
import glob
import os
import shlex
import shutil
import subprocess
from contextlib import suppress
from logging import WARNING
from pathlib import Path, PurePath
from urllib.parse import parse_qs, urlparse

import yt_dlp

import gradio as gr

import soundfile as sf
import sox
from audio_separator.separator import Separator
from pedalboard import Compressor, HighpassFilter, Reverb
from pedalboard._pedalboard import Pedalboard
from pedalboard.io import AudioFile
from pydub import AudioSegment
from pydub import utils as pydub_utils

from vc.rvc import Config, get_vc, load_hubert, rvc_infer

from backend.common import (
    INTERMEDIATE_AUDIO_DIR,
    OUTPUT_AUDIO_DIR,
    display_progress,
    get_file_hash,
    get_hash,
    get_path_stem,
    get_rvc_model,
    json_dump,
    json_load,
)
from backend.exceptions import InputMissingError, InvalidPathError, PathNotFoundError

from common import RVC_MODELS_DIR, SEPARATOR_MODELS_DIR

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
    mdxc_params={"segment_size": 256, "batch_size": 1, "overlap": 2},
)


def _get_youtube_video_id(url: str, ignore_playlist: bool = True) -> str | None:
    """
    Get video id from a YouTube URL.

    Parameters
    ----------
    url : str
        The YouTube URL.
    ignore_playlist : bool, default=True
        Whether to get id of first video in playlist or the playlist id itself.

    Returns
    -------
    str
        The video id.
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
    return None


def _yt_download(link: str, song_dir: str) -> str:
    """
    Download audio from a YouTube link.

    Parameters
    ----------
    link : str
        The YouTube link.
    song_dir : str
        The directory to save the downloaded audio to.

    Returns
    -------
    str
        The path to the downloaded audio file.
    """
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
        if not result:
            raise PathNotFoundError("No audio found in the provided YouTube link!")
        download_path = ydl.prepare_filename(result, outtmpl=f"{outtmpl}.wav")

    return download_path


def _get_input_audio_paths() -> list[str]:
    """
    Get the paths of all cached input audio files.

    Returns
    -------
    list[str]
        The paths of all cached input audio files
    """
    # TODO if we later add .json file for input then we need to exclude those here
    return glob.glob(os.path.join(INTERMEDIATE_AUDIO_DIR, "*", "0_*_Original*"))


def _get_input_audio_path(song_dir: str) -> str | None:
    """
    Get the path of the cached input audio file in a given song directory.

    Parameters
    ----------
    song_dir : str
        The path to a song directory.

    Returns
    -------
    str
        The path of the cached input audio file, if it exists.
    """
    # NOTE orig_song_paths should never contain more than one element
    return next(iter(glob.glob(os.path.join(song_dir, "0_*_Original*"))), None)


def _pitch_shift(audio_path: str, output_path: str, n_semi_tones: int) -> None:
    """
    Pitch-shift an audio file.

    Parameters
    ----------
    audio_path : str
        The path of the audio file to pitch-shift.
    output_path : str
        The path to save the pitch-shifted audio file to.
    n_semi_tones : int
        The number of semi-tones to pitch-shift the audio by.
    """
    y, sr = sf.read(audio_path)
    tfm = sox.Transformer()
    tfm.pitch(n_semi_tones)
    y_shifted = tfm.build_array(input_array=y, sample_rate_in=sr)
    sf.write(output_path, y_shifted, sr)


# TODO consider increasing hash_size to 16
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
    """
    Get a unique base path for an audio file in a song directory
    by hashing the arguments used to generate the audio.

    Parameters
    ----------
    song_dir : str
        The path to a song directory.
    prefix : str
        The prefix to use for the base path.
    arg_dict : dict
        The dictionary of arguments used to generate the audio in the given file.
    progress_bar : gr.Progress, optional
        Gradio progress bar to update.
    percentage : float, default=0.0
        Percentage to display in the progress bar.
    hash_size : int, default=5
        The size (in bytes) of the hash to use for the base path.

    Returns
    -------
    str
        The unique base path for the audio file.
    """
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


def _convert_voice(
    voice_model: str,
    voice_path: str,
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
    """
    Convert a voice track using a voice model.

    Parameters
    ----------
    voice_model : str
        The name of the voice model to use.
    voice_path : str
        The path to the voice track to convert.
    output_path : str
        The path to save the converted voice to.
    pitch_change : int
        The number of semi-tones to pitch-shift the converted voice by.
    f0_method : F0Method
        The method to use for pitch extraction.
    index_rate : float
        The influence of index file on voice conversion.
    filter_radius : int
        The filter radius to use for the voice conversion.
    rms_mix_rate : float
        The blending rate of the volume envelope of converted voice.
    protect : float
        The protection rate for consonants and breathing sounds.
    crepe_hop_length : int
        The hop length to use for Crepe pitch extraction method.
    output_sr : int
        The sample rate to use for the output audio.
    """
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
        voice_path,
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
    """
    Add high-pass filter, compressor and reverb effects to an audio file.

    Parameters
    ----------
    audio_path : str
        The path of the audio file to add effects to.
    output_path : str
        The path to save the effected audio file to.
    reverb_rm_size : float
        The room size of the reverb effect.
    reverb_wet : float
        The wet level of the reverb effect.
    reverb_dry : float
        The dry level of the reverb effect.
    reverb_damping : float
        The damping of the reverb effect.
    """
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
    """
    Map an input audio extension to an output audio extension.

    Parameters
    ----------
    input_audio_ext : InputAudioExt
        The input audio extension.

    Returns
    -------
    OutputAudioExt
        The output audio extension.
    """
    match input_audio_ext:
        case "m4a":
            return "ipod"
        case "aac":
            return "adts"
        case _:
            return input_audio_ext


def _mix_audio(
    main_vocal_path: str,
    backup_vocal_path: str,
    instrumental_path: str,
    main_gain: int,
    backup_gain: int,
    inst_gain: int,
    output_format: InputAudioExt,
    output_sr: int,
    output_path: str,
) -> None:
    """
    Mix main vocals, backup vocals and instrumentals.

    Parameters
    ----------
    main_vocal_path : str
        The path of an audio file containing main vocals.
    backup_vocal_path : str
        The path of an audio file containing backup vocals.
    instrumental_path : str
        The path of an audio file containing instrumentals.
    main_gain : int
        The gain to apply to the main vocals.
    backup_gain : int
        The gain to apply to the backup vocals.
    inst_gain : int
        The gain to apply to the instrumental.
    output_format : InputAudioExt
        The format to save the mixed audio file in.
    output_sr : int
        The sample rate to use for the mixed audio file.
    output_path : str
        The path to save the mixed audio file to.
    """
    main_vocal_audio = AudioSegment.from_wav(main_vocal_path) + main_gain
    backup_vocal_audio = AudioSegment.from_wav(backup_vocal_path) + backup_gain
    instrumental_audio = AudioSegment.from_wav(instrumental_path) + inst_gain
    combined_audio = main_vocal_audio.overlay(backup_vocal_audio).overlay(
        instrumental_audio
    )
    combined_audio_resampled = combined_audio.set_frame_rate(output_sr)
    mapped_output_format = _map_audio_ext(output_format)
    combined_audio_resampled.export(output_path, format=mapped_output_format)


def get_named_song_dirs() -> list[tuple[str, str]]:
    """
    Get the names and paths of all song directories.

    Returns
    -------
    list[tuple[str, str]]
        A list of tuples containing the name and path of each song directory.
    """
    input_paths = _get_input_audio_paths()
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
    """
    Converts an audio file to stereo.

    Parameters
    ----------
    song_path : str
        The path to the audio file to convert.
    song_dir : str
        The path to the directory where the stereo audio file will be saved.
    progress_bar : gr.Progress, optional
        Gradio progress bar to update.
    percentage : float, default=0.0
        Percentage to display in the progress bar.

    Returns
    -------
    str
        The path to the stereo audio file.

    Raises
    ------
    InputMissingError
        If no audio file or song directory path is provided.
    PathNotFoundError
        If the provided audio file or song directory path does not point
        to an existing file or directory.
    """
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
                "[~] Converting song to stereo...", percentage, progress_bar
            )
            command = shlex.split(
                f'ffmpeg -y -loglevel error -i "{song_path}" -ac 2 -f wav'
                f' "{stereo_path}"'
            )
            subprocess.run(command)
            json_dump(arg_dict, stereo_json_path)

    return stereo_path


def _make_song_dir(
    song_input: str, progress_bar: gr.Progress | None = None, percentage: float = 0.0
) -> tuple[str, InputType]:
    """
    Create a song directory for a given song input.

    * If the song input is a YouTube URL,
    the song directory will be named after the video id.
    * If the song input is a local audio file,
    the song directory will be named after the file hash.
    * if the song input is a song directory,
    the song directory will be used as is.

    Parameters
    ----------
    song_input : str
        The song input to create a directory for.
    progress_bar : gr.Progress, optional
        Gradio progress bar to update.
    percentage : float, default=0.0
        Percentage to display in the progress bar.

    Returns
    -------
    song_dir : str
        The path to the created song directory.
    input_type : InputType
        The type of input provided.

    Raises
    ------
    InputMissingError
        If no song input is provided.
    InvalidPathError
        If the provided YouTube URL is invalid or if the provided song directory
        is not located in the root of the intermediate audio directory.
    PathNotFoundError
        If the provided song input is neither a valid HTTPS-based URL
        nor the path of an existing song directory or audio file.
    """
    # if song directory
    if os.path.isdir(song_input):
        if not PurePath(song_input).parent == PurePath(INTERMEDIATE_AUDIO_DIR):
            raise InvalidPathError(
                "Song directory not located in the root of the intermediate audio"
                " directory."
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
    # if local audio file
    elif os.path.isfile(song_input):
        input_type = "local"
        song_id = get_file_hash(song_input)
    else:
        raise PathNotFoundError(f"Song input {song_input} does not exist.")

    song_dir = os.path.join(INTERMEDIATE_AUDIO_DIR, song_id)

    Path(song_dir).mkdir(parents=True, exist_ok=True)

    return song_dir, input_type


def retrieve_song(
    song_input: str,
    progress_bar: gr.Progress | None = None,
    percentages: tuple[float, float, float] = (0, 0.33, 0.67),
) -> tuple[str, str]:
    """
    Retrieve a song from a YouTube URL, local audio file or a song directory.

    Parameters
    ----------
    song_input : str
        A Youtube URL, the path of a local audio file
        or the path of a song directory.
    progress_bar : gr.Progress, optional
        Gradio progress bar to update.
    percentages : tuple[float,float,float], default=(0, 0.33, 0.67)
        Percentages to display in the progress bar.

    Returns
    -------
    song_path : str
        The path to the retrieved audio file
    song_dir : str
        The path to the song directory containing it.

    Raises
    ------
    InputMissingError
        If no song input is provided.
    InvalidPathError
        If the provided Youtube URL is invalid or if the provided song directory
        is not located in the root of the intermediate audio directory.
    PathNotFoundError
        If the provided song input is neither a valid HTTPS-based URL
        nor the path of an existing song directory or audio file.
    """
    if not song_input:
        raise InputMissingError(
            "Song input missing! Please provide a valid YouTube url, local audio file"
            " path or cached song directory path."
        )

    song_dir, input_type = _make_song_dir(song_input, progress_bar, percentages[0])
    orig_song_path = _get_input_audio_path(song_dir)

    if not orig_song_path:
        if input_type == "yt":
            display_progress("[~] Downloading song...", percentages[1], progress_bar)
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
    percentages: tuple[float, float] = (0.0, 0.5),
) -> tuple[str, str]:
    """
    Separate a song into vocals and instrumentals.

    Parameters
    ----------
    song_path : str
        The path to the song to separate.
    song_dir : str
        The path to the song directory where the
        separated vocals and instrumentals will be saved.
    stereofy : bool, default=True
        Whether to convert the song to stereo
        before separating its vocals and instrumentals.
    progress_bar : gr.Progress, optional
        Gradio progress bar to update.
    percentages : tuple[float,float], default=(0.0, 0.5)
        Percentages to display in the progress bar.

    Returns
    -------
    vocals_path : str
        The path to the separated vocals.
    instrumentals_path : str
        The path to the separated instrumentals.

    Raises
    ------
    InputMissingError
        If no song path or song directory path is provided.
    PathNotFoundError
        If the provided song path or song directory path does not point
        to an existing file or directory.
    """
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
            "[~] Separating vocals from instrumentals...", percentages[1], progress_bar
        )
        SEPARATOR.arch_specific_params["MDX"]["segment_size"] = 512
        SEPARATOR.load_model("UVR-MDX-NET-Voc_FT.onnx")
        temp_instrumentals_name, temp_vocals_name = SEPARATOR.separate(song_path)
        shutil.move(
            os.path.join(INTERMEDIATE_AUDIO_DIR, temp_instrumentals_name),
            instrumentals_path,
        )
        shutil.move(os.path.join(INTERMEDIATE_AUDIO_DIR, temp_vocals_name), vocals_path)
        json_dump(arg_dict, vocals_json_path)
        json_dump(arg_dict, instrumentals_json_path)
    return vocals_path, instrumentals_path


def separate_main_vocals(
    vocals_path: str,
    song_dir: str,
    stereofy: bool = True,
    progress_bar: gr.Progress | None = None,
    percentages: tuple[float, float] = (0.0, 0.5),
) -> tuple[str, str]:
    """
    Separate a vocals track into main vocals and backup vocals.

    Parameters
    ----------
    vocals_path : str
        The path to the vocals track to separate.
    song_dir : str
        The path to the directory where the separated main vocals
        and backup vocals will be saved.
    stereofy : bool, default=True
        Whether to convert the vocals track to stereo
        before separating its main vocals and backup vocals.
    progress_bar : gr.Progress, optional
        Gradio progress bar to update.
    percentages : tuple[float,float], default=(0.0, 0.5)
        Percentages to display in the progress bar.

    Returns
    -------
    main_vocals_path : str
        The path to the separated main vocals.
    backup_vocals_path : str
        The path to the separated backup vocals.

    Raises
    ------
    InputMissingError
        If no vocals track path or song directory path is provided.
    PathNotFoundError
        If the provided vocals path or song directory path does not point
        to an existing file or directory.
    """
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
            {"name": os.path.basename(vocals_path), "hash": get_file_hash(vocals_path)}
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
    percentages: tuple[float, float] = (0.0, 0.5),
) -> tuple[str, str]:
    """
    De-reverb a vocals track.

    Parameters
    ----------
    vocals_path : str
        The path to the vocals track to de-reverb.
    song_dir : str
        The path to the directory where the de-reverbed vocals will be saved.
    stereofy : bool, default=True
        Whether to convert the vocals track to stereo before de-reverbing it.
    progress_bar : gr.Progress, optional
        Gradio progress bar to update.
    percentages : tuple[float,float], default=(0.0, 0.5)
        Percentages to display in the progress bar.

    Returns
    -------
    vocals_dereverb_path : str
        The path to the de-reverbed vocals.
    vocals_reverb_path : str
        The path to the reverb of the vocals.

    Raises
    ------
    InputMissingError
        If no vocals track path or song directory path is provided.
    PathNotFoundError
        If the provided vocals path or song directory path does not point
        to an existing file or directory.
    """
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
            {"name": os.path.basename(vocals_path), "hash": get_file_hash(vocals_path)}
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
        display_progress("[~] De-reverbing vocals...", percentages[1], progress_bar)
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
    """
    Convert a vocals track using a voice model.

    Parameters
    ----------
    vocals_path : str
        The path to the vocals track to convert.
    song_dir : str
        The path to the directory where the converted vocals will be saved.
    voice_model : str
        The name of the voice model to use.
    pitch_change_octaves : int, default=0
        The number of octaves to pitch-shift the converted vocals by.
    pitch_change_semi_tones : int, default=0
        The number of semi-tones to pitch-shift the converted vocals by.
    index_rate : float, default=0.5
        The influence of the index file on the vocal conversion.
    filter_radius : int, default=3
        The filter radius to use for the vocal conversion.
    rms_mix_rate : float, default=0.25
        The blending rate of the volume envelope of the converted vocals.
    protect : float, default=0.33
        The protection rate for consonants and breathing sounds.
    f0_method : F0Method, default="rmvpe"
        The method to use for pitch extraction.
    crepe_hop_length : int, default=128
        The hop length to use for crepe-based pitch extraction.
    progress_bar : gr.Progress, optional
        Gradio progress bar to update.
    percentage : float, default=0.0
        Percentage to display in the progress bar.

    Returns
    -------
    str
        The path to the converted vocals.

    Raises
    ------
    InputMissingError
        If no vocals track path, song directory path or voice model name is provided.
    PathNotFoundError
        If the provided vocals path, song directory path or voice model name
        does not point to an existing file or directory.
    """
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
            {"name": os.path.basename(vocals_path), "hash": get_file_hash(vocals_path)}
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
        _convert_voice(
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
    """
    Apply high-pass filter, compressor and reverb effects to a vocals track.

    Parameters
    ----------
    vocals_path : str
        The path to the vocals track to add effects to.
    song_dir : str
        The path to the directory where the effected vocals will be saved.
    reverb_rm_size : float, default=0.15
        The room size of the reverb effect.
    reverb_wet : float, default=0.2
        The wet level of the reverb effect.
    reverb_dry : float, default=0.8
        The dry level of the reverb effect.
    reverb_damping : float, default=0.7
        The damping of the reverb effect.
    progress_bar : gr.Progress, optional
        Gradio progress bar to update.
    percentage : float, default=0.0
        Percentage to display in the progress bar.

    Returns
    -------
    str
        The path to the effected vocals.

    Raises
    ------
    InputMissingError
        If no vocals track path or song directory path is provided.
    PathNotFoundError
        If the provided vocals path or song directory path does not point
        to an existing file or directory.
    """
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
            {"name": os.path.basename(vocals_path), "hash": get_file_hash(vocals_path)}
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
            "[~] Applying audio effects to vocals...", percentage, progress_bar
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
    percentages: tuple[float, float] = (0.0, 0.5),
) -> tuple[str, str]:
    """
    Pitch shift instrumentals and backup vocals by a given number of semi-tones.

    Parameters
    ----------
    instrumentals_path : str
        The path to the instrumentals to pitch shift.
    backup_vocals_path : str
        The path to the backup vocals to pitch shift.
    song_dir : str
        The path to the directory where the pitch-shifted instrumentals
        and backup vocals will be saved.
    pitch_change : int, default=0
        The number of semi-tones to pitch-shift the instrumentals
        and backup vocals by.
    progress_bar : gr.Progress, optional
        Gradio progress bar to update.
    percentages : tuple[float,float], default=(0.0, 0.5)
        Percentages to display in the progress bar.

    Returns
    -------
    instrumentals_shifted_path : str
        The path to the pitch-shifted instrumentals.
    backup_vocals_shifted_path : str
        The path to the pitch-shifted backup vocals.

    Raises
    ------
    InputMissingError
        If no instrumentals path, backup vocals path or song directory path is provided.
    PathNotFoundError
        If the provided instrumentals path, backup vocals path or song directory path
        does not point to an existing file or directory.
    """
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
            _pitch_shift(instrumentals_path, instrumentals_shifted_path, pitch_change)
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
            _pitch_shift(backup_vocals_path, backup_vocals_shifted_path, pitch_change)
            json_dump(backup_vocals_dict, backup_vocals_shifted_json_path)
    return instrumentals_shifted_path, backup_vocals_shifted_path


def _get_voice_model(
    mixed_vocals_path: str | None = None, song_dir: str | None = None
) -> str:
    """
    Infer the voice model used for vocal conversion from a
    mixed vocals file in a given song directory.

    If the voice model cannot be inferred, "Unknown" is returned.

    Parameters
    ----------
    mixed_vocals_path : str, optional
        The path to a mixed vocals file.
    song_dir : str, optional
        The path to a song directory.

    Returns
    -------
    str
        The voice model used for vocal conversion.
    """
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
    """
    Generates a suitable name for a cover of a song based on that song's
    original name and the voice model used for vocal conversion.

    If the path of an existing song directory is provided, the original song
    name is inferred from that directory. If a voice model is not provided but
    the path of an existing song directory and the path of a mixed
    vocals file in that directory are provided, then the voice model is
    inferred from the mixed vocals file.

    Parameters
    ----------
    mixed_vocals_path : str, optional
        The path to a mixed vocals file.
    song_dir : str, optional
        The path to a song directory.
    voice_model : str, optional
        A voice model name.
    progress_bar : gr.Progress, optional
        Gradio progress bar to update.
    percentage : float, default=0.0
        Percentage to display in the progress bar.

    Returns
    -------
    str
        The song cover name
    """
    display_progress("[~] Getting song cover name...", percentage, progress_bar)

    orig_song_path = _get_input_audio_path(song_dir) if song_dir else None
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
    percentages: tuple[float, float] = (0.0, 0.5),
) -> str:
    """
    Mix main vocals, instrumentals, and backup vocals to create a song cover.

    Parameters
    ----------
    main_vocals_path : str
        The path to the main vocals to mix.
    instrumentals_path : str
        The path to the instrumentals to mix.
    backup_vocals_path : str
        The path to the backup vocals to mix.
    song_dir : str
        The path to the song directory where the song cover will be saved.
    main_gain : int, default=0
        The gain to apply to the main vocals.
    inst_gain : int, default=0
        The gain to apply to the instrumentals.
    backup_gain : int, default=0
        The gain to apply to the backup vocals.
    output_sr : int, default=44100
        The sample rate of the song cover.
    output_format : InputAudioExt, default="mp3"
        The audio format of the song cover.
    output_name : str, optional
        The name of the song cover.
    progress_bar : gr.Progress, optional
        Gradio progress bar to update.
    percentages : tuple[float,float], default=(0.0, 0.5)
        Percentages to display in the progress bar.

    Returns
    -------
    str
        The path to the song cover.

    Raises
    ------
    InputMissingError
        If no main vocals, instrumentals, backup vocals or song directory path is provided.
    PathNotFoundError
        If the provided main vocals, instrumentals, backup vocals or song directory path
        does not point to an existing file or directory.
    """
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
            main_vocals_path,
            backup_vocals_path,
            instrumentals_path,
            main_gain,
            backup_gain,
            inst_gain,
            output_format,
            output_sr,
            mixdown_path,
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
    """
    Run the song cover generation pipeline.

    Parameters
    ----------
    song_input : str
        A Youtube URL, the path of a local audio file or the path of a song directory.
    voice_model : str
        The name of the voice model to use for vocal conversion.
    pitch_change_vocals : int, default=0
        The number of octaves to pitch-shift the converted vocals by.
    pitch_change_all : int, default=0
        The number of semi-tones to pitch-shift the converted vocals,
        instrumentals, and backup vocals by.
    index_rate : float, default=0.5
        The influence of the index file on the vocal conversion.
    filter_radius : int, default=3
        The filter radius to use for the vocal conversion.
    rms_mix_rate : float, default=0.25
        The blending rate of the volume envelope of the converted vocals.
    protect : float, default=0.33
        The protection rate for consonants and breathing sounds in the vocal conversion.
    f0_method : F0Method, default="rmvpe"
        The method to use for pitch extraction in the vocal conversion.
    crepe_hop_length : int, default=128
        The hop length to use for crepe-based pitch extraction.
    reverb_rm_size : float, default=0.15
        The room size of the reverb effect to apply to the converted vocals.
    reverb_wet : float, default=0.2
        The wet level of the reverb effect to apply to the converted vocals.
    reverb_dry : float, default=0.8
        The dry level of the reverb effect to apply to the converted vocals.
    reverb_damping : float, default=0.7
        The damping of the reverb effect to apply to the converted vocals.
    main_gain : int, default=0
        The gain to apply to the post-processed vocals.
    inst_gain : int, default=0
        The gain to apply to the pitch-shifted instrumentals.
    backup_gain : int, default=0
        The gain to apply to the pitch-shifted backup vocals.
    output_sr : int, default=44100
        The sample rate of the song cover.
    output_format : InputAudioExt, default="mp3"
        The audio format of the song cover.
    output_name : str, optional
        The name of the song cover.
    return_files : bool, default=False
        Whether to return the paths of the generated intermediate audio files.
    progress_bar : gr.Progress, optional
        Gradio progress bar to update.

    Returns
    -------
    str | tuple[str,...]
        The path to the generated song cover and, if `return_files=True`,
        also the paths of any generated intermediate audio files.
    """
    if not song_input:
        raise InputMissingError(
            "Song input missing! Please provide a valid YouTube url, local audio file"
            " path or cached song directory path."
        )
    if not voice_model:
        raise InputMissingError("Voice model missing!")
    if not os.path.isdir(os.path.join(RVC_MODELS_DIR, voice_model)):
        raise PathNotFoundError("Voice model does not exist!")
    display_progress("[~] Starting song cover generation pipeline...", 0, progress_bar)
    orig_song_path, song_dir = retrieve_song(
        song_input, progress_bar, (0 / 15, 1 / 15, 2 / 15)
    )
    vocals_path, instrumentals_path = separate_vocals(
        orig_song_path, song_dir, False, progress_bar, (3 / 15, 4 / 15)
    )
    main_vocals_path, backup_vocals_path = separate_main_vocals(
        vocals_path, song_dir, False, progress_bar, (5 / 15, 6 / 15)
    )
    vocals_dereverb_path, reverb_path = dereverb_vocals(
        main_vocals_path, song_dir, False, progress_bar, (7 / 15, 8 / 15)
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
        9 / 15,
    )
    vocals_mixed_path = postprocess_vocals(
        converted_vocals_path,
        song_dir,
        reverb_rm_size,
        reverb_wet,
        reverb_dry,
        reverb_damping,
        progress_bar,
        10 / 15,
    )
    instrumentals_shifted_path, backup_vocals_shifted_path = pitch_shift_background(
        instrumentals_path,
        backup_vocals_path,
        song_dir,
        pitch_change_all,
        progress_bar,
        (11 / 15, 12 / 15),
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
        (13 / 15, 14 / 15),
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
