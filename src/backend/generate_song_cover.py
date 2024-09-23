"""
Module which defines functions that faciliatate song cover generation
using RVC.
"""

import gc
import operator
import shutil
from collections.abc import Sequence
from contextlib import suppress
from itertools import starmap
from logging import WARNING
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from pydantic import ValidationError

import yt_dlp

import gradio as gr

import ffmpeg
import soundfile as sf
import sox
from audio_separator.separator import Separator
from pedalboard import Compressor, HighpassFilter, Reverb
from pedalboard._pedalboard import Pedalboard  # noqa: PLC2701
from pedalboard.io import AudioFile
from pydub import AudioSegment
from pydub import utils as pydub_utils

from exceptions import (
    Entity,
    InvalidLocationError,
    Location,
    NotFoundError,
    NotProvidedError,
    UIMessage,
    VoiceModelNotFoundError,
    YoutubeUrlError,
)

from common import RVC_MODELS_DIR, SEPARATOR_MODELS_DIR
from typing_extra import AudioExt, F0Method, Json, StrPath

from vc.rvc import Config, get_vc, load_hubert, rvc_infer

from backend.common import (
    INTERMEDIATE_AUDIO_BASE_DIR,
    OUTPUT_AUDIO_DIR,
    display_progress,
    get_file_hash,
    get_hash,
    json_dump,
    json_load,
    validate_url,
)
from backend.typing_extra import (
    AudioExtInternal,
    ConvertedVocalsMetaData,
    EffectedVocalsMetaData,
    FileMetaData,
    SourceType,
)

AUDIO_SEPARATOR = Separator(
    log_level=WARNING,
    model_file_dir=SEPARATOR_MODELS_DIR,
    output_dir=INTERMEDIATE_AUDIO_BASE_DIR,
    mdx_params={
        "hop_length": 1024,
        "segment_size": 256,
        "overlap": 0.001,
        "batch_size": 1,
        "enable_denoise": False,
    },
    mdxc_params={"segment_size": 256, "batch_size": 1, "overlap": 2},
)


def _validate_exists(
    identifier: StrPath,
    entity: Entity,
) -> Path:
    """
    Validate that the provided identifier is not none and that it
    identifies an existing entity, which can be either a voice model,
    a song directory or an audio track.

    Parameters
    ----------
    identifier : StrPath
        The identifier to validate.
    entity : Entity
        The entity that the identifier should identify.

    Returns
    -------
    Path
        The path to the identified entity.

    Raises
    ------
    NotProvidedError
        If the identifier is None.
    NotFoundError
        If the identifier does not identify an existing entity.
    VoiceModelNotFoundError
        If the identifier does not identify an existing voice model.
    NotImplementedError
        If the provided entity is not supported.

    """
    match entity:
        case Entity.MODEL_NAME:
            if not identifier:
                raise NotProvidedError(entity=entity, ui_msg=UIMessage.NO_VOICE_MODEL)
            path = RVC_MODELS_DIR / identifier
            if not path.is_dir():
                raise VoiceModelNotFoundError(str(identifier))
        case Entity.SONG_DIR:
            if not identifier:
                raise NotProvidedError(entity=entity, ui_msg=UIMessage.NO_SONG_DIR)
            path = Path(identifier)
            if not path.is_dir():
                raise NotFoundError(entity=entity, location=path)
        case (
            Entity.SONG
            | Entity.AUDIO_TRACK
            | Entity.VOCALS_TRACK
            | Entity.INSTRUMENTALS_TRACK
            | Entity.MAIN_VOCALS_TRACK
            | Entity.BACKUP_VOCALS_TRACK
        ):
            if not identifier:
                raise NotProvidedError(entity=entity)
            path = Path(identifier)
            if not path.is_file():
                raise NotFoundError(entity=entity, location=path)
        case _:
            error_msg = f"Entity {entity} not supported."
            raise NotImplementedError(error_msg)
    return path


def _validate_all_exist(
    identifier_entity_pairs: Sequence[tuple[StrPath, Entity]],
) -> list[Path]:
    """
    Validate that all provided identifiers are not none and that they
    identify existing entities, which can be either voice models, song
    directories or audio tracks.

    Parameters
    ----------
    identifier_entity_pairs : Sequence[tuple[StrPath, Entity]]
        The pairs of identifiers and entities to validate.

    Returns
    -------
    list[Path]
        The paths to the identified entities.

    """
    return list(starmap(_validate_exists, identifier_entity_pairs))


def _get_input_audio_path(directory: StrPath) -> Path | None:
    """
    Get the path to the input audio file in the provided directory, if
    it exists.

    The provided directory must be located in the root of the
    intermediate audio base directory.

    Parameters
    ----------
    directory : StrPath
        The path to a directory.

    Returns
    -------
    Path | None
        The path to the input audio file in the provided directory, if
        it exists.

    Raises
    ------
    NotFoundError
        If the provided path does not point to an existing directory.
    InvalidLocationError
        If the provided path is not located in the root of the
        intermediate audio base directory"

    """
    dir_path = Path(directory)

    if not dir_path.is_dir():
        raise NotFoundError(entity=Entity.DIRECTORY, location=dir_path)

    if dir_path.parent != INTERMEDIATE_AUDIO_BASE_DIR:
        raise InvalidLocationError(
            entity=Entity.DIRECTORY,
            location=Location.INTERMEDIATE_AUDIO_ROOT,
            path=dir_path,
        )
    # NOTE directory should never contain more than one element which
    # matches the pattern "0_*"
    return next(dir_path.glob("0_*"), None)


def _get_input_audio_paths() -> list[Path]:
    """
    Get the paths to all input audio files in the intermediate audio
    base directory.

    Returns
    -------
    list[Path]
        The paths to all input audio files in the intermediate audio
        base directory.

    """
    # NOTE if we later add .json file for input then
    # we need to exclude those here
    return list(INTERMEDIATE_AUDIO_BASE_DIR.glob("*/0_*"))


def _get_youtube_id(url: str, ignore_playlist: bool = True) -> str:
    """
    Get the id of a YouTube video or playlist.

    Parameters
    ----------
    url : str
        URL which points to a YouTube video or playlist.
    ignore_playlist : bool, default=True
        Whether to get the id of the first video in a playlist or the
        playlist id itself.

    Returns
    -------
    str
        The id of a YouTube video or playlist.

    Raises
    ------
    YoutubeUrlError
        If the provided URL does not point to a YouTube video
        or playlist.

    """
    yt_id = None
    validate_url(url)
    query = urlparse(url)
    if query.hostname == "youtu.be":
        yt_id = query.query[2:] if query.path[1:] == "watch" else query.path[1:]

    elif query.hostname in {"www.youtube.com", "youtube.com", "music.youtube.com"}:
        if not ignore_playlist:
            with suppress(KeyError):
                yt_id = parse_qs(query.query)["list"][0]
        elif query.path == "/watch":
            yt_id = parse_qs(query.query)["v"][0]
        elif query.path[:7] == "/watch/":
            yt_id = query.path.split("/")[1]
        elif query.path[:7] == "/embed/" or query.path[:3] == "/v/":
            yt_id = query.path.split("/")[2]
    if yt_id is None:
        raise YoutubeUrlError(url=url, playlist=True)

    return yt_id


def _get_youtube_audio(url: str, directory: StrPath) -> Path:
    """
    Download audio from a YouTube video.

    Parameters
    ----------
    url : str
        URL which points to a YouTube video.
    directory : StrPath
        The directory to save the downloaded audio file to.

    Returns
    -------
    Path
        The path to the downloaded audio file.

    Raises
    ------
    YoutubeUrlError
        If the provided URL does not point to a YouTube video.

    """
    validate_url(url)
    outtmpl = str(Path(directory, "0_%(title)s"))
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
            },
        ],
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        result = ydl.extract_info(url, download=True)
        if not result:
            raise YoutubeUrlError(url, playlist=False)
        file = ydl.prepare_filename(result, outtmpl=f"{outtmpl}.wav")

    return Path(file)


def _get_rvc_files(model_name: str) -> tuple[Path, Path | None]:
    """
    Get the RVC model file and potential index file of a voice model.

    Parameters
    ----------
    model_name : str
        The name of the voice model to get the RVC files of.

    Returns
    -------
    model_file : Path
        The path to the RVC model file.
    index_file : Path | None
        The path to the RVC index file, if it exists.

    Raises
    ------
    NotFoundError
        If no model file exists in the voice model directory.


    """
    model_dir_path = _validate_exists(model_name, Entity.MODEL_NAME)
    file_path_map = {
        ext: path
        for path in model_dir_path.iterdir()
        for ext in [".pth", ".index"]
        if ext == path.suffix
    }

    if ".pth" not in file_path_map:
        raise NotFoundError(
            entity=Entity.MODEL_FILE,
            location=model_dir_path,
            is_path=False,
        )

    model_file = model_dir_path / file_path_map[".pth"]
    index_file = (
        model_dir_path / file_path_map[".index"] if ".index" in file_path_map else None
    )

    return model_file, index_file


def _convert(
    voice_track: StrPath,
    output_file: StrPath,
    model_name: str,
    n_semitones: int = 0,
    f0_method: F0Method = F0Method.RMVPE,
    index_rate: float = 0.5,
    filter_radius: int = 3,
    rms_mix_rate: float = 0.25,
    protect: float = 0.33,
    hop_length: int = 128,
    output_sr: int = 44100,
) -> None:
    """
    Convert a voice track using a voice model and save the result to a
    an output file.

    Parameters
    ----------
    voice_track : StrPath
        The path to the voice track to convert.
    output_file : StrPath
        The path to the file to save the converted voice track to.
    model_name : str
        The name of the model to use for voice conversion.
    n_semitones : int, default=0
        The number of semitones to pitch-shift the converted voice by.
    f0_method : F0Method, default=F0Method.RMVPE
        The method to use for pitch detection.
    index_rate : float, default=0.5
        The influence of the index file on the voice conversion.
    filter_radius : int, default=3
        The filter radius to use for the voice conversion.
    rms_mix_rate : float, default=0.25
        The blending rate of the volume envelope of the converted voice.
    protect : float, default=0.33
        The protection rate for consonants and breathing sounds.
    hop_length : int, default=128
        The hop length to use for crepe-based pitch detection.
    output_sr : int, default=44100
        The sample rate of the output audio file.

    """
    rvc_model_path, rvc_index_path = _get_rvc_files(model_name)
    device = "cuda:0"
    config = Config(device, is_half=True)
    hubert_model = load_hubert(
        device,
        str(RVC_MODELS_DIR / "hubert_base.pt"),
        is_half=config.is_half,
    )
    cpt, version, net_g, tgt_sr, vc = get_vc(
        device,
        config,
        str(rvc_model_path),
        is_half=config.is_half,
    )

    # convert main vocals
    rvc_infer(
        str(rvc_index_path) if rvc_index_path else "",
        index_rate,
        str(voice_track),
        str(output_file),
        n_semitones,
        f0_method,
        cpt,
        version,
        net_g,
        filter_radius,
        tgt_sr,
        rms_mix_rate,
        protect,
        hop_length,
        vc,
        hubert_model,
        output_sr,
    )
    del hubert_model, cpt
    gc.collect()


def _add_effects(
    audio_track: StrPath,
    output_file: StrPath,
    room_size: float = 0.15,
    wet_level: float = 0.2,
    dry_level: float = 0.8,
    damping: float = 0.7,
) -> None:
    """
    Add high-pass filter, compressor and reverb effects to an audio
    track.

    Parameters
    ----------
    audio_track : StrPath
        The path to the audio track to add effects to.
    output_file : StrPath
        The path to the file to save the effected audio track to.
    room_size : float, default=0.15
        The room size of the reverb effect.
    wet_level : float, default=0.2
        The wetness level of the reverb effect.
    dry_level : float, default=0.8
        The dryness level of the reverb effect.
    damping : float, default=0.7
        The damping of the reverb effect.

    """
    board = Pedalboard(
        [
            HighpassFilter(),
            Compressor(ratio=4, threshold_db=-15),
            Reverb(
                room_size=room_size,
                dry_level=dry_level,
                wet_level=wet_level,
                damping=damping,
            ),
        ],
    )

    with (
        AudioFile(str(audio_track)) as f,
        AudioFile(str(output_file), "w", f.samplerate, f.num_channels) as o,
    ):
        # Read one second of audio at a time, until the file is empty:
        while f.tell() < f.frames:
            chunk = f.read(int(f.samplerate))
            effected = board(chunk, f.samplerate, reset=False)
            o.write(effected)


def _pitch_shift(audio_track: StrPath, output_file: StrPath, n_semi_tones: int) -> None:
    """
    Pitch-shift an audio track.

    Parameters
    ----------
    audio_track : StrPath
        The path to the audio track to pitch-shift.
    output_file : StrPath
        The path to the file to save the pitch-shifted audio track to.
    n_semi_tones : int
        The number of semi-tones to pitch-shift the audio track by.

    """
    y, sr = sf.read(audio_track)
    tfm = sox.Transformer()
    tfm.pitch(n_semi_tones)
    y_shifted = tfm.build_array(input_array=y, sample_rate_in=sr)
    sf.write(output_file, y_shifted, sr)


def _get_model_name(
    effected_vocals_track: StrPath | None = None,
    song_dir: StrPath | None = None,
) -> str:
    """
    Infer the name of the voice model used for vocal conversion from a
    an effected vocals track in a given song directory.

    If a voice model name cannot be inferred, "Unknown" is returned.

    Parameters
    ----------
    effected_vocals_track : StrPath, optional
        The path to an effected vocals track.
    song_dir : StrPath, optional
        The path to a song directory.

    Returns
    -------
    str
        The name of the voice model used for vocal conversion.

    """
    model_name = "Unknown"
    if not (effected_vocals_track and song_dir):
        return model_name
    effected_vocals_path = Path(effected_vocals_track)
    song_dir_path = Path(song_dir)
    effected_vocals_json_path = song_dir_path / f"{effected_vocals_path.stem}.json"
    if not effected_vocals_json_path.is_file():
        return model_name
    effected_vocals_dict = json_load(effected_vocals_json_path)
    try:
        effected_vocals_metadata = EffectedVocalsMetaData.model_validate(
            effected_vocals_dict,
        )
    except ValidationError:
        return model_name
    converted_vocals_track_name = effected_vocals_metadata.vocals_track.name
    converted_vocals_json_path = song_dir_path / Path(
        converted_vocals_track_name,
    ).with_suffix(
        ".json",
    )
    if not converted_vocals_json_path.is_file():
        return model_name
    converted_vocals_dict = json_load(converted_vocals_json_path)
    try:
        converted_vocals_metadata = ConvertedVocalsMetaData.model_validate(
            converted_vocals_dict,
        )
    except ValidationError:
        return model_name
    return converted_vocals_metadata.model_name


def _to_internal(audio_ext: AudioExt) -> AudioExtInternal:
    """
    Map an audio extension to an internally recognized format.

    Parameters
    ----------
    audio_ext : AudioExt
        The audio extension to map.

    Returns
    -------
    AudioExtInternal
        The internal audio extension.

    """
    match audio_ext:
        case AudioExt.M4A:
            return AudioExtInternal.IPOD
        case AudioExt.AAC:
            return AudioExtInternal.ADTS
        case _:
            return AudioExtInternal(audio_ext)


def _mix_song(
    main_vocals_track: StrPath,
    instrumentals_track: StrPath,
    backup_vocals_track: StrPath,
    output_file: StrPath,
    main_gain: int = 0,
    inst_gain: int = 0,
    backup_gain: int = 0,
    output_sr: int = 44100,
    output_format: AudioExt = AudioExt.MP3,
) -> None:
    """
    Mix a main vocals track, an instrumentals track, and a backup vocals
    track to create a song.

    Parameters
    ----------
    main_vocals_track : StrPath
        The path to the main vocals track to mix.
    backup_vocals_track : StrPath
        The path to the backup vocals track to mix.
    instrumentals_track : StrPath
        The path to the instrumentals track to mix.
    output_file : StrPath
        The path to the file to save the mixed song to.
    main_gain : int, default=0
        The gain to apply to the main vocals track.
    inst_gain : int, default=0
        The gain to apply to the instrumentals track.
    backup_gain : int, default=0
        The gain to apply to the backup vocals track.
    output_sr : int, default=44100
        The sample rate of the mixed song.
    output_format : AudioExt, default=AudioExt.MP3
        The audio format of the mixed song.

    """
    main_vocal_audio = AudioSegment.from_wav(main_vocals_track) + main_gain
    instrumental_audio = AudioSegment.from_wav(instrumentals_track) + inst_gain
    backup_vocal_audio = AudioSegment.from_wav(backup_vocals_track) + backup_gain
    mixed_audio = main_vocal_audio.overlay(backup_vocal_audio).overlay(
        instrumental_audio,
    )
    mixed_audio_resampled = mixed_audio.set_frame_rate(output_sr)
    mixed_audio_resampled.export(
        output_file,
        format=_to_internal(output_format),
    )


def get_named_song_dirs() -> list[tuple[str, str]]:
    """
    Get the names of all saved songs and the paths to the
    directories where they are stored.

    Returns
    -------
    list[tuple[str, Path]]
        A list of tuples containing the name of each saved song
        and the path to the directory where it is stored.

    """
    return sorted(
        [
            (
                path.stem.removeprefix("0_"),
                str(path.parent),
            )
            for path in _get_input_audio_paths()
        ],
        key=operator.itemgetter(0),
    )


def init_song_dir(
    source: str,
    progress_bar: gr.Progress | None = None,
    percentage: float = 0.5,
) -> tuple[Path, SourceType]:
    """
    Initialize a directory for a song provided by a given source.


    The song directory is initialized as follows:

    * If the source is a YouTube URL, the id of the video which
    that URL points to is extracted. A new song directory with the name
    of that id is then created, if it does not already exist.
    * If the source is a path to a local audio file, the hash of
    that audio file is extracted. A new song directory with the name of
    that hash is then created, if it does not already exist.
    * if the source is a path to an existing song directory, then
    that song directory is used as is.

    Parameters
    ----------
    source : str
        The source providing the song to initialize a directory for.
    progress_bar : gr.Progress, optional
        Gradio progress bar to update.
    percentage : float, default=0.5
        Percentage to display in the progress bar.

    Returns
    -------
    song_dir : Path
        The path to the initialized song directory.
    source_type : SourceType
        The type of source provided.

    Raises
    ------
    NotProvidedError
        If no source is provided.
    InvalidLocationError
        If a provided path points to a directory that is not located in
        the root of the intermediate audio base directory.
    NotFoundError
        If the provided source is a path to a file that does not exist.

    """
    if not source:
        raise NotProvidedError(entity=Entity.SOURCE, ui_msg=UIMessage.NO_SOURCE)
    source_path = Path(source)

    display_progress("[~] Initializing song directory...", percentage, progress_bar)

    # if source is a path to an existing song directory
    if source_path.is_dir():
        if source_path.parent != INTERMEDIATE_AUDIO_BASE_DIR:
            raise InvalidLocationError(
                entity=Entity.DIRECTORY,
                location=Location.INTERMEDIATE_AUDIO_ROOT,
                path=source_path,
            )
        display_progress(
            "[~] Using existing song directory...",
            percentage,
            progress_bar,
        )
        source_type = SourceType.SONG_DIR
        return source_path, source_type

    # if source is a URL
    if urlparse(source).scheme == "https":
        source_type = SourceType.URL
        song_id = _get_youtube_id(source)

    # if source is a path to a local audio file
    elif source_path.is_file():
        source_type = SourceType.FILE
        song_id = get_file_hash(source_path)
    else:
        raise NotFoundError(entity=Entity.FILE, location=source_path)

    song_dir_path = INTERMEDIATE_AUDIO_BASE_DIR / song_id

    song_dir_path.mkdir(parents=True, exist_ok=True)

    return song_dir_path, source_type


# NOTE consider increasing hash_size to 16. Otherwise
# we might have problems with hash collisions when using app as CLI
def get_unique_base_path(
    song_dir: StrPath,
    prefix: str,
    args_dict: Json,
    hash_size: int = 5,
    progress_bar: gr.Progress | None = None,
    percentage: float = 0.5,
) -> Path:
    """
    Get a unique base path (a path without any extension) for a file in
    a song directory by hashing the arguments used to generate
    the audio that is stored or will be stored in that file.

    Parameters
    ----------
    song_dir :StrPath
        The path to a song directory.
    prefix : str
        The prefix to use for the base path.
    args_dict : Json
        A JSON-serializable dictionary of named arguments used to
        generate the audio that is stored or will be stored in a file
        in the song directory.
    hash_size : int, default=5
        The size (in bytes) of the hash to use for the base path.
    progress_bar : gr.Progress, optional
        Gradio progress bar to update.
    percentage : float, default=0.5
        Percentage to display in the progress bar.

    Returns
    -------
    Path
        The unique base path for a file in a song directory.

    Raises
    ------
    NotProvidedError
        If no song directory is provided.

    """
    if not song_dir:
        raise NotProvidedError(entity=Entity.SONG_DIR, ui_msg=UIMessage.NO_SONG_DIR)
    song_dir_path = Path(song_dir)
    dict_hash = get_hash(args_dict, size=hash_size)
    while True:
        base_path = song_dir_path / f"{prefix}_{dict_hash}"
        json_path = base_path.with_suffix(".json")
        if json_path.exists():
            file_dict = json_load(json_path)
            if file_dict == args_dict:
                return base_path
            display_progress("[~] Rehashing...", percentage, progress_bar)
            dict_hash = get_hash(dict_hash, size=hash_size)
        else:
            return base_path


# NOTE this function perhaps should be renamed to waveify and then just
# to conversion to .wav format, possibly with a parameter ac to specify
# the number of audio channels
def stereoize(
    song: StrPath,
    song_dir: StrPath,
    progress_bar: gr.Progress | None = None,
    percentage: float = 0.5,
) -> Path:
    """
    Convert a song to stereo.

    If the song is already in stereo format, it will not be converted.

    Parameters
    ----------
    song : StrPath
        The path to the song to convert.
    song_dir : StrPath
        The path to the song directory where the converted song will
        be saved.
    progress_bar : gr.Progress, optional
        Gradio progress bar to update.
    percentage : float, default=0.5
        Percentage to display in the progress bar.

    Returns
    -------
    Path
        The path to the song in stereo format.

    """
    song_path, song_dir_path = _validate_all_exist(
        [(song, Entity.SONG), (song_dir, Entity.SONG_DIR)],
    )

    stereo_path = song_path

    song_info = pydub_utils.mediainfo(str(song_path))
    if song_info["channels"] == "1":
        args_dict = {
            "song": {"name": song_path.name, "hash_id": get_file_hash(song_path)},
        }

        paths = [
            get_unique_base_path(
                song_dir_path,
                "0_Stereo",
                args_dict,
                progress_bar=progress_bar,
                percentage=percentage,
            ).with_suffix(suffix)
            for suffix in [".wav", ".json"]
        ]
        stereo_path, stereo_json_path = paths
        if not all(path.exists() for path in paths):
            display_progress(
                "[~] Converting song to stereo format...",
                percentage,
                progress_bar,
            )

            ffmpeg.input(song_path).output(filename=stereo_path, f="wav", ac=2).run(
                overwrite_output=True,
                quiet=True,
            )
            json_dump(args_dict, stereo_json_path)

    return stereo_path


def retrieve_song(
    source: str,
    progress_bar: gr.Progress | None = None,
    percentage: float = 0.5,
) -> tuple[Path, Path]:
    """
    Retrieve a song from a source that can either be a YouTube URL, a
    local audio file or a song directory.

    Parameters
    ----------
    source : str
        A Youtube URL, the path to a local audio file or the path to a
        song directory.
    progress_bar : gr.Progress, optional
        Gradio progress bar to update.
    percentage : float, default=0.5
        Percentage to display in the progress bar.

    Returns
    -------
    song : Path
        The path to the retrieved song.
    song_dir : Path
        The path to the song directory containing the retrieved song.

    Raises
    ------
    NotProvidedError
        If no source is provided.

    """
    if not source:
        raise NotProvidedError(entity=Entity.SOURCE, ui_msg=UIMessage.NO_SOURCE)

    song_dir_path, source_type = init_song_dir(source, progress_bar, percentage)
    song_path = _get_input_audio_path(song_dir_path)

    if not song_path:
        if source_type == SourceType.URL:
            display_progress("[~] Downloading song...", percentage, progress_bar)
            song_url = source.split("&")[0]
            song_path = _get_youtube_audio(song_url, song_dir_path)

        else:
            display_progress("[~] Copying song...", percentage, progress_bar)
            source_path = Path(source)
            song_name = f"0_{source_path.name}"
            song_path = song_dir_path / song_name
            shutil.copyfile(source_path, song_path)

    return song_path, song_dir_path


def separate_audio(
    audio_track: StrPath,
    song_dir: StrPath,
    primary_prefix: str,
    secondary_prefix: str,
    model_name: str,
    segment_size: int,
    display_msg: str,
    progress_bar: gr.Progress | None = None,
    percentage: float = 0.5,
) -> tuple[Path, Path]:
    """
    Separate an audio track into a primary stem and a secondary stem.

    Parameters
    ----------
    audio_track : StrPath
        The path to the audio track to separate.
    song_dir : StrPath
        The path to the song directory where the separated primary stem
        and secondary stem will be saved.
    primary_prefix : str
        The prefix to use for the name of the primary stem.
    secondary_prefix : str
        The prefix to use for the name of the secondary stem.
    model_name : str
        The name of the model to use for audio separation.
    segment_size : int
        The segment size to use for audio separation.
    display_msg : str
        The message to display when separating the audio track.
    progress_bar : gr.Progress, optional
        Gradio progress bar to update.
    percentage : float, default=0.5
        Percentage to display in the progress bar.

    Returns
    -------
    primary_path : Path
        The path to the separated primary stem.
    secondary_path : Path
        The path to the separated secondary stem.

    """
    audio_path, song_dir_path = _validate_all_exist(
        [(audio_track, Entity.AUDIO_TRACK), (song_dir, Entity.SONG_DIR)],
    )

    args_dict = {
        "audio_track": {
            "name": audio_path.name,
            "hash_id": get_file_hash(audio_path),
        },
    }

    paths = [
        get_unique_base_path(
            song_dir_path,
            prefix,
            args_dict,
            progress_bar=progress_bar,
            percentage=percentage,
        ).with_suffix(suffix)
        for prefix in [primary_prefix, secondary_prefix]
        for suffix in [".wav", ".json"]
    ]

    (
        primary_path,
        primary_json_path,
        secondary_path,
        secondary_json_path,
    ) = paths

    if not all(path.exists() for path in paths):

        display_progress(display_msg, percentage, progress_bar)
        AUDIO_SEPARATOR.arch_specific_params["MDX"]["segment_size"] = segment_size
        AUDIO_SEPARATOR.load_model(model_name)
        primary_temp_name, secondary_temp_name = AUDIO_SEPARATOR.separate(
            str(audio_path),
        )
        primary_temp_path = INTERMEDIATE_AUDIO_BASE_DIR / primary_temp_name
        secondary_temp_path = INTERMEDIATE_AUDIO_BASE_DIR / secondary_temp_name
        primary_temp_path.rename(primary_path)
        secondary_temp_path.rename(secondary_path)
        json_dump(args_dict, primary_json_path)
        json_dump(args_dict, secondary_json_path)

    return primary_path, secondary_path


def separate_vocals(
    song: StrPath,
    song_dir: StrPath,
    progress_bar: gr.Progress | None = None,
    percentage: float = 0.5,
) -> tuple[Path, Path]:
    """
    Separate a song into a vocals track and an instrumentals track.

    Parameters
    ----------
    song : StrPath
        The path to the song to separate.
    song_dir : StrPath
        The path to the song directory where the separated vocals track
        and instrumentals track will be saved.
    progress_bar : gr.Progress, optional
        Gradio progress bar to update.
    percentage : float, default=0.5
        Percentage to display in the progress bar.

    Returns
    -------
    vocals_track : Path
        The path to the separated vocals track.
    instrumentals_track : Path
        The path to the separated instrumentals track.


    """
    song_path, song_dir_path = _validate_all_exist(
        [(song, Entity.SONG), (song_dir, Entity.SONG_DIR)],
    )

    instrumentals_path, vocals_path = separate_audio(
        song_path,
        song_dir_path,
        "1_Instrumentals",
        "1_Vocals",
        "UVR-MDX-NET-Voc_FT.onnx",
        512,
        "[~] Separating vocals from instrumentals...",
        progress_bar,
        percentage,
    )
    return vocals_path, instrumentals_path


def separate_main_vocals(
    vocals_track: StrPath,
    song_dir: StrPath,
    progress_bar: gr.Progress | None = None,
    percentage: float = 0.5,
) -> tuple[Path, Path]:
    """
    Separate a vocals track into a main vocals track and a backup vocals
    track.

    Parameters
    ----------
    vocals_track : StrPath
        The path to the vocals track to separate.
    song_dir : StrPath
        The path to the song directory where the separated main vocals
        track and backup vocals track will be saved.
    progress_bar : gr.Progress, optional
        Gradio progress bar to update.
    percentage : float, default=0.5
        Percentage to display in the progress bar.

    Returns
    -------
    main_vocals_track : Path
        The path to the separated main vocals track.
    backup_vocals_track : Path
        The path to the separated backup vocals track.


    """
    vocals_path, song_dir_path = _validate_all_exist(
        [(vocals_track, Entity.VOCALS_TRACK), (song_dir, Entity.SONG_DIR)],
    )

    return separate_audio(
        vocals_path,
        song_dir_path,
        "2_Vocals_Main",
        "2_Vocals_Backup",
        "UVR_MDXNET_KARA_2.onnx",
        512,
        "[~] Separating main vocals from backup vocals...",
        progress_bar,
        percentage,
    )


def dereverb(
    vocals_track: StrPath,
    song_dir: StrPath,
    progress_bar: gr.Progress | None = None,
    percentage: float = 0.5,
) -> tuple[Path, Path]:
    """
    De-reverb a vocals track.

    Parameters
    ----------
    vocals_track : StrPath
        The path to the vocals track to de-reverb.
    song_dir : StrPath
        The path to the song directory where the de-reverbed vocals
        track will be saved.
    progress_bar : gr.Progress, optional
        Gradio progress bar to update.
    percentage : float, default=0.5
        Percentage to display in the progress bar.

    Returns
    -------
    vocals_dereverb_track : Path
        The path to the de-reverbed vocals track.
    vocals_reverb_track : Path
        The path to the vocals reverb track.

    """
    vocals_path, song_dir_path = _validate_all_exist(
        [(vocals_track, Entity.VOCALS_TRACK), (song_dir, Entity.SONG_DIR)],
    )

    return separate_audio(
        vocals_path,
        song_dir_path,
        "3_Vocals_DeReverb",
        "3_Vocals_Reverb",
        "Reverb_HQ_By_FoxJoy.onnx",
        256,
        "[~] De-reverbing vocals...",
        progress_bar,
        percentage,
    )


def convert(
    vocals_track: StrPath,
    song_dir: StrPath,
    model_name: str,
    n_octaves: int = 0,
    n_semitones: int = 0,
    f0_method: F0Method = F0Method.RMVPE,
    index_rate: float = 0.5,
    filter_radius: int = 3,
    rms_mix_rate: float = 0.25,
    protect: float = 0.33,
    hop_length: int = 128,
    progress_bar: gr.Progress | None = None,
    percentage: float = 0.5,
) -> Path:
    """
    Convert a vocals track using a voice model.

    Parameters
    ----------
    vocals_track : StrPath
        The path to the vocals track to convert.
    song_dir : StrPath
        The path to the song directory where the converted vocals track
        will be saved.
    model_name : str
        The name of the model to use for vocal conversion.
    n_octaves : int, default=0
        The number of octaves to pitch-shift the converted vocals by.
    n_semitones : int, default=0
        The number of semitones to pitch-shift the converted vocals by.
    f0_method : F0Method, default=F0Method.RMVPE
        The method to use for pitch detection.
    index_rate : float, default=0.5
        The influence of the index file on the vocal conversion.
    filter_radius : int, default=3
        The filter radius to use for the vocal conversion.
    rms_mix_rate : float, default=0.25
        The blending rate of the volume envelope of the converted
        vocals.
    protect : float, default=0.33
        The protection rate for consonants and breathing sounds.
    hop_length : int, default=128
        The hop length to use for crepe-based pitch detection.
    progress_bar : gr.Progress, optional
        Gradio progress bar to update.
    percentage : float, default=0.5
        Percentage to display in the progress bar.

    Returns
    -------
    Path
        The path to the converted vocals track.

    """
    vocals_path, song_dir_path, _ = _validate_all_exist(
        [
            (vocals_track, Entity.VOCALS_TRACK),
            (song_dir, Entity.SONG_DIR),
            (model_name, Entity.MODEL_NAME),
        ],
    )

    n_semitones = n_octaves * 12 + n_semitones

    args_dict = ConvertedVocalsMetaData(
        vocals_track=FileMetaData(
            name=vocals_path.name,
            hash_id=get_file_hash(vocals_path),
        ),
        model_name=model_name,
        n_semitones=n_semitones,
        f0_method=f0_method,
        index_rate=index_rate,
        filter_radius=filter_radius,
        rms_mix_rate=rms_mix_rate,
        protect=protect,
        hop_length=hop_length,
    ).model_dump()

    paths = [
        get_unique_base_path(
            song_dir_path,
            "4_Vocals_Converted",
            args_dict,
            progress_bar=progress_bar,
            percentage=percentage,
        ).with_suffix(suffix)
        for suffix in [".wav", ".json"]
    ]

    converted_vocals_path, converted_vocals_json_path = paths

    if not all(path.exists() for path in paths):
        display_progress("[~] Converting vocals using RVC...", percentage, progress_bar)
        _convert(
            vocals_path,
            converted_vocals_path,
            model_name,
            n_semitones,
            f0_method,
            index_rate,
            filter_radius,
            rms_mix_rate,
            protect,
            hop_length,
            output_sr=44100,
        )
        json_dump(args_dict, converted_vocals_json_path)
    return converted_vocals_path


def postprocess(
    vocals_track: StrPath,
    song_dir: StrPath,
    room_size: float = 0.15,
    wet_level: float = 0.2,
    dry_level: float = 0.8,
    damping: float = 0.7,
    progress_bar: gr.Progress | None = None,
    percentage: float = 0.5,
) -> Path:
    """
    Apply high-pass filter, compressor and reverb effects to a vocals
    track.

    Parameters
    ----------
    vocals_track : StrPath
        The path to the vocals track to add effects to.
    song_dir : StrPath
        The path to the song directory where the effected vocals track
        will be saved.
    room_size : float, default=0.15
        The room size of the reverb effect.
    wet_level : float, default=0.2
        The wetness level of the reverb effect.
    dry_level : float, default=0.8
        The dryness level of the reverb effect.
    damping : float, default=0.7
        The damping of the reverb effect.
    progress_bar : gr.Progress, optional
        Gradio progress bar to update.
    percentage : float, default=0.5
        Percentage to display in the progress bar.

    Returns
    -------
    Path
        The path to the effected vocals track.

    """
    vocals_path, song_dir_path = _validate_all_exist(
        [(vocals_track, Entity.VOCALS_TRACK), (song_dir, Entity.SONG_DIR)],
    )

    args_dict = EffectedVocalsMetaData(
        vocals_track=FileMetaData(
            name=vocals_path.name,
            hash_id=get_file_hash(vocals_path),
        ),
        room_size=room_size,
        wet_level=wet_level,
        dry_level=dry_level,
        damping=damping,
    ).model_dump()

    paths = [
        get_unique_base_path(
            song_dir_path,
            "5_Vocals_Effected",
            args_dict,
            progress_bar=progress_bar,
            percentage=percentage,
        ).with_suffix(suffix)
        for suffix in [".wav", ".json"]
    ]

    effected_vocals_path, effected_vocals_json_path = paths

    if not all(path.exists() for path in paths):
        display_progress(
            "[~] Applying audio effects to vocals...",
            percentage,
            progress_bar,
        )
        _add_effects(
            vocals_path,
            effected_vocals_path,
            room_size,
            wet_level,
            dry_level,
            damping,
        )
        json_dump(args_dict, effected_vocals_json_path)
    return effected_vocals_path


def pitch_shift(
    audio_track: StrPath,
    song_dir: StrPath,
    n_semitones: int,
    prefix: str,
    display_msg: str,
    progress_bar: gr.Progress | None = None,
    percentage: float = 0.5,
) -> Path:
    """
    Pitch shift an audio track by a given number of semi-tones.

    Parameters
    ----------
    audio_track : StrPath
        The path to the audio track to pitch shift.
    song_dir : StrPath
        The path to the song directory where the pitch-shifted audio
        track will be saved.
    n_semitones : int
        The number of semi-tones to pitch-shift the audio track by.
    prefix : str
        The prefix to use for the name of the pitch-shifted audio track.
    display_msg : str
        The message to display when pitch-shifting the audio track.
    progress_bar : gr.Progress, optional
        Gradio progress bar to update.
    percentage : float, default=0.5
        Percentage to display in the progress bar.

    Returns
    -------
    Path
        The path to the pitch-shifted audio track.

    """
    audio_path, song_dir_path = _validate_all_exist(
        [(audio_track, Entity.AUDIO_TRACK), (song_dir, Entity.SONG_DIR)],
    )

    shifted_audio_path = audio_path

    if n_semitones != 0:

        args_dict = {
            "audio_track": {
                "name": audio_path.name,
                "hash_id": get_file_hash(audio_path),
            },
            "n_semitones": n_semitones,
        }

        paths = [
            get_unique_base_path(
                song_dir_path,
                prefix,
                args_dict,
                progress_bar=progress_bar,
                percentage=percentage,
            ).with_suffix(suffix)
            for suffix in [".wav", ".json"]
        ]

        shifted_audio_path, shifted_audio_json_path = paths

        if not all(path.exists() for path in paths):
            display_progress(display_msg, percentage, progress_bar)
            _pitch_shift(audio_path, shifted_audio_path, n_semitones)
            json_dump(args_dict, shifted_audio_json_path)

    return shifted_audio_path


def pitch_shift_background(
    instrumentals_track: StrPath,
    backup_vocals_track: StrPath,
    song_dir: StrPath,
    n_semitones: int = 0,
    progress_bar: gr.Progress | None = None,
    percentages: tuple[float, float] = (0.0, 0.5),
) -> tuple[Path, Path]:
    """
    Pitch shift an instrumentals track and a backup vocals track by a
    given number of semi-tones.

    Parameters
    ----------
    instrumentals_track : StrPath
        The path to the instrumentals track to pitch shift.
    backup_vocals_track : StrPath
        The path to the backup vocals track to pitch shift.
    song_dir : StrPath
        The path to the directory where the pitch-shifted instrumentals
        track and backup vocals track will be saved.
    n_semitones : int, default=0
        The number of semi-tones to pitch-shift the instrumentals track
        and backup vocals track by.
    progress_bar : gr.Progress, optional
        Gradio progress bar to update.
    percentages : tuple[float,float], default=(0.0, 0.5)
        Percentages to display in the progress bar.

    Returns
    -------
    shifted_instrumentals_track : Path
        The path to the pitch-shifted instrumentals track.
    shifted:backup_vocals_track : Path
        The path to the pitch-shifted backup vocals track.

    """
    instrumentals_path, backup_vocals_path, song_dir_path = _validate_all_exist(
        [
            (instrumentals_track, Entity.INSTRUMENTALS_TRACK),
            (backup_vocals_track, Entity.BACKUP_VOCALS_TRACK),
            (song_dir, Entity.SONG_DIR),
        ],
    )

    shifted_instrumentals_path = pitch_shift(
        instrumentals_path,
        song_dir_path,
        n_semitones,
        "6_Instrumentals_Shifted",
        "[~] Pitch-shifting instrumentals...",
        progress_bar,
        percentages[0],
    )

    shifted_backup_vocals_path = pitch_shift(
        backup_vocals_path,
        song_dir_path,
        n_semitones,
        "6_Backup_Vocals_Shifted",
        "[~] Pitch-shifting backup vocals...",
        progress_bar,
        percentages[1],
    )
    return shifted_instrumentals_path, shifted_backup_vocals_path


def get_song_cover_name(
    effected_vocals_track: StrPath | None = None,
    song_dir: StrPath | None = None,
    model_name: str | None = None,
    progress_bar: gr.Progress | None = None,
    percentage: float = 0.5,
) -> str:
    """
    Generate a suitable name for a cover of a song based on the name
    of that song and the voice model used for vocal conversion.

    If the path of an existing song directory is provided, the name
    of the song is inferred from that directory. If a voice model is not
    provided but the path of an existing song directory and the path of
    an effected vocals track in that directory are provided, then the
    voice model is inferred from the effected vocals track.

    Parameters
    ----------
    effected_vocals_track : StrPath, optional
        The path to an effected vocals track.
    song_dir : StrPath, optional
        The path to a song directory.
    model_name : str, optional
        The name of a voice model.
    progress_bar : gr.Progress, optional
        Gradio progress bar to update.
    percentage : float, default=0.5
        Percentage to display in the progress bar.

    Returns
    -------
    str
        The song cover name

    """
    display_progress("[~] Getting song cover name...", percentage, progress_bar)

    song_name = "Unknown"
    if song_dir and (song_path := _get_input_audio_path(song_dir)):
        song_name = song_path.stem.removeprefix("0_")
    model_name = model_name or _get_model_name(effected_vocals_track, song_dir)

    return f"{song_name} ({model_name} Ver)"


def mix_song_cover(
    main_vocals_track: StrPath,
    instrumentals_track: StrPath,
    backup_vocals_track: StrPath,
    song_dir: StrPath,
    main_gain: int = 0,
    inst_gain: int = 0,
    backup_gain: int = 0,
    output_sr: int = 44100,
    output_format: AudioExt = AudioExt.MP3,
    output_name: str | None = None,
    progress_bar: gr.Progress | None = None,
    percentage: float = 0.5,
) -> Path:
    """
    Mix a main vocals track, an instrumentals track, and a backup vocals
    track to create a song cover.

    Parameters
    ----------
    main_vocals_track : StrPath
        The path to the main vocals track to mix.
    instrumentals_track : StrPath
        The path to the instrumentals track to mix.
    backup_vocals_track : StrPath
        The path to the backup vocals track to mix.
    song_dir : StrPath
        The path to the song directory where the song cover will be
        saved.
    main_gain : int, default=0
        The gain to apply to the main vocals track.
    inst_gain : int, default=0
        The gain to apply to the instrumentals track.
    backup_gain : int, default=0
        The gain to apply to the backup vocals track.
    output_sr : int, default=44100
        The sample rate of the song cover.
    output_format : AudioExt, default=AudioExt.MP3
        The audio format of the song cover.
    output_name : str, optional
        The name of the song cover.
    progress_bar : gr.Progress, optional
        Gradio progress bar to update.
    percentage : float, default=0.5
        Percentage to display in the progress bar.

    Returns
    -------
    Path
        The path to the song cover.


    """
    main_vocals_path, instrumentals_path, backup_vocals_path, song_dir_path = (
        _validate_all_exist(
            [
                (main_vocals_track, Entity.MAIN_VOCALS_TRACK),
                (instrumentals_track, Entity.INSTRUMENTALS_TRACK),
                (backup_vocals_track, Entity.BACKUP_VOCALS_TRACK),
                (song_dir, Entity.SONG_DIR),
            ],
        )
    )
    args_dict = {
        "main_vocals_track": {
            "name": main_vocals_path.name,
            "hash_id": get_file_hash(main_vocals_path),
        },
        "instrumentals_track": {
            "name": instrumentals_path.name,
            "hash_id": get_file_hash(instrumentals_path),
        },
        "backup_vocals_track": {
            "name": backup_vocals_path.name,
            "hash_id": get_file_hash(backup_vocals_path),
        },
        "main_gain": main_gain,
        "inst_gain": inst_gain,
        "backup_gain": backup_gain,
        "output_sr": output_sr,
        "output_format": output_format,
    }

    paths = [
        get_unique_base_path(
            song_dir_path,
            "7_Mix",
            args_dict,
            progress_bar=progress_bar,
            percentage=percentage,
        ).with_suffix(suffix)
        for suffix in ["." + output_format, ".json"]
    ]

    mix_path, mix_json_path = paths

    if not all(path.exists() for path in paths):
        display_progress(
            "[~] Mixing main vocals, instrumentals, and backup vocals...",
            percentage,
            progress_bar,
        )

        _mix_song(
            main_vocals_path,
            instrumentals_path,
            backup_vocals_path,
            mix_path,
            main_gain,
            inst_gain,
            backup_gain,
            output_sr,
            output_format,
        )
        json_dump(args_dict, mix_json_path)

    output_name = output_name or get_song_cover_name(
        main_vocals_path,
        song_dir_path,
        None,
        progress_bar,
        percentage,
    )
    song_cover_path = OUTPUT_AUDIO_DIR / f"{output_name}.{output_format}"
    OUTPUT_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(mix_path, song_cover_path)

    return song_cover_path


def run_pipeline(
    source: str,
    model_name: str,
    n_octaves: int = 0,
    n_semitones: int = 0,
    f0_method: F0Method = F0Method.RMVPE,
    index_rate: float = 0.5,
    filter_radius: int = 3,
    rms_mix_rate: float = 0.25,
    protect: float = 0.33,
    hop_length: int = 128,
    room_size: float = 0.15,
    wet_level: float = 0.2,
    dry_level: float = 0.8,
    damping: float = 0.7,
    main_gain: int = 0,
    inst_gain: int = 0,
    backup_gain: int = 0,
    output_sr: int = 44100,
    output_format: AudioExt = AudioExt.MP3,
    output_name: str | None = None,
    return_intermediate: bool = False,
    progress_bar: gr.Progress | None = None,
) -> Path | tuple[Path, ...]:
    """
    Run the song cover generation pipeline.

    Parameters
    ----------
    source : str
        A Youtube URL, the path to a local audio file or the path to a
        song directory.
    model_name : str
        The name of the voice model to use for vocal conversion.
    n_octaves : int, default=0
        The number of octaves to pitch-shift the converted vocals by.
    n_semitones : int, default=0
        The number of semi-tones to pitch-shift the converted vocals,
        instrumentals, and backup vocals by.
    f0_method : F0Method, default=F0Method.RMVPE
        The method to use for pitch detection during vocal conversion.
    index_rate : float, default=0.5
        The influence of the index file on the vocal conversion.
    filter_radius : int, default=3
        The filter radius to use for the vocal conversion.
    rms_mix_rate : float, default=0.25
        The blending rate of the volume envelope of the converted
        vocals.
    protect : float, default=0.33
        The protection rate for consonants and breathing sounds during
        vocal conversion.
    hop_length : int, default=128
        The hop length to use for crepe-based pitch detection.
    room_size : float, default=0.15
        The room size of the reverb effect to apply to the converted
        vocals.
    wet_level : float, default=0.2
        The wetness level of the reverb effect to apply to the converted
        vocals.
    dry_level : float, default=0.8
        The dryness level of the reverb effect to apply to the converted
        vocals.
    damping : float, default=0.7
        The damping of the reverb effect to apply to the converted
        vocals.
    main_gain : int, default=0
        The gain to apply to the post-processed vocals.
    inst_gain : int, default=0
        The gain to apply to the pitch-shifted instrumentals.
    backup_gain : int, default=0
        The gain to apply to the pitch-shifted backup vocals.
    output_sr : int, default=44100
        The sample rate of the song cover.
    output_format : AudioExt, default=AudioExt.MP3
        The audio format of the song cover.
    output_name : str, optional
        The name of the song cover.
    return_intermediate : bool, default=False
        Whether to return the paths of any intermediate audio
        files generated during execution of the pipeline.
    progress_bar : gr.Progress, optional
        Gradio progress bar to update.

    Returns
    -------
    Path | tuple[Path,...]
        The path to the generated song cover and, if
        `return_intermediate=True`, also the paths to any intermediate
        audio files that were generated.

    """
    _validate_exists(model_name, Entity.MODEL_NAME)
    display_progress("[~] Starting song cover generation pipeline...", 0, progress_bar)
    song, song_dir = retrieve_song(
        source,
        progress_bar=progress_bar,
        percentage=0 / 9,
    )
    vocals_track, instrumentals_track = separate_vocals(
        song,
        song_dir,
        progress_bar=progress_bar,
        percentage=1 / 9,
    )
    main_vocals_track, backup_vocals_track = separate_main_vocals(
        vocals_track,
        song_dir,
        progress_bar=progress_bar,
        percentage=2 / 9,
    )
    vocals_dereverb_track, reverb_track = dereverb(
        main_vocals_track,
        song_dir,
        progress_bar=progress_bar,
        percentage=3 / 9,
    )
    converted_vocals_track = convert(
        vocals_dereverb_track,
        song_dir,
        model_name,
        n_octaves,
        n_semitones,
        f0_method,
        index_rate,
        filter_radius,
        rms_mix_rate,
        protect,
        hop_length,
        progress_bar=progress_bar,
        percentage=4 / 9,
    )
    vocals_mixed_track = postprocess(
        converted_vocals_track,
        song_dir,
        room_size,
        wet_level,
        dry_level,
        damping,
        progress_bar=progress_bar,
        percentage=5 / 9,
    )
    instrumentals_shifted_track, backup_vocals_shifted_track = pitch_shift_background(
        instrumentals_track,
        backup_vocals_track,
        song_dir,
        n_semitones,
        progress_bar=progress_bar,
        percentages=(6 / 9, 7 / 9),
    )

    song_cover = mix_song_cover(
        vocals_mixed_track,
        instrumentals_shifted_track,
        backup_vocals_shifted_track,
        song_dir,
        main_gain,
        inst_gain,
        backup_gain,
        output_sr,
        output_format,
        output_name,
        progress_bar=progress_bar,
        percentage=8 / 9,
    )
    if return_intermediate:
        return (
            song,
            vocals_track,
            instrumentals_track,
            main_vocals_track,
            backup_vocals_track,
            vocals_dereverb_track,
            reverb_track,
            converted_vocals_track,
            vocals_mixed_track,
            instrumentals_shifted_track,
            backup_vocals_shifted_track,
            song_cover,
        )
    return song_cover
