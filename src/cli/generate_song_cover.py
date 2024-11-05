"""
Module which defines the command-line interface for generating a song
cover.
"""

from typing import Annotated

import typer
from rich import print as rprint
from rich.panel import Panel
from rich.table import Table

from typing_extra import AudioExt, F0Method

from core.generate_song_cover import run_pipeline as _run_pipeline

app = typer.Typer(
    name="song-cover",
    no_args_is_help=True,
    help="Generate song covers",
    rich_markup_mode="markdown",
)


def complete_name(incomplete: str, enumeration: list[str]) -> list[str]:
    """
    Return a list of names that start with the incomplete string.

    Parameters
    ----------
    incomplete : str
        The incomplete string to complete.
    enumeration : list[str]
        The list of names to complete from.

    Returns
    -------
    list[str]
        The list of names that start with the incomplete string.

    """
    return [name for name in list(enumeration) if name.startswith(incomplete)]


def complete_audio_ext(incomplete: str) -> list[str]:
    """
    Return a list of audio extensions that start with the incomplete
    string.

    Parameters
    ----------
    incomplete : str
        The incomplete string to complete.

    Returns
    -------
    list[str]
        The list of audio extensions that start with the incomplete
        string.

    """
    return complete_name(incomplete, list(AudioExt))


def complete_f0_method(incomplete: str) -> list[str]:
    """
    Return a list of F0 methods that start with the incomplete string.

    Parameters
    ----------
    incomplete : str
        The incomplete string to complete.

    Returns
    -------
    list[str]
        The list of F0 methods that start with the incomplete string.

    """
    return complete_name(incomplete, list(F0Method))


@app.command(no_args_is_help=True)
def run_pipeline(
    source: Annotated[
        str,
        typer.Argument(
            help=(
                "A Youtube URL, the path to a local audio file or the path to a"
                " song directory."
            ),
        ),
    ],
    model_name: Annotated[
        str,
        typer.Argument(help="The name of the voice model to use for vocal conversion."),
    ],
    n_octaves: Annotated[
        int,
        typer.Option(
            rich_help_panel="Vocal Conversion Options",
            help=(
                "The number of octaves to pitch-shift the converted vocals by.Use 1 for"
                " male-to-female and -1 for vice-versa."
            ),
        ),
    ] = 0,
    n_semitones: Annotated[
        int,
        typer.Option(
            rich_help_panel="Vocal Conversion Options",
            help=(
                "The number of semi-tones to pitch-shift the converted vocals,"
                " instrumentals, and backup vocals by. Altering this slightly reduces"
                " sound quality"
            ),
        ),
    ] = 0,
    f0_method: Annotated[
        F0Method,
        typer.Option(
            case_sensitive=False,
            autocompletion=complete_f0_method,
            rich_help_panel="Vocal Conversion Options",
            help=(
                "The method to use for pitch detection during vocal conversion. Best"
                " option is RMVPE (clarity in vocals), then Mangio-Crepe (smoother"
                " vocals)."
            ),
        ),
    ] = F0Method.RMVPE,
    index_rate: Annotated[
        float,
        typer.Option(
            min=0,
            max=1,
            rich_help_panel="Vocal Conversion Options",
            help=(
                "A decimal number e.g. 0.5, Controls how much of the accent in the"
                " voice model to keep in the converted vocals. Increase to bias the"
                " conversion towards the accent of the voice model."
            ),
        ),
    ] = 0.5,
    filter_radius: Annotated[
        int,
        typer.Option(
            min=0,
            max=7,
            rich_help_panel="Vocal Conversion Options",
            help=(
                "A number between 0 and 7. If >=3: apply median filtering to the pitch"
                " results harvested during vocal conversion. Can help reduce"
                " breathiness in the converted vocals."
            ),
        ),
    ] = 3,
    rms_mix_rate: Annotated[
        float,
        typer.Option(
            min=0,
            max=1,
            rich_help_panel="Vocal Conversion Options",
            help=(
                "A decimal number e.g. 0.25. Controls how much to mimic the loudness of"
                " the input vocals (0) or a fixed loudness (1) during vocal conversion."
            ),
        ),
    ] = 0.25,
    protect: Annotated[
        float,
        typer.Option(
            min=0,
            max=0.5,
            rich_help_panel="Vocal Conversion Options",
            help=(
                "A decimal number e.g. 0.33. Controls protection of voiceless"
                " consonants and breath sounds during vocal conversion. Decrease to"
                " increase protection at the cost of indexing accuracy. Set to 0.5 to"
                " disable."
            ),
        ),
    ] = 0.33,
    hop_length: Annotated[
        int,
        typer.Option(
            rich_help_panel="Vocal Conversion Options",
            help=(
                "Controls how often the CREPE-based pitch detection algorithm checks"
                " for pitch changes during vocal conversion. Measured in milliseconds."
                " Lower values lead to longer conversion times and a higher risk of"
                " voice cracks, but better pitch accuracy. Recommended value: 128."
            ),
        ),
    ] = 128,
    room_size: Annotated[
        float,
        typer.Option(
            min=0,
            max=1,
            rich_help_panel="Vocal Post-processing Options",
            help=(
                "The room size of the reverb effect applied to the converted vocals."
                " Increase for longer reverb time. Should be a value between 0 and 1."
            ),
        ),
    ] = 0.15,
    wet_level: Annotated[
        float,
        typer.Option(
            min=0,
            max=1,
            rich_help_panel="Vocal Post-processing Options",
            help=(
                "The loudness of the converted vocals with reverb effect applied."
                " Should be a value between 0 and 1"
            ),
        ),
    ] = 0.2,
    dry_level: Annotated[
        float,
        typer.Option(
            min=0,
            max=1,
            rich_help_panel="Vocal Post-processing Options",
            help=(
                "The loudness of the converted vocals wihout reverb effect applied."
                " Should be a value between 0 and 1."
            ),
        ),
    ] = 0.8,
    damping: Annotated[
        float,
        typer.Option(
            min=0,
            max=1,
            rich_help_panel="Vocal Post-processing Options",
            help=(
                "The absorption of high frequencies in the reverb effect applied to the"
                " converted vocals. Should be a value between 0 and 1."
            ),
        ),
    ] = 0.7,
    main_gain: Annotated[
        int,
        typer.Option(
            rich_help_panel="Audio Mixing Options",
            help="The gain to apply to the post-processed vocals. Measured in dB.",
        ),
    ] = 0,
    inst_gain: Annotated[
        int,
        typer.Option(
            rich_help_panel="Audio Mixing Options",
            help=(
                "The gain to apply to the pitch-shifted instrumentals. Measured in dB."
            ),
        ),
    ] = 0,
    backup_gain: Annotated[
        int,
        typer.Option(
            rich_help_panel="Audio Mixing Options",
            help=(
                "The gain to apply to the pitch-shifted backup vocals. Measured in dB."
            ),
        ),
    ] = 0,
    output_sr: Annotated[
        int,
        typer.Option(
            rich_help_panel="Audio Mixing Options",
            help="The sample rate of the song cover.",
        ),
    ] = 44100,
    output_format: Annotated[
        AudioExt,
        typer.Option(
            case_sensitive=False,
            autocompletion=complete_audio_ext,
            rich_help_panel="Audio Mixing Options",
            help="The audio format of the song cover.",
        ),
    ] = AudioExt.MP3,
    output_name: Annotated[
        str | None,
        typer.Option(
            rich_help_panel="Audio Mixing Options",
            help="The name of the song cover.",
        ),
    ] = None,
) -> None:
    """Run the song cover generation pipeline."""
    [song_cover_path, *intermediate_audio_file_paths] = _run_pipeline(
        source=source,
        model_name=model_name,
        n_octaves=n_octaves,
        n_semitones=n_semitones,
        f0_method=f0_method,
        index_rate=index_rate,
        filter_radius=filter_radius,
        rms_mix_rate=rms_mix_rate,
        protect=protect,
        hop_length=hop_length,
        room_size=room_size,
        wet_level=wet_level,
        dry_level=dry_level,
        damping=damping,
        main_gain=main_gain,
        inst_gain=inst_gain,
        backup_gain=backup_gain,
        output_sr=output_sr,
        output_format=output_format,
        output_name=output_name,
        progress_bar=None,
    )
    table = Table()
    table.add_column("Type")
    table.add_column("Path")
    for name, path in zip(
        [
            "Song",
            "Vocals",
            "Instrumentals",
            "Main vocals",
            "Backup vocals",
            "De-reverbed main vocals",
            "Main vocals reverb",
            "Converted vocals",
            "Post-processed vocals",
            "Pitch-shifted instrumentals",
            "Pitch-shifted backup vocals",
        ],
        intermediate_audio_file_paths,
        strict=True,
    ):
        table.add_row(name, f"[green]{path}")
    rprint("[+] Song cover succesfully generated!")
    rprint(Panel(f"[green]{song_cover_path}", title="Song Cover Path"))
    rprint(Panel(table, title="Intermediate Audio Files"))
