"""
Module which defines the command-line interface for generating a song
cover.
"""

from argparse import ArgumentParser, BooleanOptionalAction

from typing_extra import AudioExt, F0Method, SampleRate

from backend.generate_song_cover import run_pipeline

if __name__ == "__main__":
    parser = ArgumentParser(
        description="Generate a cover of a given song using RVC.",
        add_help=True,
    )
    parser.add_argument(
        "-s",
        "--source",
        type=str,
        required=True,
        help=(
            "A Youtube URL, the path to a local audio file or the path to a song"
            " directory."
        ),
    )
    parser.add_argument(
        "-m",
        "--model-name",
        type=str,
        required=True,
        help="The name of the voice model to use for vocal conversion.",
    )
    parser.add_argument(
        "-no",
        "--n-octaves",
        type=int,
        default=0,
        help=(
            "The number of octaves to pitch-shift the converted vocals by. Use 1 for"
            " male-to-female and -1 for vice-versa."
        ),
    )
    parser.add_argument(
        "-ns",
        "--n-semitones",
        type=int,
        default=0,
        help=(
            "The number of semi-tones to pitch-shift the converted vocals,"
            " instrumentals, and backup vocals by. Altering this slightly reduces sound"
            " quality"
        ),
    )
    parser.add_argument(
        "-f0",
        "--f0-method",
        type=F0Method,
        default=F0Method.RMVPE,
        help=(
            "The method to use for pitch detection during vocal conversion. Best"
            " option is RMVPE (clarity in vocals), then Mangio-Crepe (smoother vocals)."
        ),
    )
    parser.add_argument(
        "-ir",
        "--index-rate",
        type=float,
        default=0.5,
        help=(
            "A decimal number e.g. 0.5, Controls how much of the accent in the voice"
            " model to keep in the converted vocals. Increase to bias the conversion"
            " towards the accent of the voice model."
        ),
    )
    parser.add_argument(
        "-fr",
        "--filter-radius",
        type=int,
        default=3,
        help=(
            "A number between 0 and 7. If >=3: apply median filtering to the pitch"
            " results harvested during vocal conversion. Can help reduce breathiness in"
            " the converted vocals"
        ),
    )
    parser.add_argument(
        "-rms",
        "--rms-mix-rate",
        type=float,
        default=0.25,
        help=(
            "A decimal number e.g. 0.25. Controls how much to mimic the loudness of the"
            " input vocals (0) or a fixed loudness (1) during vocal conversion."
        ),
    )
    parser.add_argument(
        "-pr",
        "--protect",
        type=float,
        default=0.33,
        help=(
            "A decimal number e.g. 0.33. Controls protection of voiceless consonants"
            " and breath sounds during vocal conversion. Decrease to increase"
            " protection at the cost of indexing accuracy. Set to 0.5 to disable. "
        ),
    )
    parser.add_argument(
        "-hl",
        "--hop-length",
        type=int,
        default=128,
        help=(
            " Controls how often the CREPE-based pitch detection algorithm checks for"
            " pitch changes during vocal conversion. Measured in milliseconds. Lower"
            " values lead to longer conversion times and a higher risk of"
            " voice cracks, but better pitch accuracy. Recommended value: 128."
        ),
    )
    parser.add_argument(
        "-rs",
        "--room-size",
        type=float,
        default=0.15,
        help=(
            " The room size of the reverb effect applied to the converted vocals."
            " Increase for longer reverb time. Should be a value between 0 and 1."
        ),
    )
    parser.add_argument(
        "-rwl",
        "--wet-level",
        type=float,
        default=0.2,
        help=(
            "The loudness of the converted vocals with reverb effect applied. Should be"
            " a value between 0 and 1"
        ),
    )
    parser.add_argument(
        "-rdl",
        "--dry-level",
        type=float,
        default=0.8,
        help=(
            "The loudness of the converted vocals wihout reverb effect applied. Should"
            " be a value between 0 and 1"
        ),
    )
    parser.add_argument(
        "-rd",
        "--damping",
        type=float,
        default=0.7,
        help=(
            "The absorption of high frequencies in the reverb effect applied"
            " to the converted vocals. Should be a value between 0 and 1"
        ),
    )
    parser.add_argument(
        "-mg",
        "--main-gain",
        type=int,
        default=0,
        help="The gain to apply to the post-processed vocals. Measured in dB",
    )
    parser.add_argument(
        "-ig",
        "--inst-gain",
        type=int,
        default=0,
        help="The gain to apply to the pitch-shifted instrumentals. Measured in dB",
    )
    parser.add_argument(
        "-bg",
        "--backup-gain",
        type=int,
        default=0,
        help="The gain to apply to the pitch-shifted backup vocals. Measured in dB",
    )

    parser.add_argument(
        "-osr",
        "--output-sr",
        type=int,
        default=SampleRate.HZ_44100,
        help="The sample rate of the song cover.",
    )
    parser.add_argument(
        "-of",
        "--output-format",
        type=AudioExt,
        default=AudioExt.MP3,
        help="The audio format of the song cover.",
    )
    parser.add_argument(
        "-on",
        "--output-name",
        type=str,
        default=None,
        help="The name of the song cover.",
    )
    parser.add_argument(
        "-ri",
        "--return-intermediate",
        action=BooleanOptionalAction,
        default=False,
        help=(
            "Whether to return the paths of any intermediate audio files generated"
            " during execution of the pipeline."
        ),
    )
    args = parser.parse_args()

    song_cover_path = run_pipeline(
        source=args.source,
        model_name=args.model_name,
        n_octaves=args.n_octaves,
        n_semitones=args.n_semitones,
        f0_method=args.f0_method,
        index_rate=args.index_rate,
        filter_radius=args.filter_radius,
        rms_mix_rate=args.rms_mix_rate,
        protect=args.protect,
        hop_length=args.hop_length,
        room_size=args.room_size,
        wet_level=args.wet_level,
        dry_level=args.dry_level,
        damping=args.damping,
        main_gain=args.main_gain,
        inst_gain=args.inst_gain,
        backup_gain=args.backup_gain,
        output_sr=args.output_sr,
        output_format=args.output_format,
        output_name=args.output_name,
        return_intermediate=args.return_intermediate,
        progress_bar=None,
    )
    print(f"[+] Cover generated at {song_cover_path}")  # noqa: T201
