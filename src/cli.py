import argparse
from backend.generate_song_cover import run_pipeline


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
        help="Link to a song on YouTube, the full path of a local audio file or a cached input song",
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
        help="Whether to keep song directory with intermediate audio files generated during song cover generation.",
    )
    args = parser.parse_args()

    rvc_dirname = args.rvc_dirname

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
        progress_bar=None,
    )
    print(f"[+] Cover generated at {song_cover_path}")
