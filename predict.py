# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import os
import sys
import shutil
import zipfile
import urllib.request
from argparse import Namespace
from cog import BasePredictor, Input, Path as CogPath

sys.path.insert(0, os.path.abspath("src"))

import main as m


def download_online_model(url, dir_name):
    print(f"[~] Downloading voice model with name {dir_name}...")
    zip_name = url.split("/")[-1]
    extraction_folder = os.path.join(m.rvc_models_dir, dir_name)
    if os.path.exists(extraction_folder):
        print(f"Voice model directory {dir_name} already exists! Skipping download.")
        return

    if "pixeldrain.com" in url:
        url = f"https://pixeldrain.com/api/file/{zip_name}"

    urllib.request.urlretrieve(url, zip_name)

    print("[~] Extracting zip...")
    with zipfile.ZipFile(zip_name, "r") as zip_ref:
        for member in zip_ref.infolist():
            # skip directories
            if member.is_dir():
                continue

            # create target directory if it does not exist
            os.makedirs(extraction_folder, exist_ok=True)

            # extract only files directly to extraction_folder
            with zip_ref.open(member) as source, open(
                os.path.join(extraction_folder, os.path.basename(member.filename)), "wb"
            ) as target:
                shutil.copyfileobj(source, target)
    print(f"[+] {dir_name} Model successfully downloaded!")


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        pass

    def predict(
        self,
        song_input: CogPath = Input(
            description="Upload your audio file here.",
            default=None,
        ),
        rvc_model: str = Input(
            description="RVC model for a specific voice. If using a custom model, this should match the name of the downloaded model. If a 'custom_rvc_model_download_url' is provided, this will be automatically set to the name of the downloaded model.",
            default="Squidward",
            choices=[
                "Squidward",
                "MrKrabs",
                "Plankton",
                "Drake",
                "Vader",
                "Trump",
                "Biden",
                "Obama",
                "Guitar",
                "Voilin",
                "CUSTOM",
                "SamA",  # TODO REMOVE THIS
            ],
        ),
        custom_rvc_model_download_url: str = Input(
            description="URL to download a custom RVC model. If provided, the model will be downloaded (if it doesn't already exist) and used for prediction, regardless of the 'rvc_model' value.",
            default=None,
        ),
        pitch_change: str = Input(
            description="Adjust pitch of AI vocals. Options: `no-change`, `male-to-female`, `female-to-male`.",
            default="no-change",
            choices=["no-change", "male-to-female", "female-to-male"],
        ),
        index_rate: float = Input(
            description="Control how much of the AI's accent to leave in the vocals.",
            default=0.5,
            ge=0,
            le=1,
        ),
        filter_radius: int = Input(
            description="If >=3: apply median filtering median filtering to the harvested pitch results.",
            default=3,
            ge=0,
            le=7,
        ),
        rms_mix_rate: float = Input(
            description="Control how much to use the original vocal's loudness (0) or a fixed loudness (1).",
            default=0.25,
            ge=0,
            le=1,
        ),
        pitch_detection_algorithm: str = Input(
            description="Best option is rmvpe (clarity in vocals), then mangio-crepe (smoother vocals).",
            default="rmvpe",
            choices=["rmvpe", "mangio-crepe"],
        ),
        crepe_hop_length: int = Input(
            description="When `pitch_detection_algo` is set to `mangio-crepe`, this controls how often it checks for pitch changes in milliseconds. Lower values lead to longer conversions and higher risk of voice cracks, but better pitch accuracy.",
            default=128,
        ),
        protect: float = Input(
            description="Control how much of the original vocals' breath and voiceless consonants to leave in the AI vocals. Set 0.5 to disable.",
            default=0.33,
            ge=0,
            le=0.5,
        ),
        main_vocals_volume_change: float = Input(
            description="Control volume of main AI vocals. Use -3 to decrease the volume by 3 decibels, or 3 to increase the volume by 3 decibels.",
            default=0,
        ),
        backup_vocals_volume_change: float = Input(
            description="Control volume of backup AI vocals.",
            default=0,
        ),
        instrumental_volume_change: float = Input(
            description="Control volume of the background music/instrumentals.",
            default=0,
        ),
        pitch_change_all: float = Input(
            description="Change pitch/key of background music, backup vocals and AI vocals in semitones. Reduces sound quality slightly.",
            default=0,
        ),
        reverb_size: float = Input(
            description="The larger the room, the longer the reverb time.",
            default=0.15,
            ge=0,
            le=1,
        ),
        reverb_wetness: float = Input(
            description="Level of AI vocals with reverb.",
            default=0.2,
            ge=0,
            le=1,
        ),
        reverb_dryness: float = Input(
            description="Level of AI vocals without reverb.",
            default=0.8,
            ge=0,
            le=1,
        ),
        reverb_damping: float = Input(
            description="Absorption of high frequencies in the reverb.",
            default=0.7,
            ge=0,
            le=1,
        ),
        output_format: str = Input(
            description="wav for best quality and large file size, mp3 for decent quality and small file size.",
            default="mp3",
            choices=["mp3", "wav"],
        ),
    ) -> CogPath:
        """
        Runs a single prediction on the model.

        Required Parameters:
            song_input (CogPath): Upload your audio file here.
            rvc_model (str): RVC model for a specific voice. Default is "Squidward". If a 'custom_rvc_model_download_url' is provided, this will be automatically set to the name of the downloaded model.
            pitch_change (float): Change pitch of AI vocals in octaves. Set to 0 for no change. Generally, use 1 for male to female conversions and -1 for vice-versa.

        Optional Parameters:
            custom_rvc_model_download_url (str): URL to download a custom RVC model. If provided, the model will be downloaded (if it doesn't already exist) and used for prediction, regardless of the 'rvc_model' value. Defaults to None.
            index_rate (float): Control how much of the AI's accent to leave in the vocals. 0 <= INDEX_RATE <= 1. Defaults to 0.5.
            filter_radius (int): If >=3: apply median filtering median filtering to the harvested pitch results. 0 <= FILTER_RADIUS <= 7. Defaults to 3.
            rms_mix_rate (float): Control how much to use the original vocal's loudness (0) or a fixed loudness (1). 0 <= RMS_MIX_RATE <= 1. Defaults to 0.25.
            pitch_detection_algorithm (str): Best option is rmvpe (clarity in vocals), then mangio-crepe (smoother vocals). Defaults to "rmvpe".
            crepe_hop_length (int): Controls how often it checks for pitch changes in milliseconds when using mangio-crepe algo specifically. Lower values leads to longer conversions and higher risk of voice cracks, but better pitch accuracy. Defaults to 128.
            protect (float): Control how much of the original vocals' breath and voiceless consonants to leave in the AI vocals. Set 0.5 to disable. 0 <= PROTECT <= 0.5. Defaults to 0.33.
            main_vocals_volume_change (float): Control volume of main AI vocals. Use -3 to decrease the volume by 3 decibels, or 3 to increase the volume by 3 decibels. Defaults to 0.
            backup_vocals_volume_change (float): Control volume of backup AI vocals. Defaults to 0.
            instrumental_volume_change (float): Control volume of the background music/instrumentals. Defaults to 0.
            pitch_change_all (float): Change pitch/key of background music, backup vocals and AI vocals in semitones. Reduces sound quality slightly. Defaults to 0.
            reverb_size (float): The larger the room, the longer the reverb time. 0 <= REVERB_SIZE <= 1. Defaults to 0.15.
            reverb_wetness (float): Level of AI vocals with reverb. 0 <= REVERB_WETNESS <= 1. Defaults to 0.2.
            reverb_dryness (float): Level of AI vocals without reverb. 0 <= REVERB_DRYNESS <= 1. Defaults to 0.8.
            reverb_damping (float): Absorption of high frequencies in the reverb. 0 <= REVERB_DAMPING <= 1. Defaults to 0.7.
            output_format (str): wav for best quality and large file size, mp3 for decent quality and small file size. Defaults to "mp3".

        Returns:
            CogPath: The output path of the generated audio file.
        """

        if custom_rvc_model_download_url:
            custom_rvc_model_download_name = urllib.parse.unquote(
                custom_rvc_model_download_url.split("/")[-1]
            )
            custom_rvc_model_download_name = os.path.splitext(
                custom_rvc_model_download_name
            )[0]
            print(
                f"[!] The model will be downloaded as '{custom_rvc_model_download_name}'."
            )
            download_online_model(
                url=custom_rvc_model_download_url,
                dir_name=custom_rvc_model_download_name,
            )
            rvc_model = custom_rvc_model_download_name
        else:
            print(
                "[!] Since URL was provided, we will try to download the model and use it (even if `rvc_model` is not set to 'CUSTOM')."
            )

        # Convert pitch_change from string to numerical value for processing
        # 0 for no change, 1 for male to female, -1 for female to male
        if pitch_change == "no-change":
            pitch_change = 0
        elif pitch_change == "male-to-female":
            pitch_change = 1
        else:  # pitch_change == "female-to-male"
            pitch_change = -1

        args = Namespace(
            song_input=str(song_input),
            rvc_dirname=(model_dir_name := rvc_model),
            pitch_change=pitch_change,
            keep_files=(keep_files := False),
            index_rate=index_rate,
            filter_radius=filter_radius,
            rms_mix_rate=rms_mix_rate,
            pitch_detection_algo=pitch_detection_algorithm,
            crepe_hop_length=crepe_hop_length,
            protect=protect,
            main_vol=main_vocals_volume_change,
            backup_vol=backup_vocals_volume_change,
            inst_vol=instrumental_volume_change,
            pitch_change_all=pitch_change_all,
            reverb_size=reverb_size,
            reverb_wetness=reverb_wetness,
            reverb_dryness=reverb_dryness,
            reverb_damping=reverb_damping,
            output_format=output_format,
        )

        rvc_dirname = args.rvc_dirname
        if not os.path.exists(os.path.join(m.rvc_models_dir, rvc_dirname)):
            raise Exception(
                f"The folder {os.path.join(m.rvc_models_dir, rvc_dirname)} does not exist."
            )

        cover_path = m.song_cover_pipeline(
            args.song_input,
            rvc_dirname,
            args.pitch_change,
            args.keep_files,
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
        )
        print(f"[+] Cover generated at {cover_path}")

        # Return the output path
        return CogPath(cover_path)
