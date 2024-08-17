import requests
import os
from common import RVC_MODELS_DIR

RVC_DOWNLOAD_LINK = "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/"


def dl_model(link: str, model_name: str, dir_name: str) -> None:
    with requests.get(f"{link}{model_name}") as r:
        r.raise_for_status()
        with open(os.path.join(dir_name, model_name), "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)


if __name__ == "__main__":

    rvc_model_names = ["hubert_base.pt", "rmvpe.pt"]
    for model in rvc_model_names:
        print(f"Downloading {model}...")
        dl_model(RVC_DOWNLOAD_LINK, model, RVC_MODELS_DIR)

    print("All models downloaded!")
