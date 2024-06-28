import requests
import os
from common import MDXNET_MODELS_DIR, RVC_MODELS_DIR

MDX_DOWNLOAD_LINK = (
    "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/"
)
RVC_DOWNLOAD_LINK = "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/"


def dl_model(link, model_name, dir_name):
    with requests.get(f"{link}{model_name}") as r:
        r.raise_for_status()
        with open(os.path.join(dir_name, model_name), "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)


if __name__ == "__main__":
    mdx_model_names = [
        "UVR-MDX-NET-Voc_FT.onnx",
        "UVR_MDXNET_KARA_2.onnx",
        "Reverb_HQ_By_FoxJoy.onnx",
    ]
    for model in mdx_model_names:
        print(f"Downloading {model}...")
        dl_model(MDX_DOWNLOAD_LINK, model, MDXNET_MODELS_DIR)

    rvc_model_names = ["hubert_base.pt", "rmvpe.pt"]
    for model in rvc_model_names:
        print(f"Downloading {model}...")
        dl_model(RVC_DOWNLOAD_LINK, model, RVC_MODELS_DIR)

    print("All models downloaded!")
