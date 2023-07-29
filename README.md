# AICoverGen
An autonomous pipeline to create covers with any RVC v2 trained AI voice from YouTube videos. For developers who may want to add a singing functionality into their AI assistant/chatbot/vtuber, or for people who want to hear their favourite characters sing their favourite song.

Showcase: https://www.youtube.com/watch?v=2qZuE4WM7CM

## Colab notebook

For those without a powerful enough NVIDIA GPU, you may try AICoverGen out using Google Colab.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SociallyIneptWeeb/AICoverGen/blob/main/AICoverGen_colab.ipynb)

For those who want to run this locally, follow the setup guide below.

## Setup

### Install Git and Python

Follow the instructions [here](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) to install Git on your computer. Also follow this [guide](https://realpython.com/installing-python/) to install Python if you haven't already.

### Install ffmpeg

Follow the instructions [here](https://www.hostinger.com/tutorials/how-to-install-ffmpeg) to install ffmpeg on your computer.

### Clone AICoverGen repository

Open a command line window and run these commands to clone this entire repository and install the additional dependencies required.

```
git clone https://github.com/SociallyIneptWeeb/AICoverGen
pip install -r requirements.txt
```

### Download required models

Run the following command to download the required MDXNET vocal separation models and hubert base model.

```
python src/download_models.py
```

### Download RVC models

You may search the [AI Hub Discord](https://discord.gg/aihub) where already trained voice models are available for download.
Unzip (if needed) and transfer the `.pth` and `.index` files to a new folder in the [rvc_models](rvc_models) directory. Each folder should only contain one `.pth` and one `.index` file.

The directory structure should look something like this:
```
├── rvc_models
│   ├── John
│   │   ├── JohnV2.pth
│   │   └── added_IVF2237_Flat_nprobe_1_v2.index
│   ├── May
│   │   ├── May.pth
│   │   └── added_IVF2237_Flat_nprobe_1_v2.index
│   ├── MODELS.txt
│   └── hubert_base.pt
├── mdxnet_models
├── song_output
└── src
 ```

## Usage

To run the AI cover generation pipeline, run the following command.

```
python src/main.py -yt YOUTUBE_LINK -dir MODEL_DIR_NAME -p PITCH_CHANGE
```

- Replace YOUTUBE_LINK with any link to a song on YouTube.
- Replace MODEL_DIR_NAME with the name of the folder in the [rvc_models](rvc_models) directory containing your `.pth` and `.index` files.
- Replace PITCH_CHANGE with 0 for no change in pitch to the AI vocals. Generally use 12 for male to female conversions or -12 for vice-versa.


## Terms of Use

The use of the converted voice for the following purposes is prohibited.

* Criticizing or attacking individuals.

* Advocating for or opposing specific political positions, religions, or ideologies.

* Publicly displaying strongly stimulating expressions without proper zoning.

* Selling of voice models and generated voice clips.

* Impersonation of the original owner of the voice with malicious intentions to harm/hurt others.

* Fraudulent purposes that lead to identity theft or fraudulent phone calls.

## Disclaimer

I am not liable for any direct, indirect, consequential, incidental, or special damages arising out of or in any way connected with the use/misuse or inability to use this software.
