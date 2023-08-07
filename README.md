# AICoverGen
An autonomous pipeline to create covers with any RVC v2 trained AI voice from YouTube videos or a local audio file. For developers who may want to add a singing functionality into their AI assistant/chatbot/vtuber, or for people who want to hear their favourite characters sing their favourite song.

Showcase: https://www.youtube.com/watch?v=2qZuE4WM7CM

Setup Guide: https://www.youtube.com/watch?v=pdlhk4vVHQk

![](images/webui_generate.png?raw=true)

WebUI is under constant development and testing, but you can try it out right now on both local and colab!

## Changelog

- WebUI for easier conversions and downloading of voice models
- Support for cover generations from a local audio file
- Option to keep intermediate files generated. e.g. Isolated vocals/instrumentals
- Download suggested public voice models from table with search/tag filters
- Support for Pixeldrain download links for voice models
- Implement new rmvpe pitch extraction technique for faster and higher quality vocal conversions
- Volume control for AI main vocals, backup vocals and instrumentals
- Index Rate for Voice conversion
- Reverb Control for AI main vocals

## Colab notebook

For those without a powerful enough NVIDIA GPU, you may try AICoverGen out using Google Colab.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SociallyIneptWeeb/AICoverGen/blob/main/AICoverGen_colab.ipynb)

For those who want to run this locally, follow the setup guide below.

## Setup

### Install Git and Python

Follow the instructions [here](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) to install Git on your computer. Also follow this [guide](https://realpython.com/installing-python/) to install Python **VERSION 3.9** if you haven't already. Using other versions of Python may result in dependency conflicts.

### Install ffmpeg

Follow the instructions [here](https://www.hostinger.com/tutorials/how-to-install-ffmpeg) to install ffmpeg on your computer.

### Clone AICoverGen repository

Open a command line window and run these commands to clone this entire repository and install the additional dependencies required.

```
git clone https://github.com/SociallyIneptWeeb/AICoverGen
cd AICoverGen
pip install -r requirements.txt
```

### Download required models

Run the following command to download the required MDXNET vocal separation models and hubert base model.

```
python src/download_models.py
```


## Usage with WebUI

To run the AICoverGen WebUI, run the following command.

```
python src/webui.py
```

Once the following output message `Running on local URL:  http://127.0.0.1:7860` appears, you can click on the link to open a tab with the WebUI.

### Download RVC models via WebUI

![](images/webui_dl_model.png?raw=true)

Navigate to the `Download model` tab, and paste the download link to the RVC model and give it a unique name.
You may search the [AI Hub Discord](https://discord.gg/aihub) where already trained voice models are available for download. You may refer to the examples for how the download link should look like.
The downloaded zip file should contain the .pth model file and an optional .index file.

Once the 2 input fields are filled in, simply click `Download`! Once the output message says `[NAME] Model successfully downloaded!`, you should be able to use it in the `Generate` tab!


### Running the pipeline via WebUI

![](images/webui_generate.png?raw=true)

- From the Voice Models dropdown menu, select the voice model to use. Click `Update` if you added the files manually to the [rvc_models](rvc_models) directory to refresh the list.
- In the song input field, copy and paste the link to any song on YouTube or the full path to a local audio file.
- Pitch should be set to either -12, 0, or 12 depending on the original vocals and the RVC AI modal. This ensures the voice is not *out of tune*.
- Other advanced options for Voice conversion and audio mixing can be viewed by clicking the accordion arrow to expand.

Once all Main Options are filled in, click `Generate` and the AI generated cover should appear in a less than a few minutes depending on your GPU.

## Usage with CLI

### Manual Download of RVC models

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

### Running the pipeline

To run the AI cover generation pipeline using the command line, run the following command.

```
python src/main.py [-h] -i SONG_INPUT -dir RVC_DIRNAME -p PITCH_CHANGE [-k | --keep-files | --no-keep-files] [-ir INDEX_RATE] [-mv MAIN_VOL] [-bv BACKUP_VOL] [-iv INST_VOL] [-rsize REVERB_SIZE] [-rwet REVERB_WETNESS] [-rdry REVERB_DRYNESS] [-rdamp REVERB_DAMPING]
```

| Flag                                       | Description |
|--------------------------------------------|-------------|
| `-h`, `--help`                             | Show this help message and exit. |
| `-i SONG_INPUT`                            | Link to a song on YouTube or path to a local audio file. Should be enclosed in double quotes for Windows and single quotes for Unix-like systems. |
| `-dir MODEL_DIR_NAME`                      | Name of folder in [rvc_models](rvc_models) directory containing your `.pth` and `.index` files for a specific voice. |
| `-p PITCH_CHANGE`                          | Generally use 12 for male to female conversions or -12 for vice-versa. Set to 0 for no change in pitch for the AI vocals. |
| `-k`                                       | Optional. Can be added to keep all intermediate audio files generated. e.g. Isolated AI vocals/instrumentals. Leave out to save space. |
| `-ir INDEX_RATE`                           | Optional. Default 0.5. Control how much of the AI's accent to leave in the vocals. 0 <= INDEX_RATE <= 1. |
| `-mv MAIN_VOCALS_VOLUME_CHANGE`            | Optional. Default 0. Control volume of main AI vocals. Use -3 to decrease the volume by 3 decibels, or 3 to increase the volume by 3 decibels. |
| `-bv BACKUP_VOCALS_VOLUME_CHANGE`          | Optional. Default 0. Control volume of backup AI vocals. |
| `-iv INSTRUMENTAL_VOLUME_CHANGE`           | Optional. Default 0. Control volume of the background music/instrumentals. |
| `-rsize REVERB_SIZE`                       | Optional. Default 0.15. The larger the room, the longer the reverb time. 0 <= REVERB_SIZE <= 1. |
| `-rwet REVERB_WETNESS`                     | Optional. Default 0.2. Level of AI vocals with reverb. 0 <= REVERB_WETNESS <= 1. |
| `-rdry REVERB_DRYNESS`                     | Optional. Default 0.8. Level of AI vocals without reverb. 0 <= REVERB_DRYNESS <= 1. |
| `-rdamp REVERB_DAMPING`                    | Optional. Default 0.7. Absorption of high frequencies in the reverb. 0 <= REVERB_DAMPING <= 1. |


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
