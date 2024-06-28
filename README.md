# Ultimate RVC
An autonomous pipeline to create covers with any RVC v2 trained AI voice from YouTube videos or a local audio file. For developers who may want to add a singing functionality into their AI assistant/chatbot/vtuber, or for people who want to hear their favourite characters sing their favourite song.

Showcase: TBA

Setup Guide: TBA

![](images/webui_generate.png?raw=true)

Ultimate RVC is under constant development and testing, but you can try it out right locally!

## Changelog

TBA

## Setup

### Install Git and Python

Follow the instructions [here](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) to install Git on your computer. Also follow this [guide](https://realpython.com/installing-python/) to install Python **VERSION 3.9** if you haven't already. Using other versions of Python may result in dependency conflicts.

### Install ffmpeg

Follow the instructions [here](https://www.hostinger.com/tutorials/how-to-install-ffmpeg) to install ffmpeg on your computer.

### Install sox

Follow the instructions [here](https://www.tutorialexample.com/a-step-guide-to-install-sox-sound-exchange-on-windows-10-python-tutorial/) to install sox and add it to your Windows path environment.

### Fairseq dependencies

In order to run fairseq on Windows you might need to install C++ build tools from [Visual Studio](https://visualstudio.microsoft.com/). Windows 11 SDK and MSVC v14 should be sufficient in most cases.

### Clone Ultimate RVC repository

Open a command line window and run these commands to clone this entire repository and install the additional dependencies required.

```
git clone https://github.com/JackismyShephard/ultimate-rvc
cd ultimate-rvc
py -3.9 -m venv .venv
```
`.venv\Scripts\activate` (Windows) or `source .venv/bin/activate` (Unix)
```
pip install -r requirements.txt
```

### Download required models

Run the following command to download the required MDXNET vocal separation models and hubert base model.

```
python ./src/init.py
```

## Update Ultimate RVC to latest version

Install and pull any new requirements and changes by opening a command line window in the `ultimate-rvc` directory and running the following commands.

`.venv\Scripts\activate` (Windows) or `source .venv/bin/activate` (Unix)

```
pip install -r requirements.txt
git pull
```

## App

To run the Ultimate RVC web app, run the following command.

```
python ./src/app.py
```

| Flag                                       | Description |
|--------------------------------------------|-------------|
| `-h`, `--help`                             | Show this help message and exit. |
| `--share`                                  | Create a public URL. This is useful for running the web app on Google Colab. |
| `--listen`                                 | Make the web app reachable from your local network. |
| `--listen-host LISTEN_HOST`                | The hostname that the server will use. |
| `--listen-port LISTEN_PORT`                | The listening port that the server will use. |

Once the following output message `Running on local URL:  http://127.0.0.1:7860` appears, you can click on the link to open a tab with the web app.

### Download RVC models

![](images/webui_dl_model.png?raw=true)

Navigate to the `Download model` subtab under the `Manage models` tab, and paste the download link to the RVC model and give it a unique name.
You may search the [AI Hub Discord](https://discord.gg/aihub) where already trained voice models are available for download. You may refer to the examples for how the download link should look like.
The downloaded zip file should contain the .pth model file and an optional .index file.

Once the 2 input fields are filled in, simply click `Download`! Once the output message says `[NAME] Model successfully downloaded!`, you should be able to use it in the `Generate song covers` tab!

### Upload RVC models

![](images/webui_upload_model.png?raw=true)

For people who have trained RVC v2 models locally and would like to use them for AI cover generations.
Navigate to the `Upload model` subtab under the `Manage models` tab, and follow the instructions.
Once the output message says `[NAME] Model successfully uploaded!`, you should be able to use it in the `Generate` tab after clicking the refresh models button!

### Delete RVC models

TBA

### Running the pipeline

![](images/webui-front-page?raw=true)

- From the Voice model dropdown menu, select the voice model to use.
- In the song input field, copy and paste the link to any song on YouTube, the full path to a local audio file, or select a cached input song.
- Pitch should be set to either -12, 0, or 12 depending on the original vocals and the RVC AI modal. This ensures the voice is not *out of tune*.
- Other advanced options for vocal conversion, audio mixing and etc. can be viewed by clicking the  appropriate accordion arrow to expand.

Once all options are filled in, click `Generate` and the AI generated cover should appear in a less than a few minutes depending on your GPU.

### Running each step of the pipeline separately
TBA

## CLI
TBA


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
