#!/bin/bash
#
# Launcher script for Ultimate RVC on Debian-based linux systems.
# Currently only supports Ubuntu 22.04 and Ubuntu 24.04.

main() {
    case $1 in
        install)
            echo "Installing Ultimate RVC"
            sudo apt install -y build-essential software-properties-common
            install_distro_specifics
            install_cuda_118
            sudo add-apt-repository -y ppa:deadsnakes/ppa
            sudo apt install -y python3.11 python3.11-dev python3.11-venv
            sudo apt install -y sox libsox-dev ffmpeg

            python3.11 -m venv .venv --upgrade-deps 
            . .venv/bin/activate
            pip cache purge
            pip install -r requirements.txt
            pip install faiss-cpu==1.7.3
            pip uninstall torch torchaudio -y
            pip install torch==2.1.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
            python ./src/init.py
            deactivate
            echo
            echo "Ultimate RVC has been installed successfully"
            exit 0
            ;;
        run)
            echo "Starting Ultimate RVC"
            ./.venv/bin/python ./src/app.py
            exit 0
            ;;
        update)
            echo "Updating Ultimate RVC"
            git pull
            rm -rf .venv
            python3.11 -m venv .venv --upgrade-deps
            . .venv/bin/activate
            pip cache purge
            pip install -r requirements.txt
            pip install faiss-cpu==1.7.3
            pip uninstall torch torchaudio -y
            pip install torch==2.1.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
            deactivate
            echo
            echo "Ultimate RVC has been updated successfully"
            exit 0
            ;;
        dev)
            echo "Starting Ultimate RVC in development mode"
            ./.venv/bin/gradio ./src/app.py --demo-name app
            exit 0
            ;;
        *)
            echo "Usage ./urvc.sh [install|run|update|dev]"
            exit 1
            ;;
    esac
}

install_distro_specifics() {
    . /etc/lsb-release
    case $DISTRIB_ID in
        Ubuntu)
            case $DISTRIB_RELEASE in
                24.04)
                    # Add Ubuntu 23.10 repository to sources.list so that we can install cuda 11.8 toolkit

                    # sed command removes leading whitespace from subsequent lines
                    TEXT=$(sed 's/^[[:space:]]*//' <<< "\
                    ##
                    ## Added by Ultimate RVC installer
                    Types: deb
                    URIs: http://archive.ubuntu.com/ubuntu/
                    Suites: lunar
                    Components: universe
                    Signed-By: /usr/share/keyrings/ubuntu-archive-keyring.gpg"
                    )
                    FILE=/etc/apt/sources.list.d/ubuntu.sources
                    # Append to file if not already present
                    grep -qxF "## Added by Ultimate RVC installer" "$FILE" || echo "$TEXT" | sudo tee -a "$FILE"
                    sudo apt update
                    ;;
                22.04)
                    sudo add-apt-repository -y ppa:ubuntuhandbook1/ffmpeg6
                    ;;
                *)
                    echo "Unsupported Ubuntu version"
                    exit 1
                    ;;
            esac
            ;;
        *)
            echo "Unsupported debian distribution"
            exit 1
            ;;
    esac
}

install_cuda_118() {
    echo "Installing CUDA 11.8"
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
    sudo dpkg -i cuda-keyring_1.0-1_all.deb
    sudo apt-get update
    sudo apt-get -y install cuda-toolkit-11-8
    rm -rf cuda-keyring_1.0-1_all.deb
    echo "CUDA 11.8 has been installed successfully"
}

main $@
