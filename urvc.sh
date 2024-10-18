#!/bin/bash
#
# Launcher script for Ultimate RVC on Debian-based linux systems.
# Currently only supports Ubuntu 22.04 and Ubuntu 24.04.
DEPS_PATH="./dependencies"
ZIP_FILE=dependencies.zip
URL_MAIN="https://huggingface.co/JackismyShephard/ultimate-rvc/resolve/main/$ZIP_FILE"
VENV_PATH="$DEPS_PATH/.venv"
BIN_PATH="$VENV_PATH/bin"
ACTIVATE_PATH="$BIN_PATH/activate"
PYTHON_PATH="$BIN_PATH/python"
GRADIO_PATH="$BIN_PATH/gradio"
main() {
    case $1 in
        install)
            echo "Installing Ultimate RVC"
            sudo apt install -y build-essential software-properties-common unzip
            install_distro_specifics
            install_cuda_124
            sudo add-apt-repository -y ppa:deadsnakes/ppa
            sudo apt install -y python3.12 python3.12-dev python3.12-venv
            sudo apt install -y sox libsox-dev ffmpeg
            rm -rf $DEPS_PATH
            curl -s -LJO $URL_MAIN -o $ZIP_FILE
            unzip -q $ZIP_FILE
            rm $ZIP_FILE
            rm -rf $DEPS_PATH/sox $DEPS_PATH/ffmpeg $DEPS_PATH/miniconda3_11.exe
            python3.12 -m venv $VENV_PATH --upgrade-deps
            # shellcheck disable=SC1090
            . $ACTIVATE_PATH
            pip cache purge
            pip install -r requirements.txt
            python ./src/init.py
            deactivate
            echo
            echo "Ultimate RVC has been installed successfully"
            exit 0
            ;;
        update)
            echo "Updating Ultimate RVC"
            git pull
            rm -rf $VENV_PATH
            python3.12 -m venv $VENV_PATH --upgrade-deps
            # shellcheck disable=SC1090
            . $ACTIVATE_PATH
            pip cache purge
            pip install -r requirements.txt
            deactivate
            echo
            echo "Ultimate RVC has been updated successfully"
            exit 0
            ;;
        run)
            echo "Starting Ultimate RVC"
            shift
            $PYTHON_PATH ./src/app.py "$@"
            exit 0
            ;;
        dev)
            echo "Starting Ultimate RVC in development mode"
            $GRADIO_PATH ./src/app.py --demo-name app
            exit 0
            ;;
        cli)
            echo "Starting Ultimate RVC in CLI mode"
            shift
            $PYTHON_PATH ./src/cli.py "$@"
            exit 0
            ;;
        docs)
            cd src || exit 1
            "../$PYTHON_PATH" -m typer "$2" utils docs --output "$3"
            exit 0
            ;;
        *)
            show_help
            exit 1
            ;;
    esac
}


install_distro_specifics() {
    # shellcheck disable=SC1091
    . /etc/lsb-release
    case $DISTRIB_ID in
        Ubuntu)
            case $DISTRIB_RELEASE in
                24.04)
                    # Add Ubuntu 23.10 repository to sources.list so that we can install cuda 12.1 toolkit

                    # first define the text to append to the file. 
                    # For this we use a heredoc with removal of leading tabs
                    TEXT=$(
                        cat <<- EOF
						
						## Added by Ultimate RVC installer
						Types: deb
						URIs: http://archive.ubuntu.com/ubuntu/
						Suites: lunar
						Components: universe
						Signed-By: /usr/share/keyrings/ubuntu-archive-keyring.gpg
						EOF
                    )
                    FILE=/etc/apt/sources.list.d/ubuntu.sources
                    # Append to file if not already present
                    grep -qxF "## Added by Ultimate RVC installer" $FILE || echo "$TEXT" | sudo tee -a $FILE
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

install_cuda_124() {
    echo "Installing CUDA 12.4"
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
    sudo dpkg -i cuda-keyring_1.1-1_all.deb
    sudo apt-get update
    sudo apt-get -y install cuda-toolkit-12-4
    rm -rf cuda-keyring_1.1-1_all.deb
    echo "CUDA 12.4 has been installed successfully"
}

show_help() {
	cat <<- EOF

	Usage: ./urvc.sh [OPTIONS] COMMAND [ARGS]...

	Commands:
	  install    Install dependencies and set up environment.
	  update     Update application and dependencies.
	  run        Start Ultimate RVC in run mode.
	              Options:
	                --help      Show help message and exit.
	              [more information available, use --help to see all]
	  dev        Start Ultimate RVC in development mode.
	  cli        Start Ultimate RVC in CLI mode.
	              Options:
	                --help      Show help message and exit.
	              [more information available, use --help to see all]
	  docs       Generate documentation using Typer.
	              Args:
	                module     The module to generate documentation for.
	                output     The output directory.
	EOF
}

main "$@"
