@echo off
setlocal enabledelayedexpansion
title Ultimate RVC launcher

set ROOT=%cd%

rem huggingface model hub variables
set ZIP_FILE=dependencies.zip
set URL_MAIN=https://huggingface.co/JackismyShephard/ultimate-rvc/resolve/main/%ZIP_FILE%

rem dependencies variables
set DEPS_DIR=%ROOT%\dependencies
set DEPS_ZIP=%ROOT%\%ZIP_FILE%

rem conda variables
set CONDA_ROOT=%UserProfile%\Miniconda3
set CONDA_SCRIPT_DIR=%CONDA_ROOT%\Scripts
set CONDA_INSTALLER=%DEPS_DIR%\miniconda3_11.exe

rem virtual environment variables
set VENV_ROOT=%DEPS_DIR%\.venv
set VENV_SCRIPT_DIR=%VENV_ROOT%\Scripts
set PYTHON_PATH=%VENV_ROOT%\python.exe
set GRADIO_PATH=%VENV_SCRIPT_DIR%\gradio.exe
set REQUIREMENTS_FILE=%ROOT%\requirements.txt

rem ffmpeg and sox variables
set SOX_DIR=%DEPS_DIR%\sox
set FFMPEG_DIR=%DEPS_DIR%\ffmpeg

set PATH=%PATH%;%CONDA_SCRIPT_DIR%;%SOX_DIR%;%FFMPEG_DIR%

if "%1" == "install" (
    echo Installing Ultimate RVC

    if exist "%DEPS_DIR%" (
        echo Removing existing dependencies folder...
        rmdir /s /q "%DEPS_DIR%"
    )

    echo Downloading %ZIP_FILE%...
    curl -s -LJO %URL_MAIN% -o %ZIP_FILE%

    if exist "%DEPS_ZIP%" (
        echo %ZIP_FILE% downloaded successfully.
    ) else (
        echo Download failed, please check your internet connection
        exit /b 1
    )

    echo Extracting %ZIP_FILE%...

    tar -xf "%DEPS_ZIP%"

    if exist "%DEPS_DIR%" (
        del %ZIP_FILE%
        echo %ZIP_FILE% extracted successfully.
    ) else (
        echo Failed to extract %ZIP_FILE%. Please download the file and extract it manually:
        echo %URL_MAIN%
        exit /b 1
    )

    if not exist "%CONDA_SCRIPT_DIR%" (
        echo Installing Miniconda to %CONDA_ROOT%...
        start /wait "" "%CONDA_INSTALLER%" /InstallationType=JustMe /RegisterPython=0 /S /D=%CONDA_ROOT%
        echo Miniconda installed successfully.
    )
    if exist "%CONDA_INSTALLER%" (
        del "%CONDA_INSTALLER%"
    )
    echo Initializing conda environment...
    call conda create --prefix "%VENV_ROOT%" --no-shortcuts -y -k -q python=3.12
    echo Conda environment initialized successfully.

    echo Installing Python packages..
    call activate.bat "%VENV_ROOT%"
    pip cache purge
    python -m pip install --upgrade pip setuptools
    pip install -r "%REQUIREMENTS_FILE%"
    echo Python packages installed successfully.

    echo Installing base models...
    python ./src/init.py
    call conda deactivate

    echo Ultimate RVC has been installed successfully!
    exit /b 0
)

if "%1" == "update" (
    echo Updating Ultimate RVC
    
    if not exist "%CONDA_ROOT%" (
        echo Miniconda not found. Please run './urvc.bat install' first.
        exit /b 1
    )
    echo Updating repository...
    git pull

    echo Updating Python packages...
    call activate.bat
    call conda remove --prefix "%VENV_ROOT%" --all -y -q
    call conda create --prefix "%VENV_ROOT%" --no-shortcuts -y -k -q python=3.12
    call conda activate "%VENV_ROOT%"
    pip cache purge
    python -m pip install --upgrade pip setuptools
    pip install -r "%REQUIREMENTS_FILE%"
    call conda deactivate

    echo Ultimate RVC has been updated successfully!
    exit /b 0
)

if "%1" == "run" (
    echo Starting Ultimate RVC
    if not exist "%DEPS_DIR%" (
        echo Please run './urvc.bat install' first to set up dependencies.
        exit /b 1
    )
    call :shift_args %*
    call "%PYTHON_PATH%" ./src/app.py !args!
    exit /b 0
)


if "%1" == "dev" (
    echo Starting Ultimate RVC in development mode
    if not exist "%DEPS_DIR%" (
        echo Please run './urvc.bat install' first to set up dependencies.
        exit /b 1
    )
    call "%GRADIO_PATH%" ./src/app.py --demo-name app
    exit /b 0
)

if "%1" == "cli" (
    echo Starting Ultimate RVC in CLI mode
    if not exist "%DEPS_DIR%" (
        echo Please run './urvc.bat install' first to set up dependencies.
        exit /b 1
    )
    call :shift_args %*
    call "%PYTHON_PATH%" ./src/cli.py !args!
    exit /b 0
)

if "%1" == "docs" (
    echo Starting Ultimate RVC in CLI mode
    cd src
    call "%PYTHON_PATH%" -m typer %2 utils docs --output %3
    exit /b 0
) else (
    echo.
    echo Usage: ^.^/urvc.bat [OPTIONS] COMMAND [ARGS]...
    echo.
    echo Commands:
    echo   install    Install dependencies and set up environment.
    echo   update     Update application and dependencies.
    echo   run        Start Ultimate RVC in run mode.
    echo               Options:
    echo                 --help      Show help message and exit.
    echo               [more information available, use --help to see all]
    echo   dev        Start Ultimate RVC in development mode.
    echo   cli        Start Ultimate RVC in CLI mode.
    echo               Options:
    echo                 --help      Show help message and exit.
    echo               [more information available, use --help to see all]
    echo   docs       Generate documentation using Typer.
    echo               Args:
    echo                 module     The module to generate documentation for.
    echo                 output     The output directory
    exit /b 1
)

:shift_args
set args=
:loop
shift
if "%~1"=="" goto :eof
set args=%args% %1
goto :loop