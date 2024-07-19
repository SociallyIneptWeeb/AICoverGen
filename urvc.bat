@echo off
setlocal
title Ultimate RVC launcher

set "ROOT=%cd%"
set "URL_MAIN=https://huggingface.co/JackismyShephard/ultimate-rvc/resolve/main"

set "DEPENDENCIES_DIR=%ROOT%\dependencies"
set "VIRTUAL_ENV_DIR=%DEPENDENCIES_DIR%\.venv"
set "CONDA_ROOT=%UserProfile%\Miniconda3"
set "CONDA_EXE_DIR=%CONDA_ROOT%\Scripts"

set "SOX_DIR=%DEPENDENCIES_DIR%\sox"
set "FFMPEG_DIR=%DEPENDENCIES_DIR%\ffmpeg"
set PATH=%PATH%;%SOX_DIR%;%FFMPEG_DIR%;%CONDA_EXE_DIR%


if "%1" == "" (
    echo Usage ^.^/urvc.bat ^[install^|run^|update^|dev^]
    exit /b 1
)

if "%1" == "install" (
    echo Installing Ultimate RVC

    echo.
    
    if exist %DEPENDENCIES_DIR% (
        echo Removing existing dependencies folder...
        rmdir /s /q %DEPENDENCIES_DIR%
    )

    echo Downloading dependencies.zip file...
    curl -s -LJO %URL_MAIN%/dependencies.zip -o dependencies.zip

    if not exist "%ROOT%\dependencies.zip" (
        echo Download failed, trying with the powershell method
        powershell -Command "& {Invoke-WebRequest -Uri %URL_MAIN%/dependencies.zip -OutFile 'dependencies.zip'}"
    )

    echo Extracting dependencies folder...
    powershell -command "& { Add-Type -AssemblyName System.IO.Compression.FileSystem ; [System.IO.Compression.ZipFile]::ExtractToDirectory('%ROOT%\dependencies.zip', '%ROOT%') }"

    if not exist %DEPENDENCIES_DIR% (
        echo Extracting failed trying with the tar method...
        tar -xf %ROOT%\dependencies.zip
    )

    if exist %DEPENDENCIES_DIR% (
        del dependencies.zip
        echo Dependencies folder extracted successfully.
    ) else (
        echo Failed to extract dependencies folder. Please download the file and extract it manually.
        echo "%URL_MAIN%/dependencies.zip"
        exit /b 1
    )
    cd %DEPENDENCIES_DIR%
    if not exist "%CONDA_EXE_DIR%" (
        echo Installing Miniconda to %CONDA_ROOT%...
        start /wait "" miniconda3_11.exe /InstallationType=JustMe /RegisterPython=0 /S /D=%CONDA_ROOT%
    )
    if exist miniconda3_11.exe (
        del miniconda3_11.exe
    )
    cd %ROOT%

    call conda create --no-shortcuts -y -k --prefix %VIRTUAL_ENV_DIR% python=3.11
    call activate.bat %VIRTUAL_ENV_DIR%
    echo Installing Python packages..
    REM installing vs2015_runtime is necessary due to conda using an old vc runtime 
    REM that is not compatible with new visual studio compiler
    call conda install -y -c conda-forge faiss-cpu vs2015_runtime
    pip cache purge
    pip install --upgrade setuptools
    pip install -r "%ROOT%\requirements.txt"

    echo.
    echo Installing base models...
    python ./src/init.py
    
    echo.
    echo Ultimate RVC has been installed successfully!
    call conda deactivate

    exit /b 0
)

if "%1" == "run" (
    echo Starting Ultimate RVC
    if not exist %DEPENDENCIES_DIR% (
        echo Please run './urvc.bat install' first to set up dependencies.
        exit /b 1
    )
    call "%VIRTUAL_ENV_DIR%\python.exe" ./src/app.py
    exit /b 0
)

if "%1" == "update" (
    echo Updating Ultimate RVC
    if not exist %CONDA_ROOT% (
        echo Miniconda not found. Please run './urvc.bat install' first.
        exit /b 1
    )
    git pull
    call activate.bat
    call conda remove --prefix %VIRTUAL_ENV_DIR% --all --yes
    call conda create --no-shortcuts -y -k --prefix %VIRTUAL_ENV_DIR% python=3.11
    call conda activate %VIRTUAL_ENV_DIR%
    call conda install -y -c conda-forge vs2015_runtime faiss-cpu
    pip cache purge
    pip install --upgrade setuptools
    pip install -r "%ROOT%\requirements.txt"
    call conda deactivate

    echo.
    echo Ultimate RVC has been updated successfully!
    exit /b 0
)

if "%1" == "dev" (
    echo Starting Ultimate RVC in development mode
    if not exist %DEPENDENCIES_DIR% (
        echo Please run './urvc.bat install' first to set up dependencies.
        exit /b 1
    )

    call "%VIRTUAL_ENV_DIR%\Scripts\gradio.exe" ./src/app.py --demo-name app
    exit /b 0
)

echo echo Usage ^.^/urvc.bat ^[install^|run^|update^|dev^]
exit /b 1

