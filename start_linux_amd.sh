#!/bin/bash

# Set the name of the virtual environment directory
venv_dir="venv"

# Function to find the latest Python version below 3.11
find_latest_python_version() {
  for version in /usr/bin/python3.*; do
    if [[ $("$version" -c "import sys; print(sys.version_info[:2] < (3, 11))") == "True" ]]; then
      echo "$version"
      return
    fi
  done
}

# Check if the venv directory exists
if [ -d "$venv_dir" ]; then
  # Source the virtual environment if it exists
  source "$venv_dir/bin/activate"
else
  # Find the latest Python version below 3.11
  python_version=$(find_latest_python_version)

  if [ -z "$python_version" ]; then
    echo "No suitable Python 3.x version (below 3.11) found in /usr/bin." 
    echo "Please install a compatible Python version and try again."
    exit 1
  fi

  # If it doesn't exist, create the virtual environment
  $python_version -m venv "$venv_dir"
  echo "Virtual environment created in '$venv_dir' using Python $($python_version --version)."
  
  # Source the virtual environment
  source "$venv_dir/bin/activate"

  # Install requierd packages for amd based systems
  pip install -r requirements_amd.txt

fi

# Load environment variables from rocm_vars.txt
if [ -f "rocm_vars.txt" ]; then
  source "rocm_vars.txt"
  echo "ROCM environment variables loaded from rocm_vars.txt."
fi

# Check if .models_present is not set to 1
if [ ! -f .models_present ] || [ "$(cat .models_present)" != "1" ]; then
    echo "Downloading Models..."

    # Run the download_models.py script
    python src/download_models.py

    # Check if the download was successful (you may need to modify this condition based on the behavior of download_models.py)
    if [ $? -eq 0 ]; then
        # Set .models_present to 1 to indicate that the models are present
        echo "1" > .models_present
    else
        # Handle the case where the download failed
        echo "Model download failed."
    fi
fi
