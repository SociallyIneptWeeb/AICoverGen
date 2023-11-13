#!/bin/bash

# Set the name of the virtual environment directory
venv_dir="venv"

# Check if the venv directory exists
if [ -d "$venv_dir" ]; then
  # If it exists, do nothing silently
  :
else
  # If it doesn't exist, create the virtual environment
  python -m venv "$venv_dir"
  echo "Virtual environment created in '$venv_dir'."
  
  # Source the virtual environment
  source "$venv_dir/bin/activate"

  # Install requierd packages for amd based systems
  pip install -r requirements_amd.txt

fi
