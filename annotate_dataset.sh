#!/bin/bash

# Check for python version
if command -v python3 &>/dev/null; then
    PYTHON=python3
else
    PYTHON=python
fi

# Set the virtual environment directory name
VENV_DIR="venv"

# Activate the virtual environment
source "$VENV_DIR/bin/activate"

# Start Label Studio in the background on the free port
echo "Starting Label Studio..."
label-studio start

# deactivate venv
deactivate