#!/bin/bash

# Check for python version
if command -v python3 &>/dev/null; then
    PYTHON=python3
else
    PYTHON=python
fi

# Set the virtual environment directory name
VENV_DIR="venv"

# Create the virtual environment if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    $PYTHON -m venv "$VENV_DIR"
    echo "Virtual environment created in $VENV_DIR."
else
    echo "Virtual environment already exists."
fi

# Activate the virtual environment
source "$VENV_DIR/bin/activate"

# Install dependencies from requirements.txt
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    echo "Dependencies installed."
else
    echo "No requirements.txt file found."
fi

# Download spacy model
$PYTHON -m spacy download da_core_news_trf

# Deactivate the virtual environment
deactivate
echo "Virtual environment deactivated."
