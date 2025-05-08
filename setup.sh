#!/bin/bash

REQUIRED_VERSION="3.12"

# Try to find a python3.12 executable
if command -v python3.12 &> /dev/null; then
    PYTHON_BIN=$(command -v python3.12)
else
    # Fallback to python3
    PYTHON_BIN=$(command -v python3)
    if [ -z "$PYTHON_BIN" ]; then
        echo "Python 3 is not installed."
        exit 1
    fi

    # Check version
    VERSION=$($PYTHON_BIN -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    if [ "$VERSION" != "$REQUIRED_VERSION" ]; then
        echo "Error: Python $REQUIRED_VERSION is required, but found $VERSION."
        exit 1
    fi
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

# spacy-experimental 6.4.0 must be built from git
git clone https://github.com/explosion/spacy-experimental.git external/spacy-experimental

# Install dependencies from requirements.txt
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    echo "Dependencies installed."
else
    echo "WARNING: No requirements.txt file found."
fi

# Force transformers==4.50.0
pip install transformers==4.50.0

# Download spacy model
$PYTHON -m spacy download da_core_news_trf

# Deactivate the virtual environment
deactivate