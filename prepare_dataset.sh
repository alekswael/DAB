#!/bin/bash

# Compile raw .txt and .pdf files into the Basic Label Studio JSON format.


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

# Run scripts
$PYTHON src/data_processing/compile_dataset.py \
--data_dir ./data/raw/ \
--save_path ./data/DAB_dataset_pre_annotated.json

echo "Generating pre-annotations..."
$PYTHON src/data_processing/pre_annotate.py \
--data_path ./data/DAB_dataset.json \
--save_path ./data/DAB_dataset_pre_annotated.json \
--model dacy

# Deactivate the virtual environment
deactivate