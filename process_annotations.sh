#!/bin/bash

# Preprocess the annotations after exporting from the Label Studio UI.
# Print the text and masked annotations.

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
$PYTHON src/data_processing/add_entity_ids.py \
--data_path ./data/annotations_15_04_2025.json

$PYTHON src/data_processing/check_annotated_offsets.py \
--data_path ./data/annotations_15_04_2025.json

# Deactivate the virtual environment
deactivate