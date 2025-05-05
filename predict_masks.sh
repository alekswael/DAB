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

# Predict masks with models
echo "Predicting masks..."

$PYTHON src/predict/DaAnonymization_predict.py \
--data_path "./data/annotations_15_04_2025.json" \
--save_path "./output/predictions/"

$PYTHON src/predict/DaAnonymization_predict.py \
--data_path "./data/annotations_15_04_2025.json" \
--save_path "./output/predictions/" \
--fine_grained

$PYTHON src/predict/gemma_predict.py \
--data_path "./data/annotations_15_04_2025.json" \
--save_path "./output/predictions/" \
--cloud

# Deactivate the virtual environment
deactivate