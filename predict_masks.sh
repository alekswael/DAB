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

# DaAnonymization
$PYTHON src/predict/DaAnonymization_predict.py \
--data_path "./data/DAB_annotated_dataset.json" \
--save_path "./output/predictions/DaAnonymization_predictions.json"

# DaAnonymization with fine-grained predictions
$PYTHON src/predict/DaAnonymization_predict.py \
--data_path "./data/DAB_annotated_dataset.json" \
--save_path "./output/predictions/DaAnonymization_FG_predictions.json" \
--fine_grained

# Gemma
$PYTHON src/predict/gemma_predict.py \
--data_path "./data/DAB_annotated_dataset.json" \
--save_path "./output/predictions/Gemma_predictions.json" 

# Deactivate the virtual environment
deactivate