#!/bin/bash

# Predict masks with models
echo "Predicting masks..."

# DaAnonymization
python src/predict/DaAnonymization_predict.py \
    --data_path "./data/DAB_annotated_dataset.json" \
    --save_path "./output/predictions/DaAnonymization_predictions.json"

# DaAnonymization with fine-grained predictions
python src/predict/DaAnonymization_predict.py \
    --data_path "./data/DAB_annotated_dataset.json" \
    --save_path "./output/predictions/DaAnonymization_FG_predictions.json" \
    --fine_grained

# Gemma
python src/predict/gemma_predict.py \
    --data_path "./data/DAB_annotated_dataset.json" \
    --save_path "./output/predictions/Gemma_predictions.json" 