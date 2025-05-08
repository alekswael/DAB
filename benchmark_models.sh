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

# Run all models
echo "Benchmarking models..."

# DaAnonymization
$PYTHON src/benchmark/benchmark_model.py \
--gold_standard_file "./data/annotations_15_04_2025.json" \
--model_predictions_file "./output/predictions/DaAnonymization_predictions.json" \
--benchmark_output_file "./output/benchmarks/DaAnonymization_benchmark_result.txt" \
--bert_weighting

# DaAnonymization with fine-grained predictions
$PYTHON src/benchmark/benchmark_model.py \
--gold_standard_file "./data/annotations_15_04_2025.json" \
--model_predictions_file "./output/predictions/DaAnonymization_FG_predictions.json" \
--benchmark_output_file "./output/benchmarks/DaAnonymization_FG_benchmark_result.txt" \
--bert_weighting

# Gemma
$PYTHON src/benchmark/benchmark_model.py \
--gold_standard_file "./data/annotations_15_04_2025.json" \
--model_predictions_file "./output/predictions/Gemma_predictions.json" \
--benchmark_output_file "./output/benchmarks/Gemma_benchmark_result.txt" \
--bert_weighting

# Deactivate the virtual environment
deactivate