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

$PYTHON src/benchmark/benchmark_model.py \
--gold_standard_file "./data/annotations_15_04_2025.json" \
--masked_output_file "./output/predictions/" \
--benchmark_output_dir "./output/benchmarks/" \
--model "DaAnonymization" \
--bert_weighting

$PYTHON src/benchmark/benchmark_model.py \
--gold_standard_file "./data/annotations_15_04_2025.json" \
--masked_output_file "./output/predictions/" \
--benchmark_output_dir "./output/benchmarks/" \
--model "DaAnonymization_FG" \
--bert_weighting

$PYTHON src/benchmark/benchmark_model.py \
--gold_standard_file "./data/annotations_15_04_2025.json" \
--masked_output_file "./output/predictions/" \
--benchmark_output_dir "./output/benchmarks/" \
--model "Gemma" \
--bert_weighting

# Deactivate the virtual environment
deactivate