#!/bin/bash

# Run all models
echo "Benchmarking models..."

# DaAnonymization
python src/benchmark/benchmark_model.py \
    --gold_standard_file "./data/DAB_annotated_dataset.json" \
    --model_predictions_file "./output/predictions/DaAnonymization_predictions.json" \
    --benchmark_output_file "./output/benchmarks/DaAnonymization_benchmark_result.txt" \
    --bert_weighting

# DaAnonymization with fine-grained predictions
python src/benchmark/benchmark_model.py \
    --gold_standard_file "./data/DAB_annotated_dataset.json" \
    --model_predictions_file "./output/predictions/DaAnonymization_FG_predictions.json" \
    --benchmark_output_file "./output/benchmarks/DaAnonymization_FG_benchmark_result.txt" \
    --bert_weighting

# Gemma
python src/benchmark/benchmark_model.py \
    --gold_standard_file "./data/DAB_annotated_dataset.json" \
    --model_predictions_file "./output/predictions/Gemma_predictions.json" \
    --benchmark_output_file "./output/benchmarks/Gemma_benchmark_result.txt" \
    --bert_weighting