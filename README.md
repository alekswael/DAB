# Danish Anonymization Benchmark (DAB)

**Version:** 1.0

The _Danish Anonymization Benchmark (DAB)_ is a GDPR-oriented, open-source project for evaluating automated anonymization of Danish text data.

The current version (1.0) consists of 54 manually annotated (anonymized) documents and pipelines for annotation and benchmarking anonymization models.

## Table of Contents
- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Setup](#setup)
- [Usage](#usage)
  - [Dataset Preparation](#dataset-preparation)
  - [Pre-Annotation](#pre-annotation)
  - [Prediction](#prediction)
  - [Benchmarking](#benchmarking)
- [Annotation Guidelines](#annotation-guidelines)

## Overview

The DAB project provides a complete pipeline for data handling, pre-annotation, annotation, prediction and evaluation of anonymization models.

It current version (1.0) supports:

###  Annotation pipeline ✍️

**Data handling**

- Compile documents from the subfolders in `data/raw/` and format into the basic Label Studio JSON format (currently supports .txt and .pdf files), ready for annotation.

**Pre-annotate dataset**

- Pre-annotate the dataset to bootstrap the annotation process. The pre-annotations are generated with a _DaCy_ model (Enevoldsen et al., 2021) and a series of RegExes.

**Annotation framework**

- Annotate your own data in Label Studio. Follow the DAB Annotation Guidelines and the Label Studio setup guide in the `annotation/` folder.

### Benchmark pipeline 📊

**Model prediction**

- Generate masking predictions with an anonymization model. 

**Model evaluation**
- Evaluating anonymization models using precision, recall, and other metrics.

## Repository Structure

```
DAB/
├── annotation/                # Reference for JSON formats used in Label Studio
├── data/                      # Directory for raw, pre-annotated, and processed datasets
├── output/                    # Directory for predictions and benchmark results
├── src/                       # Source code for various components
│   ├── benchmark/             # Scripts for benchmarking anonymization models
│   ├── data_processing/       # Scripts for dataset preparation and pre-annotation
│   ├── predict/               # Scripts for generating predictions
├── prepare_dataset.sh         # Bash script for preparing datasets
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

## 🔧 Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/alekswael/DAB
   cd DAB
   ```

2. Run `setup.sh`:
   ```bash
   bash setup.sh
   ```

## Usage

### Dataset Preparation

Run the `prepare_dataset.sh` script to compile raw `.txt` and `.pdf` files into a JSON dataset and generate pre-annotations:
```bash
bash prepare_dataset.sh
```

### Pre-Annotation

Generate pre-annotations for a dataset using the `pre_annotate.py` script:
```bash
python src/data_processing/pre_annotate.py \
  --data_path ./data/DAB_dataset.json \
  --save_path ./data/DAB_dataset_pre_annotated.json \
  --model dacy
```

### Prediction

Generate anonymization predictions using the `DaAnonymization_predict.py` script:
```bash
python src/predict/DaAnonymization_predict.py \
  --data_path ./data/annotations_15_04_2025.json \
  --save_path ./output/predictions/ \
  --fine_grained
```

### Benchmarking

Evaluate the anonymization model using the `benchmark_model.py` script:
```bash
python src/benchmark/benchmark_model.py \
  --gold_standard_file ./data/annotations_15_04_2025.json \
  --masked_output_dir ./output/predictions/ \
  --benchmark_output_dir ./output/benchmarks/ \
  --model DaAnonymization
```

## Annotation Guidelines

Finalize the annotation guidelines and ensure consistency in the NER label framework. Refer to the `annotation/JSON_format_reference.py` file for details on the JSON structure used for annotations and predictions.

## Future implementations

## References

Enevoldsen, K., Hansen, L., & Nielbo, K. L. (2021). DaCy: A unified framework for danish NLP. Ceur Workshop Proceedings, 2989, 206-216.

---
