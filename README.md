# Danish Anonymization Benchmark (DAB)

##  Description

*Version:* 1.0

The _Danish Anonymization Benchmark_ (DAB) is a GDPR-oriented, open-source benchmark for evaluating automated anonymization of Danish text data. The current version (1.0) consists of 54 manually annotated (anonymized) Danish documents and pipelines for benchmarking anonymization models and expanding the dataset by adding and annotating new data. 

**The project features:**
- Generate masking predictions with instruction-tuned 🤗 HuggingFace models
- Benchmark anonymization models to obtain evaluation metrics
- DAB annotation guidelines
- Annotate new data in Label Studio with a setup guide and config
- Support for multiple annotators
- Bootstrap annotations with a pre-annotation framework

## 🔧 Setup

**System requirements:**

- This project is developed on Python 3.12.3 and **only supports Python 3.12**.
- Bash-compatible shell

**Guide for setup:**

1. Clone the repository:
    ```bash
    git clone https://github.com/alekswael/DAB
    cd DAB
    ```

2. Run `setup.sh`:
    ```bash
    bash setup.sh
    ```
    This script checks for the Python version, creates a virtual environment at `venv/`, installs dependencies from `requirements.txt` and downloads a SpaCy model.
    

3. Activate the virtual environment:

    ```bash
    # For MacOS and Linux
    source venv/bin/activate

    # For Windows (cmd.exe)
    venv\Scripts\activate.bat
    ```

**NOTE:** To use this project, you must be authenticated with the Hugging Face Hub. Please ensure you have a Hugging Face account and an access token. You can log in by running `huggingface-cli login` and following the instructions (see the [documention](https://huggingface.co/docs/huggingface_hub/en/guides/cli) for help).

## 👩‍💻 Usage

### 📊 Benchmark an anonymization model

1. 📈 **Model prediction**

    Generate masking predictions with an anonymization model. Currently, the project contains code for generating predictions with three model configurations:

    1. [*DaAnonymization*](https://github.com/martincjespersen/DaAnonymization) with _DaCy large_ (simple, adapted version for this project)
    2. [*DaAnonymization*](https://github.com/martincjespersen/DaAnonymization) with _DaCy large fine-grained_ (simple, adapted version for this project)
    3. [_google/gemma-3-12b-it_](https://huggingface.co/google/gemma-3-12b-it), implemented through 🤗 Hugging Face (locally) or Google's API (cloud based)

    To generate predictions for these models, you can run the `predict_masks.sh` script:

    ```bash
    bash predict_masks.sh
    ```

    You can add the `--cloud` flag when running `gemma_predict.py` in `predict_masks.sh` if you want to run Gemma through Google's API - make sure to set the `GOOGLE_API_KEY` environment variable in a `.env` file.

    **Instruction-tuned models from 🤗 Hugging Face**

    There is also support for generating masks by prompting an instruction-tuned model hosted on 🤗 Hugging Face. To do this, you can run the `hf_pipeline_predict.py` script and specify the `--model_name` flag:

    ```bash
    python src/predict/hf_pipeline_predict.py \
      --data_path "./data/DAB_annotated_dataset.json" \
      --save_path "./output/predictions/gemma_3_1b_it_predictions.json" \
      --model_name "google/gemma-3-1b-it"
    ```

    You can view/change the instruction prompt in the `hf_pipeline_instruction_prompt.txt` file (this prompt is also used when prompting *google/gemma-3-12b-it*).

    🕵️ **Other anonymization models**
    
    If you want to generate predictions with a different model, make sure to save the output with the correct formatting. See the model prediction JSON reference in the [formatting reference](annotation/JSON_format_reference.md) for more information.

2. 📋️ **Model evaluation**

    Evaluate an anonymization model on a series of metrics. If you want to benchmark the provided models, you can run the `benchmark_models.sh` script:

    ```bash
    bash benchmark_models.sh
    ```

    To benchmark a single model, make sure the predictions are available in `output/predictions/`. Specify the arguments and run the `benchmark_model.py` script:

    ```bash
    python src/benchmark/benchmark_model.py \
    --gold_standard_file "./data/DAB_annotated_dataset.json" \
    --model_predictions_file "./output/predictions/mymodel_predictions.json" \
    --benchmark_output_file "./output/benchmarks/mymodel_benchmark_result.txt" \
    --bert_weighting
    ```

### 🌐 Expand the dataset and annotate new documents

1. 📄 **Add new documents**

    Create a new subfolder in `data/raw/` and documents for annotation. Each subfolder should be named according to the dataset source, e.g.

    ```
    data/
    ├── raw/
    │   ├── private_docs/
    │   │   ├── document1.txt
    │   │   ├── document2.pdf
    │   │   └── ...
    │   ├── legal_cases/
    │   │   ├── case1.txt
    │   │   ├── case2.pdf
    │   │   └── ...
    │   ├── news_articles/
    │   │   ├── article1.txt
    │   │   ├── article2.pdf
    │   │   └── ...
    ```

    **NOTE:** Currently supports `.txt` and `.pdf` (native and non-native) files.

2. 📁 **Compile dataset**

    Run the `compile_dataset.py` script:

    ```bash
    python src/data_processing/compile_dataset.py \
      --data_dir "./data/raw/" \
      --save_path "./data/dataset.json"
    ```

    Compile the raw documents from the subfolders in `data/raw/` and format into a single basic Label Studio JSON.

3. 🤖 **Pre-annotate dataset**

    Pre-annotate the dataset to bootstrap the annotation process. The pre-annotations consist of fine-grained named entities generated with _DaCy_ fine-grained medium (Enevoldsen et al., 2021; Enevoldsen et al., 2024) and a series of RegExes. To pre-annotate the dataset, run the `pre_annotate.py` script:

    ```bash
    python src/data_processing/pre_annotate.py \
      --data_path "./data/dataset.json" \
      --save_path "./data/dataset_pre_annotated.json"
    ```

4. ✍️ **Annotate the documents in Label Studio**

    Annotate your own data in Label Studio. Read and follow the [DAB Annotation Guidelines](annotation/DAB_Annotation_Guidelines.pdf) and the [Label Studio setup guide](annotation/label_studio_setup_guide.md) in the `annotation/` folder.

5. 🛠️ **Post-process annotated dataset**

    After saving your annotated JSON file, post-process it to make it compatible with the prediction/evaluation pipeline. Run the `add_entity_ids.py` script:

    ```bash
    python src/data_processing/add_entity_ids.py \
        --data_path "./data/dataset_annotated.json"
    ```

    If you want to print the masked text from your annotations, you can run `check_annotated_offsets.py`:

    ```bash
    python src/data_processing/check_annotated_offsets.py \
        --data_path "./data/dataset_annotated.json"
    ```

## Known bugs/errors

- `spacy-experimental 0.6.4` must be installed from source, see the `setup.sh` file
- Running with Python 3.13 will raise an error when installing `spacy-experimental 0.6.4`
- Gemma 3 models require atleast `transformers==4.50.0`, forcing this raises dependency issues with `spacy-transformers` but has no impact on performance


## Future implementations

- Increase no. of documents & annotators
- Convert project to package

## Acknowledgements

The annotation guidelines and evaluation methodology are adapted from the *Text Anonymization Benchmark* by Pilán et al. (2022) ([Github](https://github.com/NorskRegnesentral/text-anonymization-benchmark) | [Paper](https://arxiv.org/abs/2202.00443)).

## References

Enevoldsen, K., Hansen, L., & Nielbo, K. L. (2021). DaCy: A unified framework for danish NLP. Ceur Workshop Proceedings, 2989, 206-216.

Enevoldsen, K., Jessen, E. T., & Baglini, R. (2024). DANSK: Domain Generalization of Danish Named Entity Recognition. Northern European Journal of Language Technology, 10(1), Article 1. https://doi.org/10.3384/nejlt.2000-1533.2024.5249

Pilán, I., Lison, P., Øvrelid, L., Papadopoulou, A., Sánchez, D., & Batet, M. (2022). The Text Anonymization Benchmark (TAB): A Dedicated Corpus and Evaluation Framework for Text Anonymization. Computational Linguistics, 48(4), 1053–1101. https://doi.org/10.1162/coli_a_00458

---
