# Danish Anonymization Benchmark (DAB)

*Version:* 1.0

The _Danish Anonymization Benchmark_ (DAB) is a GDPR-oriented, open-source benchmark for evaluating automated anonymization of Danish text data. The current version (1.0) consists of 54 manually annotated (anonymized) Danish documents and pipelines for benchmarking anonymization models and expanding the dataset by adding and annotating new data. 

**The project features:**
- Support for multiple annotators
- Pre-annotation framework
- Model prediction framework for instruction-tuned 🤗 HuggingFace models
- Benchmark anonymization models and obtain evaluation metrics
- Annotation guidelines
- Setup guide and .xml config for annotation in Label Studio

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

## 👩‍💻 Usage

### 📊 Benchmark an anonymization model

- **Model prediction**: Generate masking predictions with an anonymization model. Currently, the project contains code for generating predictions with three model configurations:

  1. [*DaAnonymization*](https://github.com/martincjespersen/DaAnonymization) with _DaCy large_ (simple, adapted version for this project)
  2. [*DaAnonymization*](https://github.com/martincjespersen/DaAnonymization) with _DaCy large fine-grained_ (simple, adapted version for this project)
  3. [_google/gemma-3-12b-it_](https://huggingface.co/google/gemma-3-12b-it), implemented through HuggingFace

  To generate predictions for these models, you can run the `predict_masks.sh` script:
  
  ```bash
  bash predict_masks.sh
  ```

  There is also support for generating masks by prompting an instruction-tuned model hosted on 🤗 HuggingFace. To do this, you can run the `hf_pipeline_predict.py` script and specify the model_name:

  ```bash
  python3 src/predict/hf_pipeline_predict.py \
    --data_path "./data/annotations_15_04_2025.json" \
    --save_path "./output/predictions/" \
    --model_name "google/gemma-3-4b-it"
  ```
  You can view/change the instruction prompt in the `hf_pipeline_predict.py` script.

  ***NOTE:*** *If you want to generate predictions with a different model, make sure to save the output with the correct formatting. See the model prediction JSON reference in the [formatting reference](annotation/JSON_format_reference.md) for more information.*

- **Model evaluation**: Evaluate an anonymization model on a series of metrics. If you want to benchmark the provided models, you can run the `benchmark_models.sh` script:

  ```bash
  bash benchmark_models.sh
  ```

  To benchmark a single model, specify the arguments and run the `benchmark_model.py` script:

  ```bash
  python3 src/benchmark/benchmark_model.py \
    --gold_standard_file "./data/annotations_15_04_2025.json" \
    --masked_output_dir "./output/predictions/" \
    --benchmark_output_dir "./output/benchmarks/" \
    --model "DaAnonymization" \
    --bert_weighting
  ```

### 🌐 Expand the dataset and annotate new documents

1. 📄 **Add new documents**: Create a new folder in `data/` and add subfolders with documents for annotation (currently supports `.txt` and `.pdf` files). Subfolders should be named according to the source (e.g. `wiki_bio_dk/` as seen in `data/raw/`).

    ```
    data/
    ├── my_data/
    │   ├── data_source_1/
    │   │   ├── document1.txt
    │   │   ├── document2.pdf
    │   │   └── ...
    │   ├── data_source_2/
    │   │   ├── case1.txt
    │   │   ├── case2.pdf
    │   │   └── ...
    │   ├── data_source_3/
    │   │   ├── article1.txt
    │   │   ├── article2.pdf
    │   │   └── ...
    ```

2. 📁 **Compile dataset**: Compile the raw documents from the subfolders in `data/raw/` and format into a single basic Label Studio JSON. To achieve this, run the `prepare_dataset.sh` script:

    ```bash
    python3 src/data_processing/compile_dataset.py \
      --data_dir "./data/raw/" \
      --save_path "./data/DAB_dataset_pre_annotated.json"
    ```

3. 🤖 **Pre-annotate dataset**: Pre-annotate the dataset to bootstrap the annotation process. The pre-annotations consist of fine-grained named entities generated with _DaCy_ fine-grained medium (Enevoldsen et al., 2021; Enevoldsen et al., 2024) and a series of RegExes. To pre-annotate the dataset, run the `pre_annotate.py` script:

    ```bash
    python3 src/data_processing/pre_annotate.py \
      --data_dir "./data/raw/" \
      --save_path "./data/DAB_dataset_pre_annotated.json"
    ```

4. ✍️ **Annotate the documents in Label Studio**: Annotate your own data in Label Studio. Read and follow the DAB Annotation Guidelines and the Label Studio setup guide in the `annotation/` folder.

## Annotation Guidelines

Finalize the annotation guidelines and ensure consistency in the NER label framework. Refer to the `annotation/JSON_format_reference.py` file for details on the JSON structure used for annotations and predictions.

## Future implementations

## Acknowledgements

The annotation guidelines and evaluation methodology are adapted from the [*Text Anonymization Benchmark*](https://github.com/NorskRegnesentral/text-anonymization-benchmark) by Pilán et al. (2022) (Github | Paper).

## References

Enevoldsen, K., Hansen, L., & Nielbo, K. L. (2021). DaCy: A unified framework for danish NLP. Ceur Workshop Proceedings, 2989, 206-216.

Enevoldsen, K., Jessen, E. T., & Baglini, R. (2024). DANSK: Domain Generalization of Danish Named Entity Recognition. Northern European Journal of Language Technology, 10(1), Article 1. https://doi.org/10.3384/nejlt.2000-1533.2024.5249

Pilán, I., Lison, P., Øvrelid, L., Papadopoulou, A., Sánchez, D., & Batet, M. (2022). The Text Anonymization Benchmark (TAB): A Dedicated Corpus and Evaluation Framework for Text Anonymization. Computational Linguistics, 48(4), 1053–1101. https://doi.org/10.1162/coli_a_00458

---
