# Import
import argparse
import re
import json
from transformers import AutoTokenizer
import dacy

# Constants
MAX_LENGTH = 512  # Max length of chunk size, 512 is the default for BERT-based models
TOKENIZER = "vesteinn/DanskBERT"  # DanskBERT tokenizer
MODEL = "da_dacy_medium_ner_fine_grained-0.1.0"  # DaCy model for fine-grained NER


def parse_arguments():
    """
    Parse command-line arguments for the script.

    Returns:
        argparse.Namespace: Parsed arguments including data_path, save_path, and verbose flag.
    """
    parser = argparse.ArgumentParser(
        description="This script is used for generating the pre-annotations for the annotation pipeline with DaCy + ReGex."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        help="The path to the dataset in Label Studio JSON format.",
        default="./data/dataset.json",
        required=False,
    )
    parser.add_argument(
        "--save_path",
        type=str,
        help="The path for saving the pre-annotated JSON dataset.",
        default="./data/dataset_pre_annotated.json",
        required=False,
    )
    parser.add_argument("--verbose", action="store_true", default=True)

    return parser.parse_args()


def load_data(data_path):
    """
    Load data from a JSON file.

    Args:
        data_path (str): Path to the JSON file containing the dataset.

    Returns:
        list: List of data entries loaded from the JSON file.
    """
    with open(data_path, "r", encoding="utf-8") as doc:
        data_list = json.load(doc)

    return data_list


def initiate_ner_pipeline(verbose):
    """
    Initialize the NER pipeline by loading the tokenizer.

    Args:
        verbose (bool): Flag to enable verbose logging.

    Returns:
        transformers.PreTrainedTokenizer: Initialized tokenizer.
    """
    print(f"[INFO]: Initiating DaCy model...")

    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER,
        clean_up_tokenization_spaces=True,
        is_split_into_words=False,
        truncation=False,
    )

    if verbose:
        print(f"Tokenizer init parameters: {tokenizer.init_kwargs}")
        print(f"Tokenizer: {tokenizer}")

    return tokenizer


def ner_pipeline(text):
    """
    Perform Named Entity Recognition (NER) on the given text.

    Args:
        text (str): Input text to process.

    Returns:
        tuple: A list of entities and their spans.
    """
    nlp = dacy.load(MODEL)
    doc = nlp(text)

    ents = []
    ner_spans = []

    for ent in doc.ents:

        if ent.label_ == "PERSON":
            pass

        elif ent.label_ in {"CARDINAL"}:
            ent.label_ = "CODE"

        elif ent.label_ in {"LOCATION", "FACILITY", "GPE"}:
            ent.label_ = "LOC"

        elif ent.label_ == "ORGANIZATION":
            ent.label_ = "ORG"

        elif ent.label_ in {"LANGUAGE", "NORP"}:
            ent.label_ = "DEM"

        elif ent.label_ in {"DATE", "TIME"}:
            ent.label_ = "DATETIME"

        elif ent.label_ in {"PERCENT", "QUANTITY", "MONEY", "ORDINAL"}:
            ent.label_ = "QUANTITY"

        elif ent.label_ in {"PRODUCT", "EVENT", "WORK OF ART", "LAW"}:
            ent.label_ = "MISC"

        ner_span = (ent.start_char, ent.end_char)
        ner_spans.append(ner_span)

        ent = {
            "entity_group": ent.label_,
            "word": ent.text,
            "start": ent.start_char,
            "end": ent.end_char,
        }
        ents.append(ent)

    return ents, ner_spans


def regex_pipeline(text, ner_spans):
    """
    Perform regex-based entity extraction on the given text.

    Args:
        text (str): Input text to process.
        ner_spans (list): List of spans already identified by the NER pipeline.

    Returns:
        list: List of entities extracted using regex patterns.
    """
    cpr_pattern = "|".join(
        [r"[0-3]\d{1}[0-1]\d{3}-\d{4}", r"[0-3]\d{1}[0-1]\d{3} \d{4}"]
    )

    tlf_pattern = "|".join(
        [
            r"\+\d{10}",
            r"\+\d{4} \d{2} \d{2} \d{2}",
            r"\+\d{2} \d{8}",
            r"\+\d{2} \d{2} \d{2} \d{2} \d{2}",
            r"\+\d{2} \d{4} \d{4}",
            r"\d{2} \d{4} \d{4}",
            r"\d{2} \d{4}\-\d{4}",
            r"\d{8}",
            r"\d{4} \d{4}",
            r"\d{4}\-\d{4}",
            r"\d{2} \d{2} \d{2} \d{2}",
        ]
    )

    mail_pattern = r"[\w\.-]+@[\w\.-]+(?:\.[\w]+)+"

    regex_patterns = [cpr_pattern, tlf_pattern, mail_pattern]

    ents = []

    for regex_pattern in regex_patterns:

        # Compile the combined regex pattern
        regex = re.compile(regex_pattern)

        # Find all matches in the text
        matches = regex.finditer(text)

        for match in matches:

            regex_span = (match.start(), match.end())

            if not any(
                ner_start <= regex_span[1] and regex_span[0] <= ner_end
                for ner_start, ner_end in ner_spans
            ):

                ent = {
                    "entity_group": "label",
                    "word": match.group(),
                    "start": match.start(),
                    "end": match.end(),
                }

                if regex_pattern == cpr_pattern:
                    ent["entity_group"] = "CODE"

                elif regex_pattern == tlf_pattern:
                    ent["entity_group"] = "CODE"

                elif regex_pattern == mail_pattern:
                    ent["entity_group"] = "MISC"

                ents.append(ent)

    return ents


# Function to chunk text without overlap
def tokenize_and_chunk_text(text, tokenizer, verbose, max_length=MAX_LENGTH):
    """
    Tokenize and chunk the input text into smaller segments.

    Args:
        text (str): Input text to tokenize and chunk.
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer to use for tokenization.
        verbose (bool): Flag to enable verbose logging.
        max_length (int): Maximum length of each chunk.

    Returns:
        list: List of chunks containing tokenized data and offsets.
    """
    # Tokenize
    tokens = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)

    # Save input_ids (numerical reps of tokens) and offset_mappings
    input_ids = tokens["input_ids"]
    offset_mapping = tokens["offset_mapping"]

    # Init chunk list
    chunk_counter = 0
    chunks = []

    # For each entry from 0 to length of input_ids, with an increment of 512
    for i in range(0, len(input_ids), max_length):

        input_ids_chunk = input_ids[
            i : i + max_length
        ]  # From 0 to 512, next is from 512 to 1024
        offset_mapping_chunk = offset_mapping[i : i + max_length]
        tokens_chunk = tokenizer.convert_ids_to_tokens(input_ids_chunk)
        chunk_start_i = offset_mapping_chunk[0][0]
        chunk_end_i = offset_mapping_chunk[-1][1]
        text_chunk = text[chunk_start_i:chunk_end_i]

        if verbose:

            print(f"Chunk {chunk_counter} tokens: {tokens_chunk}")
            print(f"Chunk {chunk_counter} text: {text_chunk}")
            print(f"Chunk {chunk_counter} offsets: {offset_mapping_chunk}")
            print(
                f"Chunk {chunk_counter} start/end indeces: ({chunk_start_i}, {chunk_end_i})"
            )

        chunks.append((input_ids_chunk, offset_mapping_chunk, tokens_chunk, text_chunk))

        chunk_counter += 1

    return chunks


def process_long_text(text, tokenizer, ner_pipeline, verbose):
    """
    Process long text by chunking, applying NER, and regex pipelines.

    Args:
        text (str): Input text to process.
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer to use for tokenization.
        ner_pipeline (function): NER pipeline function.
        verbose (bool): Flag to enable verbose logging.

    Returns:
        list: List of all entities extracted from the text.
    """
    chunks = tokenize_and_chunk_text(text, tokenizer, verbose)

    all_ents = []
    chunk_counter = 0

    for input_ids_chunk, offset_mapping_chunk, tokens_chunk, text_chunk in chunks:

        if not input_ids_chunk:  # Handles empty chunks
            continue

        ents, ner_spans = ner_pipeline(text_chunk)
        regex_ents = regex_pipeline(text_chunk, ner_spans)
        ents.extend(regex_ents)

        for ent in ents:
            ent["start"] += offset_mapping_chunk[0][0]
            ent["end"] += offset_mapping_chunk[0][0]

            if verbose:
                print(
                    f"Chunk {chunk_counter} entity {ent['word']} has span: {(ent['start'], ent['end'])}"
                )

        all_ents.extend(ents)

        chunk_counter += 1

    return all_ents


def get_pre_annotations(data_list, ner_pipeline, tokenizer, verbose):
    """
    Generate pre-annotations for the dataset.

    Args:
        data_list (list): List of data entries to annotate.
        ner_pipeline (function): NER pipeline function.
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer to use for tokenization.
        verbose (bool): Flag to enable verbose logging.

    Returns:
        list: Annotated dataset with predictions added.
    """
    # Loop over each file
    for entry_dict in data_list:

        text = entry_dict["data"]["text"]

        # Define the predictions_dict
        predictions_dict = {
            "model_version": dacy,  # insert model version
            "result": [],  # insert result_dict(s)
        }

        # If predictions_list is not empty (predictions are already there), skip the iteration
        if entry_dict["predictions"]:
            continue

        # Get the NER-predictions
        all_ents = process_long_text(text, tokenizer, ner_pipeline, verbose)

        for ent in all_ents:

            result_dict = {
                "from_name": "entity_mentions",
                "to_name": "doc_text",
                "type": "labels",
                "value": {
                    "start": "start",
                    "end": "end",
                    "text": "word",
                    "labels": [],  # insert label
                },
            }

            # Map the ner_result to the result_dict structure
            result_dict["value"]["start"] = ent["start"]
            result_dict["value"]["end"] = ent["end"]
            result_dict["value"]["text"] = ent["word"]
            result_dict["value"]["labels"].append(ent["entity_group"])

            predictions_dict["result"].append(result_dict)

        entry_dict["predictions"].append(predictions_dict)

        print(
            f"[INFO]: Done pre-annotating document: {entry_dict['data']['file_name']}"
        )

    return data_list


def save_json(data_list, save_path):
    """
    Save the annotated dataset to a JSON file.

    Args:
        data_list (list): Annotated dataset to save.
        save_path (str): Path to save the JSON file.
    """
    json_object = json.dumps(data_list, indent=2)

    with open(save_path, "w", encoding="utf-8") as outfile:
        outfile.write(json_object)


def main():
    """
    Main function to execute the pre-annotation pipeline.
    """
    args = parse_arguments()
    data_list = load_data(args.data_path)
    tokenizer = initiate_ner_pipeline(args.verbose)
    ner_pipeline = ner_pipeline(args.verbose)
    get_pre_annotations(data_list, ner_pipeline, tokenizer, args.verbose)
    save_json(data_list, args.save_path)


if __name__ == "__main__":
    main()
