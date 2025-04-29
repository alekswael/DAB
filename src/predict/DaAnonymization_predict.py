# Import
import argparse
import json
import dacy
import re
from transformers import AutoTokenizer

"""
This script contains a custom version of DaAnonymization,
made with the purpose of getting the masked offsets for each entity,
a function not currently supported in DaAnonymization. 
"""


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="This script is used for generating masks for anonymization using DaCy models."
    )
    parser.add_argument(
        "-d",
        "--data_path",
        type=str,
        help="The path to the annotated dataset in Label Studio JSON format.",
        default="./data/annotations_15_04_2025.json",
        required=False,
    )
    parser.add_argument(
        "-s",
        "--save_path",
        type=str,
        help="The directory for saving the predictions.",
        default="./data/predictions/",
        required=False,
    )
    parser.add_argument(
        "-fg",
        "--fine_grained",
        action="store_true",
        help="Whether to use the fine-grained NER DaCy model.",
        required=False,
    )

    return parser.parse_args()


def load_data(data_path):

    with open(data_path, "r", encoding="utf-8") as doc:
        data = json.load(doc)

    return data


def instantiate_model(fine_grained):

    tokenizer = AutoTokenizer.from_pretrained(
        "saattrupdan/nbailab-base-ner-scandi",
        clean_up_tokenization_spaces=True,
        is_split_into_words=False,
        truncation=False,
    )

    if fine_grained:
        model = dacy.load("da_dacy_large_ner_fine_grained-0.1.0")
        print("[INFO]: Using model da_dacy_large_ner_fine_grained-0.1.0")

    else:
        model = dacy.load("da_dacy_large_trf-0.2.0")
        print("[INFO]: Using model da_dacy_large_trf-0.2.0")

    return tokenizer, model


def dacy_pipeline(text, model):
    doc = model(text)

    masked_entities = []
    dacy_offsets = []

    for ent in doc.ents:

        masked_entity = ent.label_
        offset = (ent.start_char, ent.end_char)

        masked_entities.append(masked_entity)
        dacy_offsets.append(offset)

    dacy_offsets = sorted(dacy_offsets, key=lambda x: x[0], reverse=True)

    return masked_entities, dacy_offsets


def mask_dacy_predictions(text, dacy_offsets):

    dacy_text = text

    for start, end in dacy_offsets:
        if 0 <= start < end <= len(dacy_text):
            mask = "*" * (end - start)
            dacy_text = dacy_text[:start] + mask + dacy_text[end:]
        else:
            raise ValueError(
                f"Invalid span: ({start}, {end}) for text of length {len(dacy_text)}"
            )

    return dacy_text


def regex_pipeline(text, dacy_text, dacy_offsets):

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

    regex_offsets = []

    for regex_pattern in regex_patterns:

        # Compile the combined regex pattern
        regex = re.compile(regex_pattern)

        # Find all matches in the text
        matches = regex.finditer(dacy_text)

        for match in matches:

            regex_offset = (match.start(), match.end())
            regex_offsets.append(regex_offset)

    chunk_offsets = regex_offsets + dacy_offsets

    masked_text = text

    for start, end in chunk_offsets:
        if 0 <= start < end <= len(masked_text):
            mask = "*" * (end - start)
            masked_text = masked_text[:start] + mask + masked_text[end:]
        else:
            raise ValueError(
                f"Invalid span: ({start}, {end}) for text of length {len(dacy_text)}"
            )

    return chunk_offsets, masked_text


# Function to chunk text without overlap
def tokenize_and_chunk_text(text, tokenizer, max_length=512):
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

        chunks.append((input_ids_chunk, offset_mapping_chunk, tokens_chunk, text_chunk))

        chunk_counter += 1

    return chunks


def pipeline_to_chunks(chunks, model):
    all_offsets = []
    all_masked_text = ""

    for input_ids_chunk, offset_mapping_chunk, tokens_chunk, text_chunk in chunks:

        if not input_ids_chunk:  # Handles empty chunks
            continue

        _, dacy_offsets = dacy_pipeline(text_chunk, model)
        dacy_text = mask_dacy_predictions(text_chunk, dacy_offsets)
        chunk_offsets, masked_text = regex_pipeline(text_chunk, dacy_text, dacy_offsets)

        for i, offset in enumerate(chunk_offsets):
            start = offset[0] + offset_mapping_chunk[0][0]
            end = offset[1] + offset_mapping_chunk[0][0]
            chunk_offsets[i] = (start, end)

        all_offsets.extend(chunk_offsets)
        all_masked_text += masked_text

    return all_offsets, all_masked_text


def save_json(data, save_path, fine_grained):

    json_object = json.dumps(data, indent=2)

    if fine_grained:
        save_path += "DaAnonymization_FG_predictions.json"

    else:
        save_path += "DaAnonymization_predictions.json"

    with open(save_path, "w", encoding="utf-8") as outfile:
        outfile.write(json_object)


def main():
    args = parse_arguments()
    data_list = load_data(args.data_path)
    tokenizer, model = instantiate_model(args.fine_grained)

    masked_output_docs = {}

    for entry_dict in data_list:

        text = entry_dict["data"]["text"]

        chunks = tokenize_and_chunk_text(text, tokenizer)

        all_offsets, all_masked_text = pipeline_to_chunks(chunks, model)

        masked_output_docs[entry_dict["id"]] = all_offsets

        print(f"[INFO]: Masked output generated for document: {entry_dict["id"]}")
        print(f"[INFO]: Masked document: {all_masked_text}")

    save_json(masked_output_docs, args.save_path, args.fine_grained)


if __name__ == "__main__":
    main()
