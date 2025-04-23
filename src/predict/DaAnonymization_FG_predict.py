# Import
import argparse
import json
import dacy
import re

'''
This script contains a custom version of DaAnonymization,
made with the purpose of getting the masked offsets for each entity,
a function not currently supported in DaAnonymization. 
'''

def parse_arguments():
    parser = argparse.ArgumentParser(description="This script is used for generating masks for anonymization using the da_dacy_large_ner_fine_grained-0.1.0 model.")
    parser.add_argument(
        '-d', '--data_path',
        type=str, 
        help='The path to the annotated dataset in Label Studio JSON format.',
        default='/home/aleksander/projects/DAB/data/annotations_15_04_2025.json',
        required=False
    )
    parser.add_argument(
        '-s', '--save_path', 
        type=str, 
        help='The path for saving the predictions.',
        default="/home/aleksander/projects/DAB/data/predictions/DaAnonymization_FG_predictions.json",
        required=False
    )
    parser.add_argument(
        '-db', '--debug', 
        action="store_true", 
        help='Set to debug mode.',
        required=False
    )

    return parser.parse_args()

def load_data(data_path, debug):

    with open(data_path, "r", encoding="utf-8") as doc:
        data = json.load(doc)
    
    return data

def instantiate_model():

    model = dacy.load("da_dacy_large_ner_fine_grained-0.1.0")

    return model

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
            mask = '*' * (end - start)
            dacy_text = dacy_text[:start] + mask + dacy_text[end:]
        else:
            raise ValueError(f"Invalid span: ({start}, {end}) for text of length {len(dacy_text)}")

    return dacy_text

def regex_pipeline(text, dacy_text, dacy_offsets):

    cpr_pattern = "|".join(
        [
            r"[0-3]\d{1}[0-1]\d{3}-\d{4}",
            r"[0-3]\d{1}[0-1]\d{3} \d{4}"
        ]
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
    
    all_offsets = regex_offsets + dacy_offsets

    masked_text = text

    for start, end in all_offsets:
        if 0 <= start < end <= len(masked_text):
            mask = '*' * (end - start)
            masked_text = masked_text[:start] + mask + masked_text[end:]
        else:
            raise ValueError(f"Invalid span: ({start}, {end}) for text of length {len(dacy_text)}")

    
    return all_offsets, masked_text

def save_json(data, save_path):

    json_object = json.dumps(data, indent=2)

    with open(save_path, "w", encoding="utf-8") as outfile:
        outfile.write(json_object)

def main():
    args = parse_arguments()
    data_list = load_data(args.data_path, args.debug)
    model = instantiate_model()

    masked_output_docs = {}

    for entry_dict in data_list:

        text = entry_dict["data"]["text"]

        _, dacy_offsets = dacy_pipeline(text, model)
        dacy_text = mask_dacy_predictions(text, dacy_offsets)
        all_offsets, masked_text = regex_pipeline(text, dacy_text, dacy_offsets)

        masked_output_docs[entry_dict["id"]] = all_offsets

        print(f"[INFO]: Masked output generated for document: {entry_dict["id"]}")
        print(f"[INFO]: Masked document: {masked_text}")
    
    save_json(masked_output_docs, args.save_path)

if __name__ == "__main__":
    main()