# Import
import argparse
import re
import json
from transformers import pipeline, AutoTokenizer
import dacy

def parse_arguments():
    parser = argparse.ArgumentParser(description="This script is used for generating the pre-annotations for the annotation pipeline with DaCy + ReGex.")
    parser.add_argument(
        '-d', '--data_path',
        type=str, 
        help='The path to the dataset in Label Studio JSON format.',
        default='/home/aleksander/projects/DAB/data/DAB_dataset.json',
        required=False
    )
    parser.add_argument(
        '-s', '--save_path', 
        type=str, 
        help='The path for saving the pre-annotated JSON dataset.',
        default="/home/aleksander/projects/DAB/data/DAB_dataset_pre_annotated.json",
        required=False
    )
    parser.add_argument(
        '-m', '--model', 
        type=str, 
        help='The NER-model used for generating pre-annotations.',
        default="dacy",
        required=False
    )
    parser.add_argument(
        '-db', '--debug', 
        action="store_true", 
        help='Set to debug mode.',
        required=False
    )
    parser.add_argument(
    '-t', '--test', 
    action="store_true", 
    help='Set to test mode.',
    required=False
    )

    return parser.parse_args()

def load_data(data_path, debug, test):
    with open(data_path, "r", encoding="utf-8") as doc:
        data = json.load(doc)

        if test:
            test_file = "pvs_5.pdf"

            for entry_dict in data:
                if entry_dict["data"]["file_name"] == test_file:
                    data = [entry_dict]
                    break

        if debug:
            print(f"Data: {data}")
    
    return data

def initiate_ner_pipeline(ner_model, debug):
    """Initiate NER-pipeline. Im using ScandiNER."""
    print(f"[INFO]: Initiating {ner_model} model...")

    tokenizer = AutoTokenizer.from_pretrained("saattrupdan/nbailab-base-ner-scandi",
                                    clean_up_tokenization_spaces=True,
                                    is_split_into_words=False,
                                    truncation=False
                                    )
    
    if debug:
        print(tokenizer.init_kwargs)
        print(tokenizer)

    if ner_model == "scandi_ner":

        def ner_pipeline(text):

            nlp = pipeline(
                task='ner', 
                model="saattrupdan/nbailab-base-ner-scandi",
                tokenizer = tokenizer,
                aggregation_strategy='first')
            
            ents = pipeline(text)
            ner_spans = []

            for ent in ents:

                if ent["entity_group"] == "PER":
                    ent["entity_group"] = "PERSON"
                
                ner_span = (ent["start"], ent["end"])
                ner_spans.append(ner_span)
            
            return ents, ner_spans
    
    if ner_model == "dacy":

        def ner_pipeline(text):

            nlp = dacy.load("da_dacy_medium_ner_fine_grained-0.1.0")
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
                    'entity_group': ent.label_,
                    'word': ent.text,
                    'start': ent.start_char,
                    'end': ent.end_char
                }
                ents.append(ent)
            
            return ents, ner_spans

    return ner_pipeline, tokenizer

def regex_pipeline(text, ner_spans):

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

    ents = []

    for regex_pattern in regex_patterns:

        # Compile the combined regex pattern
        regex = re.compile(regex_pattern)
        
        # Find all matches in the text
        matches = regex.finditer(text)

        for match in matches:

            regex_span = (match.start(), match.end())

            if not any(ner_start <= regex_span[1] and regex_span[0] <= ner_end for ner_start, ner_end in ner_spans):

                ent = {
                    'entity_group': "label",
                    'word': match.group(),
                    'start': match.start(),
                    'end': match.end()
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
def tokenize_and_chunk_text(text, tokenizer, debug, max_length=512):
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

        input_ids_chunk = input_ids[i : i + max_length] # From 0 to 512, next is from 512 to 1024
        offset_mapping_chunk = offset_mapping[i : i + max_length]
        tokens_chunk = tokenizer.convert_ids_to_tokens(input_ids_chunk)
        chunk_start_i = offset_mapping_chunk[0][0]
        chunk_end_i = offset_mapping_chunk[-1][1]
        text_chunk = text[chunk_start_i:chunk_end_i]

        if debug:

            print(f"Chunk {chunk_counter} tokens: {tokens_chunk}")
            print(f"Chunk {chunk_counter} text: {text_chunk}")
            print(f"Chunk {chunk_counter} offsets: {offset_mapping_chunk}")
            print(f"Chunk {chunk_counter} start/end indeces: ({chunk_start_i}, {chunk_end_i})")

        chunks.append((input_ids_chunk, offset_mapping_chunk, tokens_chunk, text_chunk))

        chunk_counter += 1

    return chunks

def process_long_text(text, tokenizer, ner_pipeline, debug):

    chunks = tokenize_and_chunk_text(text, tokenizer, debug)

    all_ents = []
    chunk_counter = 0

    for input_ids_chunk, offset_mapping_chunk, tokens_chunk, text_chunk in chunks:

        if not input_ids_chunk: #Handles empty chunks
            continue

        ents, ner_spans = ner_pipeline(text_chunk)
        regex_ents = regex_pipeline(text_chunk, ner_spans)
        ents.extend(regex_ents)

        if debug:
            print(f"Regex ents: {regex_ents}")
            print(f"Ents: {ents}")

        for ent in ents:
            ent["start"] += offset_mapping_chunk[0][0] #+ chunk_start
            ent["end"] += offset_mapping_chunk[0][0] #+ chunk_start

            if debug:
                print(f"Chunk {chunk_counter} entity {ent["word"]} has span: {(ent["start"], ent["end"])}")

        all_ents.extend(ents)

        chunk_counter += 1

    return all_ents

def get_pre_annotations(data_list, ner_model, ner_pipeline, tokenizer, debug):
    # Loop over each file
    for entry_dict in data_list:

        text = entry_dict["data"]["text"]

        # Define the predictions_dict
        predictions_dict = {
            "model_version": ner_model, # insert model version
            "result": [] # insert result_dict(s)
        }

        # If predictions_list is not empty (predictions are already there), skip the iteration
        if entry_dict["predictions"]:
            continue

        # Get the NER-predictions
        all_ents = process_long_text(text, tokenizer, ner_pipeline, debug)
        
        for ent in all_ents:

            result_dict = {
                "from_name": "entity_mentions",
                "to_name": "doc_text",
                "type": "labels",
                "value": {
                    "start": "start",
                    "end": "end",
                    "text": "word",
                    "labels": [] # insert label
                }
            }


            # Map the ner_result to the result_dict structure
            result_dict["value"]["start"] = ent["start"]
            result_dict["value"]["end"] = ent["end"]
            result_dict["value"]["text"] = ent["word"]
            result_dict["value"]["labels"].append(ent["entity_group"])

            predictions_dict["result"].append(result_dict)
        
        entry_dict["predictions"].append(predictions_dict)

        print(f"Done pre-annotating document: {entry_dict["data"]["file_name"]}")
    
    return data_list

def save_json(data, save_path):

    json_object = json.dumps(data, indent=2)

    with open(save_path, "w", encoding="utf-8") as outfile:
        outfile.write(json_object)

def main():
    args = parse_arguments()
    data_list = load_data(args.data_path, args.debug, args.test)
    ner_pipeline, tokenizer = initiate_ner_pipeline(args.model, args.debug)
    get_pre_annotations(data_list, args.model, ner_pipeline, tokenizer, args.debug)
    save_json(data_list, args.save_path)

if __name__ == "__main__":
    main()
