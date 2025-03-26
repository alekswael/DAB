# Import
import argparse
import json
from transformers import pipeline, AutoTokenizer

def parse_arguments():
    parser = argparse.ArgumentParser(description="This script is used for compiling the documents into a single JSON file, using the Label Studio format.")
    parser.add_argument(
        '-d', '--data_path',
        type=str, 
        help='The path to the dataset in Label Studio JSON format.',
        default='/home/aleksander/projects/DAB/data/DAB_text_format.json',
        required=False
    )
    parser.add_argument(
        '-s', '--save_path', 
        type=str, 
        help='The path for saving the pre-annotated JSON dataset.',
        default="data/DAB_dataset_pre_annotated.json",
        required=False
    )
    parser.add_argument(
        '-m', '--model', 
        type=str, 
        help='The NER-model used for generating pre-annotations.',
        default="saattrupdan/nbailab-base-ner-scandi",
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

        if debug:
            print(f"Data: {data}")
    
    return data

def initiate_ner_pipeline(ner_model, debug):
    """Initiate NER-pipeline. Im using ScandiNER."""
    print("Initiating model...")

    tokenizer = AutoTokenizer.from_pretrained(ner_model,
                                              clean_up_tokenization_spaces=True,
                                              is_split_into_words=False,
                                              truncation=False
                                              )

    if debug:
        print(tokenizer.init_kwargs)
        print(tokenizer)

    ner_pipeline = pipeline(
        task='ner', 
        model=ner_model,
        tokenizer = tokenizer,
        aggregation_strategy='first')

    return ner_pipeline, tokenizer

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
        text_chunk = text[offset_mapping_chunk[0][0]:offset_mapping_chunk[-1][1]]

        if debug:

            print(f"Chunk {chunk_counter} tokens: {tokens_chunk}")
            print(f"Chunk {chunk_counter} text: {text_chunk}")
            print(f"Chunk {chunk_counter} offsets: {offset_mapping_chunk}")

        chunks.append((input_ids_chunk, offset_mapping_chunk, tokens_chunk, text_chunk))

        chunk_counter += 1

    return chunks

def process_long_text(text, tokenizer, ner_pipeline, debug):

    chunks = tokenize_and_chunk_text(text, tokenizer, debug)

    all_entities = []
    chunk_counter = 0

    for input_ids_chunk, offset_mapping_chunk, text_chunk in chunks:

        if not input_ids_chunk: #Handles empty chunks
            continue

        entities = ner_pipeline(text_chunk)

        for entity in entities:
            # Adjust entity positions based on chunk offset
            if not chunk_counter == 0:
                entity["start"] += offset_mapping_chunk[0][0] #+ chunk_start
                entity["end"] += offset_mapping_chunk[0][0] #+ chunk_start

            if debug:
                print(f"Chunk {chunk_counter} entity {entity["word"]} has span: {(entity["start"], entity["end"])}")

        all_entities.extend(entities)

        chunk_counter += 1

    return all_entities

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
        ner_results = process_long_text(text, tokenizer, ner_pipeline, debug)
        
        for result in ner_results:

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

            # Change PER to PERSON
            if result["entity_group"] == "PER":
                result["entity_group"] = "PERSON"

            # Map the ner_result to the result_dict structure
            result_dict["value"]["start"] = result["start"]
            result_dict["value"]["end"] = result["end"]
            result_dict["value"]["text"] = result["word"]
            result_dict["value"]["labels"].append(result["entity_group"])

            predictions_dict["result"].append(result_dict)
        
        entry_dict["predictions"].append(predictions_dict)

        print(f"Done pre-annotating document: {entry_dict["data"]["file_name"]}")
    
    return data_list

def save_json(data, save_path):

    # Serializing json
    json_object = json.dumps(data, indent=2)
    
    # Writing to sample.json
    with open(save_path, "w", encoding="utf-8") as outfile:
        outfile.write(json_object)

def main():
    args = parse_arguments()
    data_list = load_data(args.data_path, args.debug)
    ner_pipeline, tokenizer = initiate_ner_pipeline(args.model, args.debug)
    get_pre_annotations(data_list, args.model, ner_pipeline, tokenizer, args.debug)
    save_json(data_list, args.save_path)

if __name__ == "__main__":
    main()