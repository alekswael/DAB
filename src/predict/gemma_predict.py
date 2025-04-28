# Import
import argparse
import json
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch

def parse_arguments():
    parser = argparse.ArgumentParser(description="This script is used for generating masks for anonymization using the google/gemma-2-9b-it model.")
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
        default="/home/aleksander/projects/DAB/data/predictions/Gemma_predictions.json",
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

def instantiate_pipeline():
    pipe = pipeline(
        "text-generation",
        model="google/gemma-2-9b-it",
        model_kwargs={"torch_dtype": torch.bfloat16},
        device="cpu",  # replace with "mps" to run on a Mac device
    )

    return pipe

def gemma_pipeline(text, pipe):

    prompt = f'''Instruction:

    You are a GDPR text anonymization assistant. Your job is to anonymize Danish text by masking character spans containing personal information, as to ensure that none of the people mentioned in the text can be directly or indirectly identified. You should disregard the notion of publicly available knowledge and consider all direct and indirect identifiers as personal information. To mask a character span in the text, mark it in triple hard brackets.

    Here is an example:

    Text:

    Du kan prøve at ringe til Mogens Petersen, han har tlf nr 80 90 31 23. Han driver en barbershop og selv i centrum af Hillerød.

    Output:

    Du kan prøve at ringe til [[[Mogens Petersen]]], han har tlf nr [[[80 90 31 23]]]. Han ejer en [[[barbershop]]] og selv i [[[centrum af Hillerød]]].

    Now it's your turn.

    Text:

    {text}

    Output:

    
    '''

    messages = [
        {"role": "user", "content": prompt},
    ]

    outputs = pipe(messages, max_new_tokens=8192)
    assistant_response = outputs[0]["generated_text"][-1]["content"].strip()

    print(assistant_response)

    return masked_text, offsets

def save_json(data, save_path):

    json_object = json.dumps(data, indent=2)

    with open(save_path, "w", encoding="utf-8") as outfile:
        outfile.write(json_object)

def main():
    args = parse_arguments()
    data_list = load_data(args.data_path, args.debug)
    pipe = instantiate_pipeline()

    masked_output_docs = {}

    for entry_dict in data_list:

        text = entry_dict["data"]["text"]

        print(f"[INFO]: Generating mask for document {entry_dict["id"]}...")
        masked_text, offsets = gemma_pipeline(text, pipe)

        masked_output_docs[entry_dict["id"]] = offsets

        print(f"[INFO]: Masked output generated for document: {entry_dict["id"]}")
        print(f"[INFO]: Masked document: {masked_text}")
    
    save_json(masked_output_docs, args.save_path)

if __name__ == "__main__":
    main()