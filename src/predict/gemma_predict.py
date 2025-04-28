# Import
import argparse
import os
import json
import re
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from google import genai
import torch
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

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
    # pipe = pipeline(
    #     "text-generation",
    #     model="google/gemma-2-9b-it",
    #     model_kwargs={"torch_dtype": torch.bfloat16},
    #     device="cpu"
    # )

    pipe = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

    print("List of models that support generateContent:\n")
    for m in pipe.models.list():
        for action in m.supported_actions:
            if action == "generateContent":
                print(m.name)

    return pipe

def process_gemma_output(text):
    spans = []
    new_text_parts = []
    current_pos = 0
    pattern = r"\[\[\[(.*?)\]\]\]"
    last_end = 0

    for match in re.finditer(pattern, text):
        start, end = match.span()
        inner_text = match.group(1)
        
        # Text before brackets
        before = text[last_end:start]
        new_text_parts.append(before)
        current_pos += len(before)

        # Replace inner text
        modified = '*' * len(inner_text)
        new_text_parts.append(modified)

        # Save span
        spans.append((current_pos, current_pos + len(modified)))

        current_pos += len(modified)
        last_end = end

    # Add remaining text after last match
    after = text[last_end:]
    new_text_parts.append(after)

    new_text = ''.join(new_text_parts)
    return new_text, spans

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

    gemma_output = pipe.models.generate_content(
    model="gemma-3-12b-it",
    contents=prompt,
    )

    print(gemma_output.text)

    return gemma_output

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

        gemma_output = gemma_pipeline(text, pipe)

        masked_text, offsets = process_gemma_output(gemma_output.text)

        masked_output_docs[entry_dict["id"]] = offsets

        print(f"[INFO]: Masked output generated for document: {entry_dict["id"]}")
        print(f"[INFO]: Masked document: {masked_text}")
    
    save_json(masked_output_docs, args.save_path)

if __name__ == "__main__":
    main()