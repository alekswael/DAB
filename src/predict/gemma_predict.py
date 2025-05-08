# Import
import argparse
import os
import json
import re
from transformers import pipeline
from google import genai
import torch
from dotenv import load_dotenv

# Constants
LOCAL_MODEL = "google/gemma-3-12b-it"
CLOUD_MODEL = "gemma-3-12b-it"
MAX_NEW_TOKENS = 8192 # Maximum number of tokens for model output


def parse_arguments():
    """
    Parses command-line arguments for the script.

    Returns:
        argparse.Namespace: Parsed arguments including data_path, save_path, and cloud flag.
    """
    parser = argparse.ArgumentParser(
        description="This script is used for generating masks for anonymization using the google/gemma-3-12b-it model."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        help="The path to the annotated dataset in Label Studio JSON format.",
        default="./data/DAB_annotated_dataset.json",
        required=False,
    )
    parser.add_argument(
        "--save_path",
        type=str,
        help="The path for saving the predictions.",
        default="./output/predictions/Gemma_predictions.json",
        required=False,
    )
    parser.add_argument(
        "--cloud",
        action="store_true",
        help="If True, the model will be run through Google's API using a free-tier key (as of May 5th, 2025).",
        default=False,
    )

    return parser.parse_args()


def load_data(data_path):
    """
    Loads data from a JSON file.

    Args:
        data_path (str): Path to the JSON file.

    Returns:
        list: List of data entries loaded from the JSON file.
    """
    with open(data_path, "r", encoding="utf-8") as doc:
        data_list = json.load(doc)

    return data_list


def instantiate_pipeline(cloud):
    """
    Instantiates a text-generation pipeline or connects to Google's API.

    Args:
        cloud (bool): Whether to use Google's API or a local model.

    Returns:
        object: Hugging Face pipeline or Google GenAI client.
    """
    if cloud == True:
        print(f"[INFO]: Running model through Google's API...")
        api_key = os.getenv("GOOGLE_API_KEY")
        pipe = genai.Client(api_key=api_key)

    else:
        print(f"[INFO]: Running model locally...")
        pipe = pipeline(
            "text-generation",
            model=LOCAL_MODEL,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device="cpu",
        )

    return pipe


def process_gemma_output(text):
    """
    Processes the model output to replace masked spans with asterisks and extract offsets.

    Args:
        text (str): Text containing masked spans in the format [[[masked_text]]].

    Returns:
        tuple: Processed text with masked spans replaced and a list of offsets.
    """
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
        modified = "*" * len(inner_text)
        new_text_parts.append(modified)

        # Save span
        spans.append((current_pos, current_pos + len(modified)))

        current_pos += len(modified)
        last_end = end

    # Add remaining text after last match
    after = text[last_end:]
    new_text_parts.append(after)

    new_text = "".join(new_text_parts)
    return new_text, spans


def import_prompt(path="./src/predict/hf_pipeline_instruction_prompt.txt"):
    """
    Imports the instruction prompt from a text file.

    Args:
        path (str): Path to the instruction prompt file.

    Returns:
        str: Instruction prompt as a string.
    """
    # Get instruction prompt from .txt file
    with open(path, "r", encoding="utf-8") as f:
        instruction_prompt = f.read()

    return instruction_prompt


def generate_output(text, pipe, instruction_prompt, cloud):
    """
    Generates output text using the pipeline or Google's API.

    Args:
        text (str): Input text to process.
        pipe (object): Hugging Face pipeline or Google GenAI client.
        instruction_prompt (str): Instruction prompt to prepend to the input text.
        cloud (bool): Whether to use Google's API or a local model.

    Returns:
        str: Generated output text.
    """
    # Combine instruction and user prompt
    user_prompt = f"Text:\n\n{text}\n\nOutput:\n\n"

    full_prompt = instruction_prompt + user_prompt

    ##### Google API #####
    if cloud == True:

        output = pipe.models.generate_content(
            model=CLOUD_MODEL,
            contents=full_prompt,
        )

        return output.text

    ##### Local Model #####
    else:

        output = pipe(full_prompt, return_full_text=False, max_new_tokens=MAX_NEW_TOKENS)
        output_text = output[0]["generated_text"]

        return output_text


def save_json(data_list, save_path):
    """
    Saves data to a JSON file.

    Args:
        data_list (list): Data to save.
        save_path (str): Path to save the JSON file.
    """
    json_object = json.dumps(data_list, indent=2)

    with open(save_path, "w", encoding="utf-8") as outfile:
        outfile.write(json_object)


def main():
    """
    Main function to execute the script. Parses arguments, loads data, processes it using a model,
    and saves the output.
    """
    load_dotenv()
    args = parse_arguments()
    data_list = load_data(args.data_path)
    pipe = instantiate_pipeline(args.cloud)
    instruction_prompt = import_prompt()

    masked_offsets = {}
    masked_text = {}

    for entry_dict in data_list:

        text = entry_dict["data"]["text"]

        print(f"[INFO]: Generating mask for document {entry_dict["id"]}...")

        output_text = generate_output(text, pipe, instruction_prompt, args.cloud)

        all_masked_text, all_offsets = process_gemma_output(output_text)

        masked_offsets[entry_dict["id"]] = all_offsets
        masked_text[entry_dict["id"]] = all_masked_text

        output_format = [masked_offsets, masked_text]

        print(f"[INFO]: Masked output generated for document: {entry_dict["id"]}")
        print(f"[INFO]: Masked document: {masked_text}")

    save_json(output_format, args.save_path)


if __name__ == "__main__":
    main()
