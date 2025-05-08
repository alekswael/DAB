# Import
import argparse
import os
import json
import re
from transformers import pipeline
import torch

# Constants
PROMPT_PATH = "./src/predict/hf_pipeline_instruction_prompt.txt"


def parse_arguments():
    """
    Parses command-line arguments for the script.

    Returns:
        argparse.Namespace: Parsed arguments including data_path, save_path, model_name, and device.
    """
    parser = argparse.ArgumentParser(
        description="This script is used for generating masks for anonymization using any Hugging Face model."
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
        required=True,
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="The name of the Hugging Face model to use.",
        required=True,
    )
    parser.add_argument(
        "--device",
        type=str,
        help="Device to run the model on (e.g., 'cpu', 'cuda').",
        default="cpu",
        required=False,
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


def instantiate_pipeline(model_name, device):
    """
    Instantiates a Hugging Face text-generation pipeline.

    Args:
        model_name (str): Name of the Hugging Face model to use.
        device (str): Device to run the model on ('cpu' or 'cuda').

    Returns:
        transformers.Pipeline: Hugging Face pipeline object.
    """
    print(f"[INFO]: Loading model '{model_name}' on device '{device}'...")
    pipe = pipeline(
        "text-generation",
        model=model_name,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device=0 if device == "cuda" else -1,
    )
    return pipe


def process_model_output(text):
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


def import_prompt():
    """
    Imports the instruction prompt from a text file.

    Returns:
        str: Instruction prompt as a string.
    """
    # Get instruction prompt from .txt file
    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        instruction_prompt = f.read()

    return instruction_prompt


def generate_output(text, pipe, instruction_prompt):
    """
    Generates output text using the Hugging Face pipeline.

    Args:
        text (str): Input text to process.
        pipe (transformers.Pipeline): Hugging Face pipeline object.
        instruction_prompt (str): Instruction prompt to prepend to the input text.

    Returns:
        str: Generated output text.
    """
    # Combine instruction and user prompt
    user_prompt = f"Text:\n\n{text}\n\nOutput:\n\n"
    full_prompt = instruction_prompt + user_prompt
    output = pipe(full_prompt, return_full_text=False, max_new_tokens=8192)
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
    args = parse_arguments()
    data_list = load_data(args.data_path)
    pipe = instantiate_pipeline(args.model_name, args.device)
    instruction_prompt = import_prompt()

    masked_offsets = {}
    masked_text = {}

    for entry_dict in data_list:
        text = entry_dict["data"]["text"]

        print(f"[INFO]: Generating mask for document {entry_dict['id']}...")

        output_text = generate_output(text, pipe, instruction_prompt)
        all_masked_text, all_offsets = process_model_output(output_text)

        masked_offsets[entry_dict["id"]] = all_offsets
        masked_text[entry_dict["id"]] = all_masked_text

        output_format = [masked_offsets, masked_text]

        print(f"[INFO]: Masked output generated for document: {entry_dict['id']}")
        print(f"[INFO]: Masked document: {all_masked_text}")

    save_json(output_format, args.save_path)


if __name__ == "__main__":
    main()
