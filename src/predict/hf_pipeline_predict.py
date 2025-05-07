# Import
import argparse
import os
import json
import re
from transformers import pipeline
import torch


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="This script is used for generating masks for anonymization using any Hugging Face model."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        help="The path to the annotated dataset in Label Studio JSON format.",
        default="./data/annotations_15_04_2025.json",
        required=False,
    )
    parser.add_argument(
        "--save_path",
        type=str,
        help="The path for saving the predictions.",
        default="./output/predictions/",
        required=False,
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
    with open(data_path, "r", encoding="utf-8") as doc:
        data_list = json.load(doc)
    return data_list


def instantiate_pipeline(model_name, device):
    print(f"[INFO]: Loading model '{model_name}' on device '{device}'...")
    pipe = pipeline(
        "text-generation",
        model=model_name,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device=0 if device == "cuda" else -1,
    )
    return pipe


def process_model_output(text):
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


def generate_output(text, pipe):
    instruction_prompt = """Instruction:

You are a GDPR text anonymization assistant. Your job is to anonymize Danish text by masking character spans containing personal information, as to ensure that none of the people mentioned in the text can be directly or indirectly identified. You should disregard the notion of publicly available knowledge and consider all direct and indirect identifiers as personal information. To mask a character span in the text, mark it in triple hard brackets.

Here is an example:

Text:

Du kan prøve at ringe til Mogens Petersen, han har tlf nr 80 90 31 23. Han driver en barbershop og bor selv i centrum af Hillerød.

Output:

Du kan prøve at ringe til [[[Mogens Petersen]]], han har tlf nr [[[80 90 31 23]]]. Han ejer en [[[barbershop]]] og bor selv i [[[centrum af Hillerød]]].

Now it's your turn.

"""

    user_prompt = f"Text:\n\n{text}\n\nOutput:\n\n"

    full_prompt = instruction_prompt + user_prompt
    output = pipe(full_prompt, return_full_text=False, max_new_tokens=8192)
    output_text = output[0]["generated_text"]

    return output_text


def save_json(data_list, model_name, save_path):

    # Change the / in the model_name to _ for the filename
    model_name = model_name.replace("/", "_")

    save_path = f"{save_path}{model_name}_predictions.json"

    json_object = json.dumps(data_list, indent=2)
    with open(save_path, "w", encoding="utf-8") as outfile:
        outfile.write(json_object)


def main():
    args = parse_arguments()
    data_list = load_data(args.data_path)
    pipe = instantiate_pipeline(args.model_name, args.device)

    masked_offsets = {}
    masked_text = {}

    for entry_dict in data_list:
        text = entry_dict["data"]["text"]

        print(f"[INFO]: Generating mask for document {entry_dict['id']}...")

        output_text = generate_output(text, pipe)
        all_masked_text, all_offsets = process_model_output(output_text)

        masked_offsets[entry_dict["id"]] = all_offsets
        masked_text[entry_dict["id"]] = all_masked_text

        output_format = [masked_offsets, masked_text]

        print(f"[INFO]: Masked output generated for document: {entry_dict['id']}")
        print(f"[INFO]: Masked document: {all_masked_text}")

    save_json(output_format, args.model_name, args.save_path)


if __name__ == "__main__":
    main()
