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
api_key = os.getenv("GOOGLE_API_KEY")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="This script is used for generating masks for anonymization using the google/gemma-3-12b-it model."
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
        default="./output/predictions/Gemma_predictions.json",
        required=False,
    )
    parser.add_argument(
        "--cloud",
        action="store_true",
        help="If True, the model will be run through Google's API using a free-tier key (as of May 5th, 2025).",
        default=True,
    )

    return parser.parse_args()


def load_data(data_path):

    with open(data_path, "r", encoding="utf-8") as doc:
        data_list = json.load(doc)

    return data_list


def instantiate_pipeline(cloud):

    if cloud == True:
        print(f"[INFO]: Running model through Google's API...")
        pipe = genai.Client(api_key=api_key)

    else:
        print(f"[INFO]: Running model locally...")
        pipe = pipeline(
            "text-generation",
            model="google/gemma-3-12b-it",
            model_kwargs={"torch_dtype": torch.bfloat16},
            device="cpu",
        )

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


def generate_output(text, pipe, cloud):

    if cloud == True:

        prompt = f"""Instruction:

You are a GDPR text anonymization assistant. Your job is to anonymize Danish text by masking character spans containing personal information, as to ensure that none of the people mentioned in the text can be directly or indirectly identified. You should disregard the notion of publicly available knowledge and consider all direct and indirect identifiers as personal information. To mask a character span in the text, mark it in triple hard brackets.

Here is an example:

Text:

Du kan prøve at ringe til Mogens Petersen, han har tlf nr 80 90 31 23. Han driver en barbershop og bor selv i centrum af Hillerød.

Output:

Du kan prøve at ringe til [[[Mogens Petersen]]], han har tlf nr [[[80 90 31 23]]]. Han ejer en [[[barbershop]]] og bor selv i [[[centrum af Hillerød]]].

Now it's your turn.

Text:

{text}

Output:


"""

        output = pipe.models.generate_content(
            model="gemma-3-12b-it",
            contents=prompt,
        )

        print(output.text)

        return output.text

    else:

        instruction_prompt = f"""Instruction:

You are a GDPR text anonymization assistant. Your job is to anonymize Danish text by masking character spans containing personal information, as to ensure that none of the people mentioned in the text can be directly or indirectly identified. You should disregard the notion of publicly available knowledge and consider all direct and indirect identifiers as personal information. To mask a character span in the text, mark it in triple hard brackets.

Here is an example:

Text:

Du kan prøve at ringe til Mogens Petersen, han har tlf nr 80 90 31 23. Han driver en barbershop og bor selv i centrum af Hillerød.

Output:

Du kan prøve at ringe til [[[Mogens Petersen]]], han har tlf nr [[[80 90 31 23]]]. Han ejer en [[[barbershop]]] og bor selv i [[[centrum af Hillerød]]].

Now it's your turn.

"""

        user_prompt = f"""Text:

{text}

Output:


"""

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": instruction_prompt}],
            },
            {"role": "user", "content": [{"type": "text", "text": user_prompt}]},
        ]

        output = pipe(text=messages)
        output_text = output[0]["generated_text"][-1]["content"]

        print(output_text)

        return output_text


def save_json(data_list, save_path):

    json_object = json.dumps(data_list, indent=2)

    with open(save_path, "w", encoding="utf-8") as outfile:
        outfile.write(json_object)


def main():
    args = parse_arguments()
    data_list = load_data(args.data_path)
    pipe = instantiate_pipeline(args.cloud)

    masked_offsets = {}
    masked_text = {}

    for entry_dict in data_list:

        text = entry_dict["data"]["text"]

        print(f"[INFO]: Generating mask for document {entry_dict["id"]}...")

        output_text = generate_output(text, pipe, args.cloud)

        all_masked_text, all_offsets = process_gemma_output(output_text)

        masked_offsets[entry_dict["id"]] = all_offsets
        masked_text[entry_dict["id"]] = all_masked_text

        output_format = [masked_offsets, masked_text]

        print(f"[INFO]: Masked output generated for document: {entry_dict["id"]}")
        print(f"[INFO]: Masked document: {masked_text}")

    save_json(output_format, args.save_path)


if __name__ == "__main__":
    main()
