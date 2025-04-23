# Import
import argparse
import json

def parse_arguments():
    parser = argparse.ArgumentParser(description="This script is used for checking the offsets of the annotated masks after exporting from Label Studio.")
    parser.add_argument(
        '-d', '--data_path',
        type=str, 
        help='The path to the annotated dataset in Label Studio JSON format.',
        default='/home/aleksander/projects/DAB/data/annotations_15_04_2025.json',
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
            print(f"Data: {json.dumps(data, indent = 2)}")
            print(data[0]["annotations"][0]["result"][0]["value"]["text"])

    return data

def get_masked_text(data, index=20):

    text = data[index]["data"]["text"]

    masked_entities = []
    offsets = []

    for result_dict in data[index]["annotations"][0]["result"]:

        if result_dict["type"]=="labels":

            masked_entity = result_dict["value"]["text"]
            offset = (result_dict["value"]["start"], result_dict["value"]["end"])

            masked_entities.append(masked_entity)
            offsets.append(offset)
    
    offsets = sorted(offsets, key=lambda x: x[0], reverse=True)
    
    masked_text = text

    for start, end in offsets:
        if 0 <= start < end <= len(masked_text):
            mask = '*' * (end - start)
            masked_text = masked_text[:start] + mask + masked_text[end:]
        else:
            raise ValueError(f"Invalid span: ({start}, {end}) for text of length {len(masked_text)}")

    return text, masked_text

def main():
    args = parse_arguments()
    data = load_data(args.data_path, args.debug)
    text, masked_text = get_masked_text(data)

    print(f"ORIGINAL TEXT: {text} \n MASKED TEXT: {masked_text}")

if __name__ == "__main__":
    main()