# Import
import argparse
import json


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="This script is used for adding unique entity IDs to the annotations."
    )
    parser.add_argument(
        "-d",
        "--data_path",
        type=str,
        help="The path to the annotated dataset in Label Studio JSON format.",
        default="./data/annotations_15_04_2025.json",
        required=False,
    )
    parser.add_argument(
        "-s",
        "--save_path",
        type=str,
        help="The path for saving the pre-annotated JSON dataset. If None, overwrites the data_path JSON file.",
        required=False,
    )
    parser.add_argument(
        "-db", "--debug", action="store_true", help="Set to debug mode.", required=False
    )

    return parser.parse_args()


def load_data(data_path, debug):
    with open(data_path, "r", encoding="utf-8") as doc:
        data = json.load(doc)

        if debug:
            print(f"Data: {json.dumps(data, indent = 2)}")
            print(data[0]["annotations"][0]["result"][0]["value"]["text"])

    return data


def add_entity_ids(data_list):

    # Loop through all entities per document and ad to a set

    # Per doc
    for entry_dict in data_list:

        entities = set()

        # Per annotation
        for annotation_dict in entry_dict["annotations"]:

            # Per entity
            for result_dict in annotation_dict["result"]:

                if result_dict["type"] == "labels":

                    entities.add(result_dict["value"]["text"])

        # Add unique IDs to the set
        id_map = {item: idx for idx, item in enumerate(entities)}

        # Per annotation
        for annotation_dict in entry_dict["annotations"]:

            # Per entity
            for result_dict in annotation_dict["result"]:

                if result_dict["type"] == "labels":

                    result_dict["entity_id"] = id_map[result_dict["value"]["text"]]

    return data_list


def save_json(data, save_path):

    json_object = json.dumps(data, indent=2)

    with open(save_path, "w", encoding="utf-8") as outfile:
        outfile.write(json_object)


def main():
    args = parse_arguments()

    if args.save_path is None:
        args.save_path = args.data_path

    data = load_data(args.data_path, args.debug)
    data = add_entity_ids(data)
    save_json(data, args.save_path)


if __name__ == "__main__":
    main()
