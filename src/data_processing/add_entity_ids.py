# Import
import argparse
import json


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="This script is used for adding unique entity IDs to the annotations."
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
        help="The path for saving the pre-annotated JSON dataset. If None, overwrites the data_path JSON file.",
        required=False,
    )

    return parser.parse_args()


def load_data(data_path):
    with open(data_path, "r", encoding="utf-8") as doc:
        data_list = json.load(doc)

    return data_list


def add_entity_ids(data_list):

    # Loop through all entities per document and add to a set

    # Per doc
    for entry_dict in data_list:

        id_map_entry = +0  # ensure unique IDs per annotator

        # Per annotation / annotator
        for annotation_dict in entry_dict["annotations"]:

            entities = set()

            # Per entity
            for result_dict in annotation_dict["result"]:

                if result_dict["type"] == "labels":

                    entities.add(result_dict["value"]["text"])

            # Add unique IDs to the set
            id_map = {item: idx for idx, item in enumerate(entities)}

            # Per entity
            for result_dict in annotation_dict["result"]:

                if result_dict["type"] == "labels":

                    result_dict["entity_id"] = (
                        id_map[result_dict["value"]["text"]] + id_map_entry
                    )

            id_map_entry = +len(id_map)

    return data_list


def id_from_relation(data_list):

    for entry_dict in data_list:

        for annotation_dict in entry_dict["annotations"]:

            result_dicts = annotation_dict["result"]

            label_dicts = [d for d in result_dicts if d["type"] == "labels"]
            relation_dicts = [d for d in result_dicts if d["type"] == "relation"]

            # Build a lookup: label_id → entity_id
            label_id_to_entity_id = {d["id"]: d["entity_id"] for d in label_dicts}

            for relation_dict in relation_dicts:
                from_id = relation_dict["from_id"]
                to_id = relation_dict["to_id"]

                # Get the entity_id of the to_id label
                if to_id in label_id_to_entity_id:
                    entity_id_to_assign = label_id_to_entity_id[to_id]

                    # Update the from_id label with the to_id's entity_id
                    for label_dict in label_dicts:
                        if label_dict["id"] == from_id:
                            label_dict["entity_id"] = entity_id_to_assign
                            break  # Done updating this relation

            # Recombine result list
            annotation_dict["result"] = label_dicts + relation_dicts

    return data_list


def save_json(data_list, save_path):

    json_object = json.dumps(data_list, indent=2)

    with open(save_path, "w", encoding="utf-8") as outfile:
        outfile.write(json_object)


def main():
    args = parse_arguments()

    if args.save_path is None:
        args.save_path = args.data_path

    data_list = load_data(args.data_path)
    data_list = add_entity_ids(data_list)
    data_list = id_from_relation(data_list)
    save_json(data_list, args.save_path)


if __name__ == "__main__":
    main()
