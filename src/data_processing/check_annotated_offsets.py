# Import
import argparse
import json


def parse_arguments():
    """
    Parse command-line arguments for the script.

    Returns:
        argparse.Namespace: Parsed arguments containing `data_path`.
    """
    parser = argparse.ArgumentParser(
        description="This script is used for checking the offsets of the annotated masks after exporting from Label Studio."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        help="The path to the annotated dataset in Label Studio JSON format.",
        default="./data/DAB_annotated_dataset.json",
        required=False,
    )

    return parser.parse_args()


def load_data(data_path):
    """
    Load the annotated dataset from a JSON file.

    Args:
        data_path (str): Path to the JSON file containing the annotated dataset.

    Returns:
        list: A list of dictionaries representing the annotated dataset.
    """
    with open(data_path, "r", encoding="utf-8") as doc:
        data_list = json.load(doc)

    return data_list


def get_masked_text(data_list):
    """
    Mask entities in the text based on their offsets and labels.

    Args:
        data_list (list): A list of dictionaries representing the annotated dataset.

    Returns:
        None
    """
    # Per doc
    for entry_dict in data_list:

        # Per annotation / annotator
        for annotation_dict in entry_dict["annotations"]:

            text = entry_dict["data"]["text"]
            masked_entities = []
            offsets = []

            # Per entity
            for result_dict in annotation_dict["result"]:

                if result_dict["type"] == "labels":

                    if result_dict["value"]["labels"][0] in ["DIREKTE", "KVASI"]:

                        masked_entity = result_dict["value"]["text"]
                        offset = (
                            result_dict["value"]["start"],
                            result_dict["value"]["end"],
                        )

                        masked_entities.append(masked_entity)
                        offsets.append(offset)

            offsets = sorted(offsets, key=lambda x: x[0], reverse=True)

            masked_text = text

            for start, end in offsets:
                if 0 <= start < end <= len(masked_text):
                    mask = "*" * (end - start)
                    masked_text = masked_text[:start] + mask + masked_text[end:]
                else:
                    raise ValueError(
                        f"Invalid span: ({start}, {end}) for text of length {len(masked_text)}"
                    )

            print(
                f"Document ID: {entry_dict["id"]}\nOriginal text:\n{text}\nMasked text:\n{masked_text}"
            )

    return None


def main():
    """
    Main function to parse arguments, load data, and process masked text.
    """
    args = parse_arguments()
    data_list = load_data(args.data_path)
    get_masked_text(data_list)


if __name__ == "__main__":
    main()
