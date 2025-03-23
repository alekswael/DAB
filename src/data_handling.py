# Import
import pandas as pd
import json
import os

data_folder = '/home/aleksander/projects/DAB/data/'

data_list = []

def load_texts_from_subfolders(data_folder):

    ##### JSON STRUCTURE #####

    entry_dict = {
        "data": "data_dict",
        "predictions": "predictions_list"
    }

    data_dict = {
        "text": "text",
        "source_dataset": "source_dataset"
    }

    predictions_list = [
        {
            "model_version": "SkandiNER",
            "result": [] # insert result_dict(s)
        }
    ]

    result_dict = {
        "from_name": "entity_mentions",
        "to_name": "doc_text",
        "type": "labels",
        "value": {
            "start": "start",
            "end": "end",
            "text": "word",
            "labels": [] #insert label
        }
    }

    # dir_tree = os.walk(data_folder)

    # _, dirs, _ = next(dir_tree) # Get subfolders in data dir

    # print(dirs)

    dirs = [f.name for f in os.scandir(data_folder) if f.is_dir()]
    print(dirs)

    for dir in dirs:

        dir_path = os.path.join(data_folder, dir)
        files = [f.name for f in os.scandir(dir_path) if f.is_file()]
        print(files)

        for file in files:

            if file.endswith(".txt"):

                file_path = os.path.join(dir_path, file)

                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()

                    data_dict = {
                        "text": "text",
                        "source_dataset": "source_dataset"
                    }
                    
                    data_dict["text"] = text
                    data_dict["source_dataset"] = dir
                    print(data_dict)

                    predictions_list

    return data_dict

load_texts_from_subfolders(data_folder)

'''
#################### PREDICTIONS ####################

predictions = load_texts_from_subfolders(data_folder)

print(texts_dict)

# Print the dictionary to verify
# for filename, text in texts_dict.items():
#     print(f"Filename: {filename}\nText: {text}\n")

def load_texts_from_subfolders(data_folder):

    ##### JSON STRUCTURE #####

    entry_dict = {
        "data": "data_dict",
        "predictions": "predictions_list"
    }

    data_dict = {
        "data": {
            "text": "text",
            "source_dataset": "source_dataset"
        },
    }

    predictions_list = [
        {
            "model_version": "SkandiNER",
            "result": [] # insert result_dict(s)
        }
    ]

    result_dict = {
        "from_name": "entity_mentions",
        "to_name": "doc_text",
        "type": "labels",
        "value": {
            "start": "start",
            "end": "end",
            "text": "word",
            "labels": [] #insert label
        }
    }

    for root, dirs, files in os.walk(data_folder):

        for file in files:

            if file.endswith(".txt"):

                file_path = os.path.join(root, file)

                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                    data_dict[] = text

    return texts_dict


texts_dict = load_texts_from_subfolders(data_folder)

print(texts_dict)

# Print the dictionary to verify
# for filename, text in texts_dict.items():
#     print(f"Filename: {filename}\nText: {text}\n")

'''