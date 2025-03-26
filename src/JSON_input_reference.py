##### JSON STRUCTURE FOR INPUT INTO LABEL STUDIO #####

data_list = [] # insert entry_dict(s)

entry_dict = {
    "data": {}, # insert data_dict
    "predictions": [] # insert prediction_dict(s)
}

data_dict = {
    "text": "text",
    "source_dataset": "source_dataset"
}

predictions_dict = {
    "model_version": "model_version", # insert model version
    "result": [] # insert result_dict(s)
}

result_dict = {
    "from_name": "entity_mentions",
    "to_name": "doc_text",
    "type": "labels",
    "value": {
        "start": "start",
        "end": "end",
        "text": "word",
        "labels": [] # insert label
    }
}

ner_results = [
    {
        'entity_group': 'label',
        'score': score,
        'word': 'word_span',
        'start': 'start',
        'end': 'end'
    }
]