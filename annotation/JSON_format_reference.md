# Data formatting reference

## Label Studio pre-annotated input JSON format
```py
data_list = [entry_dict1, entry_dict2, ...]  # insert entry_dict(s)

entry_dict = {  # entry for a document
    "data": data_dict,  # insert data_dict
    "predictions": [
        prediction_dict1,
        prediction_dict2,
        ...,
    ],  # insert prediction_dict(s)
}

data_dict = {
    "text": str,  # text of the document
    "source_dataset": str,  # reference dataset
    "file_name": str,  # file name of document
}

prediction_dict = {  # Contains the predictions (pre-annotations) to be displayed in the Label Studio UI.
    "model_version": str,  # Model for generating pre-annotations
    "result": [
        prediction_result_dict1,
        prediction_result_dict2,
        ...,
    ],  # insert prediction_result_dict(s)
}

prediction_result_dict = {  # contains a single marked entity
    "from_name": str,
    "to_name": str,
    "type": str,
    "value": {
        "start": int,  # start offset for entity
        "end": int,  # end offset for entity
        "text": str,  # entity
        "labels": ["LABEL"],  # insert labels, e.g. 'PER'
    },
}
```

## Label Studio annotated output JSON format
```py
data_list = [entry_dict1, entry_dict2, ...]  # insert entry_dict(s)

entry_dict = {
    "id": int,  # document ID
    "annotations": [  # list of annotations
        annotation_dict1,
        annotation_dict2,
        ...,
    ],  # insert annotation_dict(s)
    "file_upload": str,
    "drafts": [],
    "predictions": [],
    "data": data_dict,  # insert data_dict
    "meta": {},
    "created_at": str,
    "updated_at": str,
    "inner_id": int,
    "total_annotations": int,
    "cancelled_annotations": int,
    "total_predictions": int,
    "comment_count": int,
    "unresolved_comment_count": int,
    "last_comment_updated_at": null,
    "project": int,
    "updated_by": int,
    "comment_authors": [],
}

annotation_dict = {
    "id": int,
    "completed_by": int,  # annotator ID
    "result": [
        annotation_result_dict1,  # dict per entity
        annotation_result_dict2,
        ...,
    ],  # insert annotation_result_dict(s)
    "was_cancelled": bool,
    "ground_truth": bool,
    "created_at": str,
    "updated_at": str,
    "draft_created_at": str,
    "lead_time": int,
    "prediction": {
        prediction_dict1,
        prediction_dict2,
        ...,
    },  # insert prediction_dict(s)
    "result_count": int,
    "unique_id": str,
    "import_id": null,
    "last_action": null,
    "bulk_created": bool,
    "task": int,
    "project": int,
    "updated_by": int,
    "parent_prediction": int,
    "parent_annotation": null,
    "last_created_by": null,
}

annotation_result_dict = {
    "value": {
        "start": int,  # start offset for entity
        "end": int,  # end offset for entity
        "text": str,  # entity
        "labels": ["LABEL"],  # insert label, e.g. 'PER'
    },
    "id": int,
    "from_name": str,
    "to_name": str,
    "type": str,
    "origin": str,  # what action was made, either change in pre-annotation or manual annotation
    "entity_id": int,  # unique ID for this entity, BUT differs per annotator!
}

data_dict = {
    "text": str,  # text of the document
    "source_dataset": str,  # reference dataset
    "file_name": str,  # file name of document
}

prediction_dict = {
    "id": int,
    "result": [
        prediction_result_dict1,
        prediction_result_dict2,
        ...,
    ],  # insert prediction_result_dict(s)
    "model_version": str,  # model version, e.g. "DaCy"
    "created_ago": str,
    "score": null,
    "cluster": null,
    "neighbors": null,
    "mislabeling": float,
    "created_at": str,
    "updated_at": str,
    "model": null,
    "model_run": null,
    "task": int,
    "project": int,
}

prediction_result_dict = {  # contains a single marked entity
    "from_name": str,
    "to_name": str,
    "type": str,
    "value": {
        "start": int,  # start offset for entity
        "end": int,  # end offset for entity
        "text": str,  # entity
        "labels": ["LABEL"],  # insert label, e.g. 'PER'
    },
}
```

## Model prediction JSON reference

The model predictions should be saved as a JSON file with the following structure.

```py
[
    {
        "doc_id_1": [
            [start, end],   # entity 1 offsets
            [start, end],   # entity 2 offsets
            ...
        ],
        "doc_id_2": [
            [start, end],   # entity 1 offsets
            [start, end],   # entity 2 offsets
            ...
        ],
        ...
    },
    {
        "doc_id_1": "This is the document text for doc_id_1",
        "doc_id_2": "This is the document text for doc_id_2",
        ...
    }
]
```