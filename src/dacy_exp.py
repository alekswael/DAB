import dacy

def ner_pipeline(text):

    nlp = dacy.load("da_dacy_medium_ner_fine_grained-0.1.0")
    doc = nlp(text)

    ner_results = []

    for ent in doc.ents:

        ner_result = {
            'entity_group': ent.label_,
            'word': ent.text,
            'start': ent.start_char,
            'end': ent.end_char
        }

        ner_results.append(ner_result)
    
    return ner_results

print(ner_pipeline("I udstillingen havde de Van Goghs Starry Night."))