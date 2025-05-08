# Imports
import json
import re
import abc
import argparse
import math
import numpy as np
from typing import Dict, List, Tuple
from tqdm import tqdm
import spacy
import intervaltree
from transformers import XLMRobertaForMaskedLM, AutoTokenizer
import torch


# Constants
# POS tags, tokens or characters that can be ignored from the recall scores
POS_TO_IGNORE = {"ADP", "PART", "CCONJ", "DET"}
TOKENS_TO_IGNORE = {"hr", "fru", "nr"}
CHARACTERS_TO_IGNORE = " ,.-;:/&()[]–'\" ’“”"


class TokenWeighting:
    """
    Abstract class for token weighting schemes used to compute precision.
    Subclasses must implement the `get_weights` method.
    """

    @abc.abstractmethod
    def get_weights(self, text: str, text_spans: List[Tuple[int, int]]):
        """Given a text and a list of text spans, returns a list of numeric weights
        (of same length as the list of spans) representing the information content
        conveyed by each span.

        A weight close to 0 represents a span with low information content (i.e. which
        can be easily predicted from the remaining context), while a weight close to 1
        represents a high information content (which is difficult to predict from the
        context)"""

        return


class UniformTokenWeighting(TokenWeighting):
    """
    Uniform weighting scheme where all tokens are assigned a weight of 1.0.
    """

    def get_weights(self, text: str, text_spans: List[Tuple[int, int]]):
        return [1.0] * len(text_spans)


class DanskBertTokenWeighting(TokenWeighting):
    """
    Token weighting based on the DanskBERT language model. Assigns weights
    based on the difficulty of predicting masked tokens using the model.
    """

    def __init__(self, max_segment_size=100):
        """
        Initializes the DanskBERT tokenizer and masked language model.

        Args:
            max_segment_size (int): Maximum segment size for processing long texts.
        """

        self.tokeniser = AutoTokenizer.from_pretrained("vesteinn/DanskBERT")
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = XLMRobertaForMaskedLM.from_pretrained("vesteinn/DanskBERT")
        self.model = self.model.to(self.device)
        self.max_segment_size = max_segment_size

    def get_weights(self, text: str, text_spans: List[Tuple[int, int]]):
        """
        Computes weights for text spans based on their predictability using DanskBERT.

        Args:
            text (str): The input text.
            text_spans (List[Tuple[int, int]]): List of character spans to evaluate.

        Returns:
            List[float]: Weights for each span, where higher weights indicate
            higher information content.
        """

        # STEP 1:  tokenise the text
        bert_tokens = self.tokeniser(text, return_offsets_mapping=True)
        input_ids = bert_tokens["input_ids"]
        input_ids_copy = np.array(input_ids)

        # STEP 2:  record the mapping between spans and BERT tokens
        bert_token_spans = bert_tokens["offset_mapping"]
        tokens_by_span = self._get_tokens_by_span(bert_token_spans, text_spans)

        # STEP 3:  mask the tokens that  wish to predict
        attention_mask = bert_tokens["attention_mask"]
        for token_indices in tokens_by_span.values():
            for token_idx in token_indices:
                attention_mask[token_idx] = 0
                input_ids[token_idx] = self.tokeniser.mask_token_id

        # STEP 4:  run the masked language model
        logits = self._get_model_predictions(input_ids, attention_mask)
        unnorm_probs = torch.exp(logits)
        probs = unnorm_probs / torch.sum(unnorm_probs, axis=1)[:, None]

        #  are only interested in the probs for the actual token values
        probs_actual = probs[torch.arange(len(input_ids)), input_ids_copy]
        probs_actual = probs_actual.detach().cpu().numpy()

        # STEP 5:  compute the weights from those predictions
        weights = []
        for span_start, span_end in text_spans:

            # If the span does not include any actual token, skip
            if not tokens_by_span[(span_start, span_end)]:
                weights.append(0)
                continue

            # if the span has several tokens,  take the minimum prob
            prob = np.min(
                [
                    probs_actual[token_idx]
                    for token_idx in tokens_by_span[(span_start, span_end)]
                ]
            )

            #  finally define the weight as -log(p)
            weights.append(-np.log(prob))

        return weights

    def _get_tokens_by_span(self, bert_token_spans, text_spans):
        """
        Maps text spans to their corresponding BERT token indices.

        Args:
            bert_token_spans (List[Tuple[int, int]]): Spans of BERT tokens.
            text_spans (List[Tuple[int, int]]): Spans of text to map.

        Returns:
            Dict[Tuple[int, int], List[int]]: Mapping of text spans to token indices.
        """

        #  create an interval tree to facilitate the mapping
        text_spans_tree = intervaltree.IntervalTree()
        for start, end in text_spans:
            text_spans_tree[start:end] = True

        #  create the actual mapping between spans and tokens
        tokens_by_span = {span: [] for span in text_spans}
        for token_idx, (start, end) in enumerate(bert_token_spans):
            for span_start, span_end, _ in text_spans_tree[start:end]:
                tokens_by_span[(span_start, span_end)].append(token_idx)

        # And control that everything is correct
        for span_start, span_end in text_spans:
            if len(tokens_by_span[(span_start, span_end)]) == 0:
                print(f"[WARNING]: span ({span_start},{span_end}) without any token")
        return tokens_by_span

    def _get_model_predictions(self, input_ids, attention_mask):
        """
        Runs the DanskBERT model on tokenized inputs and returns prediction scores.

        Args:
            input_ids (List[int]): Tokenized input IDs.
            attention_mask (List[int]): Attention mask for the input.

        Returns:
            torch.Tensor: Logits from the model for each token.
        """

        nb_tokens = len(input_ids)

        input_ids = torch.tensor(input_ids)[None, :].to(self.device)
        attention_mask = torch.tensor(attention_mask)[None, :].to(self.device)

        # If the number of tokens is too large,  split in segments
        if nb_tokens > self.max_segment_size:
            nb_segments = math.ceil(nb_tokens / self.max_segment_size)

            # Split the input_ids (and add padding if necessary)
            split_pos = [
                self.max_segment_size * (i + 1) for i in range(nb_segments - 1)
            ]
            input_ids_splits = torch.tensor_split(input_ids[0], split_pos)

            input_ids = torch.nn.utils.rnn.pad_sequence(
                input_ids_splits, batch_first=True
            )

            # Split the attention masks
            attention_mask_splits = torch.tensor_split(attention_mask[0], split_pos)
            attention_mask = torch.nn.utils.rnn.pad_sequence(
                attention_mask_splits, batch_first=True
            )

        # Run the model on the tokenised inputs + attention mask
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        # And get the resulting prediction scores
        scores = outputs.logits

        # If the batch contains several segments, concatenate the result
        if len(scores) > 1:
            scores = torch.vstack([scores[i] for i in range(len(scores))])
            scores = scores[:nb_tokens]
        else:
            scores = scores[0]

        return scores


class MaskedDocument:
    """
    Represents a document with masked text spans, identified by character boundaries.
    """

    def __init__(self, doc_id: str, masked_spans: List[Tuple[int, int]]):
        self.doc_id = doc_id
        self.masked_spans = masked_spans
        self.masked_offsets = None

    def get_masked_offsets(self):
        """
        Computes and returns the character offsets that are masked.

        Returns:
            set: Set of masked character offsets.
        """
        if self.masked_offsets is None:
            self.masked_offsets = {
                i for start, end in self.masked_spans for i in range(start, end)
            }
        return self.masked_offsets


class AnnotatedEntity:
    """
    Represents an annotated entity in a document, including its mentions,
    masking requirements, and type.
    """

    def __init__(
        self,
        entity_id: str,
        mentions: List[Tuple[int, int]],
        need_masking: bool,
        is_direct: bool,
        entity_type: str,
        mention_level_masking: List[bool],
    ):
        self.entity_id = entity_id
        self.mentions = mentions
        self.need_masking = need_masking
        self.is_direct = is_direct
        self.entity_type = entity_type
        self.mention_level_masking = mention_level_masking

        if self.is_direct and not self.need_masking:
            raise RuntimeError("Direct identifiers must always be masked")

    @property
    def mentions_to_mask(self):
        """
        Retrieves mentions that need to be masked.

        Returns:
            List[Tuple[int, int]]: List of mentions requiring masking.
        """
        return [
            mention
            for i, mention in enumerate(self.mentions)
            if self.mention_level_masking[i]
        ]

    def __repr__(self):
        attrs = ", ".join(f"{key}={value!r}" for key, value in self.__dict__.items())
        return f"{self.__class__.__name__}({attrs})"


class GoldCorpus:
    """
    Represents a gold standard corpus for text anonymization, loaded from a JSON file.
    Provides methods for evaluating masked spans against the gold standard.
    """

    def __init__(self, gold_standard_json_file: str, spacy_model="da_core_news_trf"):

        # Dict of GoldDocuments
        self.documents: Dict[str, GoldDocument] = {}

        # Loading the spacy model
        # This is later used to remove the POS-tags, tokens and characters to ignore
        nlp = spacy.load(spacy_model)

        # gold_standard_json_file is the annotated dataset
        fd = open(gold_standard_json_file, encoding="utf-8")
        data_list = json.load(fd)
        fd.close()
        print(f"[INFO]: Reading annotated corpus with {len(data_list)} documents")

        # Check format of data
        if type(data_list) != list:
            raise RuntimeError(
                "JSON file should be a list of annotated documents following the basic Label Studio JSON format."
            )

        for entry_dict in data_list:
            for key in ["id", "data", "annotations"]:
                if key not in entry_dict:
                    raise RuntimeError(
                        f"Annotated document is not well formed: missing variable {key}"
                    )

            for key in ["text", "source_dataset", "file_name"]:
                if key not in entry_dict["data"]:
                    raise RuntimeError(
                        f"Annotated document is not well formed: missing variable {key}"
                    )

            # Parsing the document with spacy
            spacy_doc = nlp(entry_dict["data"]["text"])

            # Creating the actual document (identifier, text and annotations)
            new_doc = GoldDocument(
                doc_id=int(entry_dict["id"]),
                text=entry_dict["data"]["text"],
                annotations=entry_dict["annotations"],
                spacy_doc=spacy_doc,
            )

            self.documents[entry_dict["id"]] = new_doc

            self.corpus_unique_annotators = set()

            for doc in self.documents.values():

                self.corpus_unique_annotators.update(doc.doc_unique_annotators)

    def __str__(self):
        attrs = ", ".join(f"{key}={value!r}" for key, value in self.__dict__.items())
        return f"{self.__class__.__name__}({attrs})"

    ### Functions for calculating eval metrics ###

    def get_entity_recall(
        self, masked_docs: List[MaskedDocument], include_direct=True, include_quasi=True
    ):
        """
        Computes entity-level recall for masked spans.

        Args:
            masked_docs (List[MaskedDocument]): Documents with masked spans.
            include_direct (bool): Include direct identifiers in the metric.
            include_quasi (bool): Include quasi identifiers in the metric.

        Returns:
            float: Entity-level recall.
        """

        nb_masked_entities = 0
        nb_entities = 0

        for doc in masked_docs:

            gold_doc = self.documents[doc.doc_id]

            entities_to_mask = gold_doc.get_entities_to_mask(
                include_direct, include_quasi
            )
            masked_entities = [
                entity for entity in entities_to_mask if gold_doc.is_masked(doc, entity)
            ]
            nb_masked_entities += len(masked_entities)
            nb_entities += len(entities_to_mask)
        try:
            return nb_masked_entities / nb_entities
        except ZeroDivisionError:
            return 0

    def get_recall(
        self,
        masked_docs: List[MaskedDocument],
        include_direct=True,
        include_quasi=True,
        token_level: bool = True,
    ):
        """
        Computes recall at the token or mention level.

        Args:
            masked_docs (List[MaskedDocument]): Documents with masked spans.
            include_direct (bool): Include direct identifiers in the metric.
            include_quasi (bool): Include quasi identifiers in the metric.
            token_level (bool): Compute recall at the token level if True, otherwise mention level.

        Returns:
            float: Recall value.
        """

        nb_masked_by_type, nb_by_type = self._get_mask_counts(
            masked_docs, include_direct, include_quasi, token_level
        )

        nb_masked_elements = sum(nb_masked_by_type.values())
        nb_elements = sum(nb_by_type.values())

        try:
            return nb_masked_elements / nb_elements
        except ZeroDivisionError:
            return 0

    def get_recall_per_entity_type(
        self,
        masked_docs: List[MaskedDocument],
        include_direct=True,
        include_quasi=True,
        token_level: bool = True,
    ):
        """
        Computes recall per entity type at the token or mention level.

        Args:
            masked_docs (List[MaskedDocument]): Documents with masked spans.
            include_direct (bool): Include direct identifiers in the metric.
            include_quasi (bool): Include quasi identifiers in the metric.
            token_level (bool): Compute recall at the token level if True, otherwise mention level.

        Returns:
            Dict[str, float]: Recall values per entity type.
        """

        nb_masked_by_type, nb_by_type = self._get_mask_counts(
            masked_docs, include_direct, include_quasi, token_level
        )

        return {
            ent_type: nb_masked_by_type[ent_type] / nb_by_type[ent_type]
            for ent_type in nb_by_type
        }

    def _get_mask_counts(
        self,
        masked_docs: List[MaskedDocument],
        include_direct=True,
        include_quasi=True,
        token_level: bool = True,
    ):

        nb_masked_elements_by_type = {}
        nb_elements_by_type = {}

        for doc in masked_docs:

            gold_doc = self.documents[doc.doc_id]
            for entity in gold_doc.get_entities_to_mask(include_direct, include_quasi):

                if entity.entity_type not in nb_elements_by_type:
                    nb_elements_by_type[entity.entity_type] = 0
                    nb_masked_elements_by_type[entity.entity_type] = 0

                spans = list(entity.mentions)
                if token_level:
                    spans = [
                        (start, end)
                        for mention_start, mention_end in spans
                        for start, end in gold_doc.split_by_tokens(
                            mention_start, mention_end
                        )
                    ]

                for start, end in spans:
                    if gold_doc.is_mention_masked(doc, start, end):
                        nb_masked_elements_by_type[entity.entity_type] += 1
                    nb_elements_by_type[entity.entity_type] += 1

        return nb_masked_elements_by_type, nb_elements_by_type

    def show_false_negatives(
        self,
        masked_docs: List[MaskedDocument],
        include_direct=True,
        include_quasi=True,
        include_partial_match=True,
        include_no_match=True,
    ):
        """
        Displays false negatives (mentions that should have been masked but weren't).

        Args:
            masked_docs (List[MaskedDocument]): Documents with masked spans.
            include_direct (bool): Include direct identifiers in the analysis.
            include_quasi (bool): Include quasi identifiers in the analysis.
            include_partial_match (bool): Include partially masked mentions.
            include_no_match (bool): Include mentions not masked at all.
        """

        if not include_partial_match and not include_no_match:
            raise RuntimeError("Must include some match to display")

        for doc in masked_docs:

            gold_doc = self.documents[doc.doc_id]
            masked_text_chars = list(gold_doc.text)
            for span_start, span_end in doc.masked_spans:
                masked_text_chars[span_start:span_end] = ["*"] * (span_end - span_start)
            masked_text = "".join(masked_text_chars)

            for entity in gold_doc.get_entities_to_mask(include_direct, include_quasi):

                for mention_start, mention_end in entity.mentions:
                    if not gold_doc.is_mention_masked(doc, mention_start, mention_end):

                        is_partial_match = "*" in masked_text[mention_start:mention_end]
                        if is_partial_match and not include_partial_match:
                            continue
                        elif not is_partial_match and not include_no_match:
                            continue

                        print(
                            f"Mention: {gold_doc.text[mention_start:mention_end]} "
                            f"(doc_id {gold_doc.doc_id}, span [{mention_start}-{mention_end}])"
                        )
                        context = masked_text[
                            max(0, mention_start - 30) : mention_end + 30
                        ]
                        context = re.sub(
                            r"\s\s+", " ", context.replace("\n", " "), re.DOTALL
                        )
                        print("Context:", context)
                        print("=============")

    def get_precision(
        self,
        masked_docs: List[MaskedDocument],
        token_weighting: TokenWeighting,
        token_level: bool = True,
    ):
        """
        Computes weighted precision for masked spans.

        Args:
            masked_docs (List[MaskedDocument]): Documents with masked spans.
            token_weighting (TokenWeighting): Token weighting scheme.
            token_level (bool): Compute precision at the token level if True, otherwise mention level.

        Returns:
            float: Precision value.
        """

        weighted_true_positives = 0.0
        weighted_system_masks = 0.0

        for doc in tqdm(masked_docs):
            gold_doc = self.documents[doc.doc_id]

            #  extract the list of spans (token- or mention-level)
            system_masks = []
            for start, end in doc.masked_spans:
                if token_level:
                    system_masks += list(gold_doc.split_by_tokens(start, end))
                else:
                    system_masks += [(start, end)]

            #  compute the weights (information content) of each mask
            weights = token_weighting.get_weights(gold_doc.text, system_masks)

            #  store the number of annotators in the gold standard document
            nb_annotators = len(
                set(entity.annotator for entity in gold_doc.entities.values())
            )

            for (start, end), weight in zip(system_masks, weights):

                #  extract the annotators that have also masked this token/span
                annotators = gold_doc.get_annotators_for_span(start, end)
                # print(f"[INFO]: Number of annotators in the corpus: {annotators}")

                # And update the (weighted) counts
                weighted_true_positives += len(annotators) * weight
                weighted_system_masks += nb_annotators * weight
        try:
            return weighted_true_positives / weighted_system_masks
        except ZeroDivisionError:
            return 0


class GoldDocument:
    """
    Represents an annotated document with text, annotations, and methods
    for evaluating masking against the gold standard.
    """

    def __init__(
        self,
        doc_id: str,
        text: str,
        annotations: Dict[str, List],
        spacy_doc: spacy.tokens.Doc,
    ):
        """Creates a new annotated document with an identifier, a text content, and
        a set of annotations (see guidelines)"""

        # The (unique) document identifier, its text and the spacy document
        self.doc_id = doc_id
        self.text = text
        self.spacy_doc = spacy_doc

        # Annotated entities (indexed by id)
        self.entities = {}

        self.doc_unique_annotators = set()

        for annotation_dict in annotations:

            if "result" not in annotation_dict:
                raise RuntimeError("Annotations must include result list")

            for entity in self._get_entities_from_mentions(annotation_dict["result"]):

                #  require each entity_id to be specific for each annotator
                if entity.entity_id in self.entities:
                    raise RuntimeError(
                        f"Entity ID {entity.entity_id} already used by another annotator"
                    )

                entity.annotator = annotation_dict["completed_by"]
                entity.doc_id = doc_id
                self.entities[entity.entity_id] = entity

                self.doc_unique_annotators.add(entity.annotator)

    def _get_entities_from_mentions(self, annotation_result_dicts):
        """Returns a set of entities based on the annotated mentions"""

        entities = {}

        for result_dict in annotation_result_dicts:

            if result_dict["type"] == "labels":  # Exclude the relation_dicts

                if result_dict["value"]["labels"][0] in [
                    "DIREKTE",
                    "KVASI",
                ]:  # Only include labels which are identifiers

                    for key in ["entity_id", "value"]:

                        if key not in result_dict:
                            raise RuntimeError(
                                "Unspecified key in entity mention: " + key
                            )

                    entity_id = result_dict["entity_id"]
                    start = result_dict["value"]["start"]
                    end = result_dict["value"]["end"]

                    if start < 0 or end > len(self.text) or start >= end:
                        raise RuntimeError(
                            f"Invalid character offsets: [{start}-{end}]"
                        )

                    if result_dict["value"]["labels"][0] not in ["DIREKTE", "KVASI"]:
                        raise RuntimeError(
                            f"Unspecified or invalid identifier type: {result_dict['identifier_type']}"
                        )

                    need_masking = result_dict["value"]["labels"][0] in [
                        "DIREKTE",
                        "KVASI",
                    ]
                    is_direct = result_dict["value"]["labels"][0] == "DIREKTE"

                    #  check whether the entity is already defined
                    if entity_id in entities:

                        # If yes,  simply add a new mention
                        current_entity = entities[entity_id]
                        current_entity.mentions.append((start, end))
                        current_entity.mention_level_masking.append(need_masking)

                    # Otherwise,  create a new entity with one single mention
                    else:
                        new_entity = AnnotatedEntity(
                            entity_id=entity_id,
                            mentions=[(start, end)],
                            need_masking=need_masking,
                            is_direct=is_direct,
                            entity_type=result_dict["value"]["labels"][0],
                            mention_level_masking=[need_masking],
                        )

                        entities[entity_id] = new_entity

        for entity in entities.values():
            if set(entity.mention_level_masking) != {entity.need_masking}:
                entity.need_masking = True
                print(
                    f"[WARNING]: Inconsistent masking of entity {entity.entity_id}: {entity.mention_level_masking}"
                )

        return list(entities.values())

    def is_masked(self, masked_doc: MaskedDocument, entity: AnnotatedEntity):
        """
        Checks if an entity is fully masked in the document.

        Args:
            masked_doc (MaskedDocument): Document with masked spans.
            entity (AnnotatedEntity): Entity to check.

        Returns:
            bool: True if the entity is fully masked, False otherwise.
        """

        for incr, (mention_start, mention_end) in enumerate(entity.mentions):

            if self.is_mention_masked(masked_doc, mention_start, mention_end):
                continue

            # The masking is sometimes inconsistent for the same entity,
            # so  verify that the mention does need masking
            elif entity.mention_level_masking[incr]:
                return False
        return True

    def is_mention_masked(
        self, masked_doc: MaskedDocument, mention_start: int, mention_end: int
    ):
        """
        Checks if a mention is fully masked in the document.

        Args:
            masked_doc (MaskedDocument): Document with masked spans.
            mention_start (int): Start character offset of the mention.
            mention_end (int): End character offset of the mention.

        Returns:
            bool: True if the mention is fully masked, False otherwise.
        """

        mention_to_mask = self.text[mention_start:mention_end].lower()

        # Computes the character offsets that must be masked
        offsets_to_mask = set(range(mention_start, mention_end))

        #  build the set of character offsets that are not covered
        non_covered_offsets = offsets_to_mask - masked_doc.get_masked_offsets()

        # If  have not covered everything,  also make sure punctuations
        # spaces, titles, etc. are ignored
        if len(non_covered_offsets) > 0:
            span = self.spacy_doc.char_span(
                mention_start, mention_end, alignment_mode="expand"
            )
            for token in span:
                if token.pos_ in POS_TO_IGNORE or token.lower_ in TOKENS_TO_IGNORE:
                    non_covered_offsets -= set(range(token.idx, token.idx + len(token)))
        for i in list(non_covered_offsets):
            if self.text[i] in set(CHARACTERS_TO_IGNORE):
                non_covered_offsets.remove(i)

        # If that set is empty,  consider the mention as properly masked
        return len(non_covered_offsets) == 0

    def get_entities_to_mask(self, include_direct=True, include_quasi=True):
        """
        Retrieves entities that should be masked based on the specified criteria.

        Args:
            include_direct (bool): Include direct identifiers.
            include_quasi (bool): Include quasi identifiers.

        Returns:
            List[AnnotatedEntity]: Entities to be masked.
        """

        to_mask = []
        for entity in self.entities.values():

            #  only consider entities that need masking and are the right type
            if not entity.need_masking:
                continue
            elif entity.is_direct and not include_direct:
                continue
            elif not entity.is_direct and not include_quasi:
                continue
            to_mask.append(entity)

        return to_mask

    def get_annotators_for_span(self, start_token: int, end_token: int):
        """
        Retrieves annotators who masked a specific text span.

        Args:
            start_token (int): Start character offset of the span.
            end_token (int): End character offset of the span.

        Returns:
            set: Set of annotator IDs.
        """

        #  compute an interval tree for fast retrieval
        if not hasattr(self, "masked_spans"):
            self.masked_spans = intervaltree.IntervalTree()
            for entity in self.entities.values():
                if entity.need_masking:
                    for i, (start, end) in enumerate(entity.mentions):
                        if entity.mention_level_masking[i]:
                            self.masked_spans[start:end] = entity.annotator

        annotators = set()
        for mention_start, mention_end, annotator in self.masked_spans[
            start_token:end_token
        ]:

            #  require that the span is fully covered by the annotator
            if mention_start <= start_token and mention_end >= end_token:
                annotators.add(annotator)

        return annotators

    def split_by_tokens(self, start: int, end: int):
        """
        Splits a text span into individual tokens.

        Args:
            start (int): Start character offset of the span.
            end (int): End character offset of the span.

        Yields:
            Tuple[int, int]: Start and end offsets of each token.
        """

        for match in re.finditer(r"\w+", self.text[start:end]):
            start_token = start + match.start(0)
            end_token = start + match.end(0)
            yield start_token, end_token

    def __repr__(self):
        attrs = ", ".join(f"{key}={value!r}" for key, value in self.__dict__.items())
        return f"{self.__class__.__name__}({attrs})"


def parse_arguments():
    """
    Parses command-line arguments for the benchmark script.

    Returns:
        argparse.Namespace: Parsed arguments including paths to input/output files,
        evaluation options, and verbosity settings.
    """

    parser = argparse.ArgumentParser(
        description="Computes evaluation metrics for the Danish Text Anonymisation benchmark."
    )

    parser.add_argument(
        "--gold_standard_file",
        type=str,
        default="./data/DAB_annotated_dataset.json",
        help="the path to the JSON file containing the gold standard annotations",
    )
    parser.add_argument(
        "--model_predictions_file",
        type=str,
        default="./output/predictions/DaAnonymization_predictions.json",
        help="the path to the file containing JSON files with actual spans masked by the system",
    )
    parser.add_argument(
        "--benchmark_output_file",
        type=str,
        default="./output/benchmarks/DaAnonymization_benchmark_result.txt",
        help="The directory to save the benchmark results in",
    )
    parser.add_argument(
        "--bert_weighting",
        action="store_true",
        default=True,
        help="Whether to calculate DanskBERT-weighted precision in addition to uniform.",
    )
    parser.add_argument(
        "--only_docs",
        default=None,
        nargs="*",
        help="list of document identifiers on which to focus the evaluation "
        + "(if not specified, computes the evaluation measures for all documents)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="provides detailed evaluation results (defaults to false)",
    )

    return parser.parse_args()


def get_masked_docs_from_file(masked_output_file: str):
    """
    Loads masked documents from a JSON file.

    Args:
        masked_output_file (str): Path to the JSON file containing masked spans.

    Returns:
        List[MaskedDocument]: List of MaskedDocument objects.
    """

    fd = open(masked_output_file)
    masked_output_docs = json.load(fd)
    fd.close()

    # Remove masked_texts
    masked_output_docs = masked_output_docs[0]

    if type(masked_output_docs) != dict:
        raise RuntimeError(
            f"{masked_output_file} must contain a mapping between document identifiers"
            + " and lists of masked spans in this document"
        )

    masked_docs = []

    # For each doc and and masket spans from the model
    for doc_id, masked_spans in masked_output_docs.items():
        doc = MaskedDocument(
            int(doc_id), []
        )  # Create a empty masked document class per doc
        if type(masked_spans) != list:
            raise RuntimeError(
                "Masked spans for the document must be a list of (start, end) tuples"
            )

        for start, end in masked_spans:
            doc.masked_spans.append(
                (start, end)
            )  # Append the masked spans to the MaskedDocument object
        masked_docs.append(doc)  # Append the doc to the docs

    return masked_docs


def main():

    args = parse_arguments()

    gold_corpus = GoldCorpus(args.gold_standard_file)

    masked_output_file = args.model_predictions_file

    masked_docs = get_masked_docs_from_file(masked_output_file)

    # Choose only a subset of docs (doc_ids) to evaluate the model on
    if args.only_docs:
        masked_docs = [doc for doc in masked_docs if doc.doc_id in args.only_docs]

    # Check whether the document is both in the annotations and the model predictions
    for masked_doc in masked_docs:

        if int(masked_doc.doc_id) not in gold_corpus.documents:
            raise RuntimeError(
                f"Document {masked_doc.doc_id} not present in gold corpus"
            )

    # Detailed eval results
    if args.verbose:
        gold_corpus.show_false_negatives(masked_docs, True, True)

    print(
        f"[INFO]: Computing evaluation metrics for {masked_output_file} ({len(masked_docs)} documents)"
    )
    print(
        f"[INFO]: Number of unique annotators in corpus: {len(gold_corpus.corpus_unique_annotators)}, annotator IDs: {gold_corpus.corpus_unique_annotators}"
    )

    token_recall = gold_corpus.get_recall(masked_docs, True, True, True)
    token_recall_by_type = gold_corpus.get_recall_per_entity_type(
        masked_docs, True, True, True
    )
    mention_recall = gold_corpus.get_recall(masked_docs, True, True, False)
    recall_direct_entities = gold_corpus.get_entity_recall(masked_docs, True, False)
    recall_quasi_entities = gold_corpus.get_entity_recall(masked_docs, False, True)
    token_precision = gold_corpus.get_precision(masked_docs, UniformTokenWeighting())
    mention_precision = gold_corpus.get_precision(
        masked_docs, UniformTokenWeighting(), False
    )

    output = f"""==> Token-level recall on all identifiers: {token_recall:.3f}
==> Token-level recall on all identifiers, factored by type:
    KVASI: {token_recall_by_type["KVASI"]:.3f}
    DIREKTE: {token_recall_by_type["DIREKTE"]:.3f}
==> Mention-level recall on all identifiers: {mention_recall:.3f}
==> Entity-level recall on direct identifiers: {recall_direct_entities:.3f}
==> Entity-level recall on quasi identifiers: {recall_quasi_entities:.3f}
==> Uniform token-level precision on all identifiers: {token_precision:.3f}
==> Uniform mention-level precision on all identifiers: {mention_precision:.3f}
"""

    if args.bert_weighting:

        weighted_token_precision = gold_corpus.get_precision(
            masked_docs, DanskBertTokenWeighting()
        )
        weighted_mention_precision = gold_corpus.get_precision(
            masked_docs, DanskBertTokenWeighting(), False
        )

        weighted_output = f"""==> DanskBERT-weighted, token-level precision on all identifiers: {weighted_token_precision:.3f}
==> DanskBERT-weighted, mention-level precision on all identifiers: {weighted_mention_precision:.3f}
"""
        output += weighted_output

    # Print benchmark result
    print(output)

    # Save benchmark result
    with open(
        args.benchmark_output_file,
        "w",
        encoding="utf-8",
    ) as f:
        f.write(output)


if __name__ == "__main__":
    main()
