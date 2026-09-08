"""Lab 1 starter: sentence segmentation."""

import spacy

from bayan.preprocessing.core import preprocess

def build_pipeline():
    # TODO(Lab 1): build the spaCy segmentation pipeline.
    nlp = spacy.blank("xx")
    nlp.add_pipe("sentencizer")
    return nlp


def split_sentences(raw: str, nlp) -> list[str]:
    # TODO(Lab 1): preprocess then return non-empty sentence strings.
    clean_text = preprocess(raw)
    doc = nlp(clean_text)

    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]
