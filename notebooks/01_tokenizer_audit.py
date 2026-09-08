"""Lab 1 starter: audit four tokenizer candidates on Bayan AR/EN text."""
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

from bayan.preprocessing.core import preprocess



CANDIDATES = {
    "bert-base-multilingual-cased": "mBERT",
    "xlm-roberta-base": "XLM-R",
    "CAMeL-Lab/bert-base-arabic-camelbert-mix": "CAMeLBERT",
    "distilbert-base-uncased": "DistilBERT",
}

DATA = Path("data/raw/bayan_feedback.csv")


def fertility(tokenizer, texts) -> float:
    total_pieces = 0
    total_words = 0

    for text in texts:
        total_pieces += len(tokenizer.tokenize(text))
        total_words += len(text.split())

    return total_pieces / total_words

def sequence_lengths(tokenizer, texts):
    lengths = []

    for text in texts:
        tokens = tokenizer(text, add_special_tokens=True)["input_ids"]
        lengths.append(len(tokens))

    return lengths    

def main():
    df = pd.read_csv(DATA)

    df["clean_text"] = df["text"].astype(str).apply(preprocess)

    ar_texts = df[df["lang"] == "ar"]["clean_text"].tolist()
    en_texts = df[df["lang"] == "en"]["clean_text"].tolist()

    print(f"Arabic messages: {len(ar_texts)}")
    print(f"English messages: {len(en_texts)}")

    results = []

    for checkpoint, name in CANDIDATES.items():
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)

        ar_fertility = fertility(tokenizer, ar_texts)
        en_fertility = fertility(tokenizer, en_texts)
        ar_lengths = sequence_lengths(tokenizer, ar_texts)
        en_lengths = sequence_lengths(tokenizer, en_texts)

        ar_p95 = pd.Series(ar_lengths).quantile(0.95)
        en_p95 = pd.Series(en_lengths).quantile(0.95)

        print(f"{name} Arabic fertility: {ar_fertility:.2f}")
        print(f"{name} English fertility: {en_fertility:.2f}")
        print(f"{name} Arabic 95th percentile length: {ar_p95:.0f}")
        print(f"{name} English 95th percentile length: {en_p95:.0f}")

if __name__ == "__main__":
    main()
