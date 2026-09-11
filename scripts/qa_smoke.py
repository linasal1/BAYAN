"""Lab 3B: run the 12-question QA smoke set."""

import json
from pathlib import Path

import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

from bayan.models.qa import best_span


MODEL_NAME = "deepset/roberta-base-squad2"

SMOKE_PATH = Path("data/eval/qa_smoke_set.json")
FULL_QA_PATH = Path("data/models/bayan_qa.json")

NULL_THRESHOLD = 0.0


def collect_examples(payload):
    examples = []

    for service in payload["data"]:
        for paragraph in service["paragraphs"]:

            context = paragraph["context"]

            for qa in paragraph["qas"]:
                examples.append(
                    {
                        "id": qa["id"],
                        "question": qa["question"],
                        "context": context,
                        "answers": qa["answers"],
                        "is_impossible": qa["is_impossible"],
                    }
                )

    return examples


def normalize(text):
    if text is None:
        return None

    return " ".join(text.lower().strip().split())


def predict(model, tokenizer, device, question, context):

    encoded = tokenizer(
        question,
        context,
        return_offsets_mapping=True,
        return_tensors="pt",
        truncation="only_second",
        max_length=384,
    )

    sequence_ids = encoded.sequence_ids(0)

    raw_offsets = encoded.pop(
        "offset_mapping"
    )[0].tolist()

    offsets = [
        tuple(offset) if sequence_ids[i] == 1 else None
        for i, offset in enumerate(raw_offsets)
    ]

    input_ids = encoded["input_ids"][0].tolist()

    cls_index = input_ids.index(
        tokenizer.cls_token_id
    )

    model_inputs = {
        key: value.to(device)
        for key, value in encoded.items()
    }

    with torch.no_grad():
        outputs = model(**model_inputs)

    start_logits = (
        outputs.start_logits[0]
        .detach()
        .cpu()
        .numpy()
    )

    end_logits = (
        outputs.end_logits[0]
        .detach()
        .cpu()
        .numpy()
    )

    null_score = (
        float(start_logits[cls_index])
        + float(end_logits[cls_index])
    )

    result = best_span(
        start_logits,
        end_logits,
        offsets,
        null_score=null_score,
        null_threshold=NULL_THRESHOLD,
    )

    if result["answer"] is None:
        return None

    start_char, end_char = result["answer"]

    return context[start_char:end_char]


def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)
    print("Model:", MODEL_NAME)
    print()

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=True,
    )

    model = (
        AutoModelForQuestionAnswering
        .from_pretrained(MODEL_NAME)
        .to(device)
    )

    model.eval()

    smoke_payload = json.loads(
        SMOKE_PATH.read_text(encoding="utf-8")
    )

    smoke_examples = collect_examples(
        smoke_payload
    )

    answerable = [
        example
        for example in smoke_examples
        if not example["is_impossible"]
    ][:9]

    full_payload = json.loads(
        FULL_QA_PATH.read_text(encoding="utf-8")
    )

    full_examples = collect_examples(
        full_payload
    )

    unanswerable = [
        example
        for example in full_examples
        if example["is_impossible"]
    ][:3]

    print("Answerable examples:", len(answerable))
    print("Unanswerable examples:", len(unanswerable))
    print()

    answerable_correct = 0
    null_correct = 0

    print("===== ANSWERABLE =====")

    for example in answerable:

        predicted = predict(
            model,
            tokenizer,
            device,
            example["question"],
            example["context"],
        )

        expected = example["answers"][0]["text"]

        passed = (
            normalize(predicted)
            == normalize(expected)
        )

        if passed:
            answerable_correct += 1

        print(
            f"{'PASS' if passed else 'FAIL'} | "
            f"{example['id']} | "
            f"pred={predicted!r} | "
            f"expected={expected!r}"
        )

    print()
    print("===== UNANSWERABLE =====")

    for example in unanswerable:

        predicted = predict(
            model,
            tokenizer,
            device,
            example["question"],
            example["context"],
        )

        passed = predicted is None

        if passed:
            null_correct += 1

        print(
            f"{'PASS' if passed else 'FAIL'} | "
            f"{example['id']} | "
            f"pred={predicted!r} | "
            f"expected=None"
        )

    print()
    print("==============================")
    print("LAB 3 QA SMOKE RESULT")
    print("==============================")

    print(
        f"Answerable:   {answerable_correct}/9"
    )

    print(
        f"Unanswerable: {null_correct}/3"
    )

    if answerable_correct == 9 and null_correct == 3:
        print()
        print("QA SMOKE TEST: PASS")
    else:
        print()
        print("QA SMOKE TEST: TARGET NOT MET")


if __name__ == "__main__":
    main()
