"""Lab 2 starter: parameter accounting for mBERT and CAMeLBERT."""


"""Lab 2 starter: parameter accounting for mBERT and CAMeLBERT."""

from transformers import AutoModel


def audit(checkpoint: str) -> dict:
    model = AutoModel.from_pretrained(checkpoint)

    buckets = {
        "embeddings": 0,
        "attention": 0,
        "ffn": 0,
        "norms": 0,
        "pooler": 0,
        "other": 0,
    }

    for name, param in model.named_parameters():
        count = param.numel()

        if "LayerNorm" in name:
            buckets["norms"] += count

        elif "embeddings" in name:
            buckets["embeddings"] += count

        elif "attention" in name:
            buckets["attention"] += count

        elif "intermediate" in name or (
            "encoder.layer" in name and ".output.dense" in name
        ):
            buckets["ffn"] += count

        elif "pooler" in name:
            buckets["pooler"] += count

        else:
            buckets["other"] += count

    total = sum(buckets.values())

    return {
        "total": total,
        **buckets,
    }


if __name__ == "__main__":
    for ckpt in [
        "bert-base-multilingual-cased",
        "CAMeL-Lab/bert-base-arabic-camelbert-mix",
    ]:
        result = audit(ckpt)

        print(f"\n=== {ckpt} ===")
        print(f"Total parameters: {result['total']:,}")

        for bucket in [
            "embeddings",
            "attention",
            "ffn",
            "norms",
            "pooler",
            "other",
        ]:
            count = result[bucket]
            percent = (count / result["total"]) * 100

            print(
                f"{bucket:>10}: "
                f"{count:,} parameters "
                f"({percent:.2f}%)"
            )