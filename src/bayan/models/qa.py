"""Lab 3 starter: extractive QA post-processing."""


def best_span(
    start_logits,
    end_logits,
    offsets,
    *,
    null_score,
    null_threshold,
    max_answer_len=30,
    top_k=20,
):
    # Keep only tokens that belong to the context.
    valid_indices = [
        i for i, offset in enumerate(offsets)
        if offset is not None
    ]

    # Get the strongest start candidates.
    start_candidates = sorted(
        valid_indices,
        key=lambda i: float(start_logits[i]),
        reverse=True,
    )[:top_k]

    # Get the strongest end candidates.
    end_candidates = sorted(
        valid_indices,
        key=lambda i: float(end_logits[i]),
        reverse=True,
    )[:top_k]

    best = None

    # Search for the highest-scoring VALID span.
    for start_index in start_candidates:
        for end_index in end_candidates:

            # The answer cannot end before it starts.
            if end_index < start_index:
                continue

            # Do not allow extremely long answers.
            if end_index - start_index + 1 > max_answer_len:
                continue

            score = (
                float(start_logits[start_index])
                + float(end_logits[end_index])
            )

            if best is None or score > best["score"]:
                best = {
                    "answer": (
                        offsets[start_index][0],
                        offsets[end_index][1],
                    ),
                    "start_token": start_index,
                    "end_token": end_index,
                    "score": score,
                    "null_score": float(null_score),
                }

    # If there is no valid span, return no answer.
    if best is None:
        return {
            "answer": None,
            "score": float("-inf"),
            "null_score": float(null_score),
        }

    # If the null answer is sufficiently stronger,
    # return no answer instead.
    if float(null_score) - best["score"] > float(null_threshold):
        return {
            "answer": None,
            "score": best["score"],
            "null_score": float(null_score),
        }

    return best
