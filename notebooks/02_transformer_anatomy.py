"""Lab 2: inspect transformer attention behaviour."""

import math
import torch
import torch.nn.functional as F

from bayan.attention import attention, MultiHeadAttention

from pathlib import Path

import matplotlib.pyplot as plt
from transformers import AutoModel, AutoTokenizer

def get_attention_weights(q, k, mask=None):
    """Return attention weights for inspection."""
    d_k = q.size(-1)

    scores = q @ k.transpose(-2, -1)
    scores = scores / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    return torch.softmax(scores, dim=-1)

def causal_mask(seq_len: int):
    return torch.tril(
        torch.ones(1, 1, seq_len, seq_len, dtype=torch.bool)
    )


def inspect_real_attention(text: str, label: str):
    checkpoint = "bert-base-multilingual-cased"

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModel.from_pretrained(
        checkpoint,
        attn_implementation="eager"
    )

    inputs = tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        outputs = model(
            **inputs,
            output_attentions=True
        )

    # Last transformer layer
    attentions = outputs.attentions[-1]

    # First batch item, first attention head
    head = attentions[0, 0].cpu()

    tokens = tokenizer.convert_ids_to_tokens(
        inputs["input_ids"][0]
    )

    print(f"\n=== Real attention: {label} ===")
    print("Tokens:", tokens)
    print("Attention matrix shape:", head.shape)

    # Save an annotated attention map
    Path("artifacts").mkdir(exist_ok=True)

    fig, ax = plt.subplots(
        figsize=(max(8, len(tokens) * 0.8),
                 max(6, len(tokens) * 0.7))
    )

    image = ax.imshow(head.numpy(), aspect="auto")

    ax.set_xticks(range(len(tokens)))
    ax.set_yticks(range(len(tokens)))

    ax.set_xticklabels(tokens, rotation=45, ha="right")
    ax.set_yticklabels(tokens)

    ax.set_xlabel("Key tokens")
    ax.set_ylabel("Query tokens")
    ax.set_title(f"mBERT attention — {label} — last layer, head 0")

    fig.colorbar(image, ax=ax, label="Attention weight")
    fig.tight_layout()

    output_path = f"artifacts/attention_{label}.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    print("Saved:", output_path)


def main():
    torch.manual_seed(42)

    # -------------------------------------------------
    # 1. Compare our attention with PyTorch
    # -------------------------------------------------
    q = torch.randn(1, 2, 4, 8)
    k = torch.randn(1, 2, 4, 8)
    v = torch.randn(1, 2, 4, 8)

    our_output = attention(q, k, v)
    pytorch_output = F.scaled_dot_product_attention(q, k, v)

    max_difference = (our_output - pytorch_output).abs().max().item()

    print("=== Numerical equivalence ===")
    print(f"Maximum difference: {max_difference:.8f}")
    print(
        "Equivalent:",
        torch.allclose(our_output, pytorch_output, atol=1e-6)
    )

    # -------------------------------------------------
    # 2. Inspect one attention head
    # -------------------------------------------------
    weights = get_attention_weights(q, k)

    print("\n=== Attention weight matrix: head 0 ===")
    print(weights[0, 0])
    print("Row sums:", weights[0, 0].sum(dim=-1))

    # -------------------------------------------------
    # 3. Exercise Multi-Head Attention
    # -------------------------------------------------
    x = torch.randn(2, 5, 32)

    mha = MultiHeadAttention(
        d_model=32,
        num_heads=4
    )

    mha_output = mha(x)

    print("\n=== Multi-Head Attention ===")
    print("Input shape :", x.shape)
    print("Output shape:", mha_output.shape)

    # -------------------------------------------------
    # 4. Verify ordinary masking
    # -------------------------------------------------
    mask = torch.ones(1, 1, 4, 4)
    mask[:, :, :, -1] = 0

    masked_weights = get_attention_weights(q, k, mask)

    print("\n=== Mask check ===")
    print(masked_weights[0, 0])
    print(
        "Attention paid to masked final position:",
        masked_weights[0, 0, :, -1]
    )

    # -------------------------------------------------
    # 5. Causal mask
    # -------------------------------------------------
    causal = causal_mask(4)

    causal_weights = get_attention_weights(q, k, causal)

    print("\n=== Causal mask ===")
    print(causal[0, 0].int())

    print("\nAttention weights with causal mask:")
    print(causal_weights[0, 0])

    # -------------------------------------------------
    # 6. Pad-attention leakage
    # -------------------------------------------------
    pad_mask = torch.tensor([[[[1, 1, 1, 0]]]], dtype=torch.bool)

    weights_without_pad_mask = get_attention_weights(q, k)
    weights_with_pad_mask = get_attention_weights(q, k, pad_mask)

    pad_mass_without_mask = weights_without_pad_mask[..., -1].mean().item()
    pad_mass_with_mask = weights_with_pad_mask[..., -1].mean().item()

    print("\n=== Pad-attention leakage ===")
    print(f"PAD attention without mask: {pad_mass_without_mask:.6f}")
    print(f"PAD attention with mask   : {pad_mass_with_mask:.6f}")

    assert pad_mass_with_mask < 1e-6

    # -------------------------------------------------
    # 7. Real Arabic / English attention
    # -------------------------------------------------
    inspect_real_attention(
        "الخدمة في الرياض ممتازة.",
        "arabic"
    )

    inspect_real_attention(
        "The service in Riyadh is excellent.",
        "english"
    )


if __name__ == "__main__":
    main()