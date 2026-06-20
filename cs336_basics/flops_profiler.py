# flops_profiler.py

from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict
from typing import Any


@dataclass
class MatmulRecord:
    name: str
    description: str
    left_shape: str
    right_shape: str
    output_shape: str
    flops: int


@dataclass
class GPTMatmulProfileConfig:
    batch_size: int
    context_length: int
    num_layers: int
    d_model: int
    num_heads: int
    d_ff: int
    vocab_size: int


def dense_matmul_flops(batch: int, m: int, n: int, p: int) -> int:
    """
    Count FLOPs for batched dense matmul:

        batch × ([m, n] @ [n, p]) -> batch × [m, p]

    We count one multiply and one add as 2 FLOPs, so:

        FLOPs = 2 * batch * m * n * p
    """
    return 2 * batch * m * n * p


def profile_gpt_matmul_flops(
    *,
    batch_size: int,
    context_length: int,
    num_layers: int,
    d_model: int,
    num_heads: int,
    d_ff: int,
    vocab_size: int,
) -> tuple[list[MatmulRecord], int]:
    """
    Statically profile matrix multiplication FLOPs for a GPT-style Transformer LM.

    This matches the model structure:

        - per layer:
            1. q_proj
            2. k_proj
            3. v_proj
            4. QK^T attention score matmul
            5. Attn @ V matmul
            6. output_proj
            7. FFN w1
            8. FFN w3
            9. FFN w2

        - final:
            10. lm_head

    This function does not count:
        - embedding lookup
        - RMSNorm
        - RoPE
        - softmax
        - masking
        - SiLU
        - elementwise multiply/add
    """
    if d_model % num_heads != 0:
        raise ValueError(
            f"d_model must be divisible by num_heads, got "
            f"d_model={d_model}, num_heads={num_heads}"
        )

    B = batch_size
    C = context_length
    L = num_layers
    d = d_model
    h = num_heads
    d_k = d_model // num_heads
    d_v = d_model // num_heads
    V = vocab_size

    records: list[MatmulRecord] = []

    def add_record(
        *,
        name: str,
        description: str,
        batch: int,
        m: int,
        n: int,
        p: int,
        left_shape: str,
        right_shape: str,
        output_shape: str,
    ) -> None:
        flops = dense_matmul_flops(batch=batch, m=m, n=n, p=p)
        records.append(
            MatmulRecord(
                name=name,
                description=description,
                left_shape=left_shape,
                right_shape=right_shape,
                output_shape=output_shape,
                flops=flops,
            )
        )

    for layer_idx in range(L):
        prefix = f"layers.{layer_idx}"

        # 1. Q projection
        # [B, C, d] @ [d, d] -> [B, C, d]
        add_record(
            name=f"{prefix}.attn.q_proj",
            description="Project hidden states to queries",
            batch=B,
            m=C,
            n=d,
            p=d,
            left_shape=f"[B={B}, C={C}, d_model={d}]",
            right_shape=f"[d_model={d}, d_model={d}]",
            output_shape=f"[B={B}, C={C}, d_model={d}]",
        )

        # 2. K projection
        add_record(
            name=f"{prefix}.attn.k_proj",
            description="Project hidden states to keys",
            batch=B,
            m=C,
            n=d,
            p=d,
            left_shape=f"[B={B}, C={C}, d_model={d}]",
            right_shape=f"[d_model={d}, d_model={d}]",
            output_shape=f"[B={B}, C={C}, d_model={d}]",
        )

        # 3. V projection
        add_record(
            name=f"{prefix}.attn.v_proj",
            description="Project hidden states to values",
            batch=B,
            m=C,
            n=d,
            p=d,
            left_shape=f"[B={B}, C={C}, d_model={d}]",
            right_shape=f"[d_model={d}, d_model={d}]",
            output_shape=f"[B={B}, C={C}, d_model={d}]",
        )

        # 4. Attention scores: QK^T
        # per head: [C, d_k] @ [d_k, C] -> [C, C]
        # total batch = B * h
        add_record(
            name=f"{prefix}.attn.qk_scores",
            description="Compute attention scores QK^T",
            batch=B * h,
            m=C,
            n=d_k,
            p=C,
            left_shape=f"[B={B}, h={h}, C={C}, d_k={d_k}]",
            right_shape=f"[B={B}, h={h}, d_k={d_k}, C={C}]",
            output_shape=f"[B={B}, h={h}, C={C}, C={C}]",
        )

        # 5. Attention output: Attn @ V
        # per head: [C, C] @ [C, d_v] -> [C, d_v]
        # total batch = B * h
        add_record(
            name=f"{prefix}.attn.attn_v",
            description="Multiply attention probabilities by values",
            batch=B * h,
            m=C,
            n=C,
            p=d_v,
            left_shape=f"[B={B}, h={h}, C={C}, C={C}]",
            right_shape=f"[B={B}, h={h}, C={C}, d_v={d_v}]",
            output_shape=f"[B={B}, h={h}, C={C}, d_v={d_v}]",
        )

        # 6. Output projection
        # [B, C, d] @ [d, d] -> [B, C, d]
        add_record(
            name=f"{prefix}.attn.output_proj",
            description="Project concatenated attention heads back to d_model",
            batch=B,
            m=C,
            n=d,
            p=d,
            left_shape=f"[B={B}, C={C}, d_model={d}]",
            right_shape=f"[d_model={d}, d_model={d}]",
            output_shape=f"[B={B}, C={C}, d_model={d}]",
        )

        # 7. FFN w1: gate projection
        # [B, C, d] @ [d, d_ff] -> [B, C, d_ff]
        add_record(
            name=f"{prefix}.ffn.w1",
            description="SwiGLU gate projection",
            batch=B,
            m=C,
            n=d,
            p=d_ff,
            left_shape=f"[B={B}, C={C}, d_model={d}]",
            right_shape=f"[d_model={d}, d_ff={d_ff}]",
            output_shape=f"[B={B}, C={C}, d_ff={d_ff}]",
        )

        # 8. FFN w3: value projection
        # [B, C, d] @ [d, d_ff] -> [B, C, d_ff]
        add_record(
            name=f"{prefix}.ffn.w3",
            description="SwiGLU value projection",
            batch=B,
            m=C,
            n=d,
            p=d_ff,
            left_shape=f"[B={B}, C={C}, d_model={d}]",
            right_shape=f"[d_model={d}, d_ff={d_ff}]",
            output_shape=f"[B={B}, C={C}, d_ff={d_ff}]",
        )

        # 9. FFN w2: down projection
        # [B, C, d_ff] @ [d_ff, d] -> [B, C, d]
        add_record(
            name=f"{prefix}.ffn.w2",
            description="SwiGLU down projection",
            batch=B,
            m=C,
            n=d_ff,
            p=d,
            left_shape=f"[B={B}, C={C}, d_ff={d_ff}]",
            right_shape=f"[d_ff={d_ff}, d_model={d}]",
            output_shape=f"[B={B}, C={C}, d_model={d}]",
        )

    # 10. Final LM head
    # [B, C, d] @ [d, vocab_size] -> [B, C, vocab_size]
    add_record(
        name="lm_head",
        description="Project final hidden states to vocabulary logits",
        batch=B,
        m=C,
        n=d,
        p=V,
        left_shape=f"[B={B}, C={C}, d_model={d}]",
        right_shape=f"[d_model={d}, vocab_size={V}]",
        output_shape=f"[B={B}, C={C}, vocab_size={V}]",
    )

    total_flops = sum(record.flops for record in records)
    return records, total_flops


def profile_model_matmul_flops(
    model: Any,
    *,
    batch_size: int = 1,
    context_length: int | None = None,
) -> tuple[list[MatmulRecord], int]:
    """
    Profile your BasicsTransformerLM object directly.

    The model is expected to have these attributes:
        model.context_length
        model.num_layers
        model.d_model
        model.num_heads
        model.d_ff
        model.vocab_size
    """
    C = context_length if context_length is not None else model.context_length

    return profile_gpt_matmul_flops(
        batch_size=batch_size,
        context_length=C,
        num_layers=model.num_layers,
        d_model=model.d_model,
        num_heads=model.num_heads,
        d_ff=model.d_ff,
        vocab_size=model.vocab_size,
    )


def summarize_by_op(records: list[MatmulRecord]) -> dict[str, int]:
    """
    Aggregate FLOPs by operation type.

    Example:
        layers.0.attn.q_proj
        layers.1.attn.q_proj
        ...
    will be aggregated into:
        attn.q_proj
    """
    summary: dict[str, int] = defaultdict(int)

    for record in records:
        parts = record.name.split(".")

        if len(parts) >= 3 and parts[0] == "layers":
            # layers.0.attn.q_proj -> attn.q_proj
            key = ".".join(parts[2:])
        else:
            key = record.name

        summary[key] += record.flops

    return dict(summary)


def format_flops(flops: int) -> str:
    if flops >= 1e12:
        return f"{flops / 1e12:.3f} TFLOPs"
    if flops >= 1e9:
        return f"{flops / 1e9:.3f} GFLOPs"
    if flops >= 1e6:
        return f"{flops / 1e6:.3f} MFLOPs"
    return f"{flops} FLOPs"


def print_summary(records: list[MatmulRecord], total_flops: int) -> None:
    summary = summarize_by_op(records)

    print("=== FLOPs by operation ===")
    for name, flops in summary.items():
        print(f"{name:<24} {flops:>20,}  {format_flops(flops)}")

    print()
    print("=== Total ===")
    print(f"Total FLOPs: {total_flops:,}")
    print(f"Total:       {format_flops(total_flops)}")


def print_detailed_records(
    records: list[MatmulRecord],
    *,
    max_rows: int | None = 20,
) -> None:
    """
    Print detailed matmul records.

    By default only prints first 20 rows because GPT-2 XL has:
        48 layers * 9 matmuls + 1 lm_head = 433 records

    Set max_rows=None to print all records.
    """
    rows = records if max_rows is None else records[:max_rows]

    print("=== Detailed matmul records ===")
    for record in rows:
        print(f"name:        {record.name}")
        print(f"description: {record.description}")
        print(f"left:        {record.left_shape}")
        print(f"right:       {record.right_shape}")
        print(f"output:      {record.output_shape}")
        print(f"flops:       {record.flops:,} ({format_flops(record.flops)})")
        print("-" * 80)

    if max_rows is not None and len(records) > max_rows:
        print(f"... omitted {len(records) - max_rows} records")


if __name__ == "__main__":
    """
    1. q_proj:       [B, C, d] × [d, d]
    2. k_proj:       [B, C, d] × [d, d]
    3. v_proj:       [B, C, d] × [d, d]
    4. QK^T:         [B, h, C, d_k] × [B, h, d_k, C]
    5. Attn @ V:     [B, h, C, C] × [B, h, C, d_v]
    6. output_proj:  [B, C, d] × [d, d]
    7. ffn.w1:       [B, C, d] × [d, d_ff]
    8. ffn.w3:       [B, C, d] × [d, d_ff]
    9. ffn.w2:       [B, C, d_ff] × [d_ff, d]
    """
    
    # GPT-2 XL-shaped config from your current implementation
    records, total_flops = profile_gpt_matmul_flops(
        batch_size=1,
        context_length=1024,
        num_layers=48,
        d_model=1600,
        num_heads=25,
        d_ff=4288,
        vocab_size=50257,
    )

    print_summary(records, total_flops)
    print()
    print_detailed_records(records, max_rows=12)

    