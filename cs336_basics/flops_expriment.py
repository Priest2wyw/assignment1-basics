"""
Experiment runner for GPT-style Transformer LM matrix-multiplication FLOPs.

This file intentionally keeps the original static profiler untouched. It adds
an experiment layer around `profile_gpt_matmul_flops`:

    build_experiment_cases -> run_experiments -> plot_results -> make_report_tables

Two experiments are included:
1. model_scale: GPT-2 small / medium / large / XL at context_length = 1024
2. context_sweep: GPT-2 XL with context_length from 1024 to 16384

Counting scope:
- Matrix multiplication FLOPs only.
- One multiply + one add = 2 FLOPs.
- Non-matmul operations such as embedding lookup, RMSNorm, RoPE, softmax,
  masking, SiLU, and elementwise ops are excluded, following flops_profiler.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
from plotnine import (
    aes,
    element_text,
    facet_wrap,
    geom_area,
    geom_col,
    geom_line,
    geom_point,
    ggplot,
    labs,
    scale_x_continuous,
    scale_y_continuous,
    theme,
    theme_bw,
)
from mizani.formatters import percent_format

from flops_profiler import (
    format_flops,
    profile_gpt_matmul_flops,
    summarize_by_op,
)


# Keep component definitions explicit. This is the central bridge between the
# original profiler's op-level records and the assignment's model-component view.
COMPONENT_MAP: dict[str, str] = {
    "attn.q_proj": "Attention projection",
    "attn.k_proj": "Attention projection",
    "attn.v_proj": "Attention projection",
    "attn.output_proj": "Attention projection",
    "attn.qk_scores": "Attention quadratic",
    "attn.attn_v": "Attention quadratic",
    "ffn.w1": "FFN / SwiGLU",
    "ffn.w3": "FFN / SwiGLU",
    "ffn.w2": "FFN / SwiGLU",
    "lm_head": "LM head",
}

COMPONENT_ORDER = [
    "Attention projection",
    "Attention quadratic",
    "FFN / SwiGLU",
    "LM head",
]

MODEL_ORDER = ["GPT-2 small", "GPT-2 medium", "GPT-2 large", "GPT-2 XL"]


# The d_ff values follow the current SwiGLU-style implementation used by the
# provided GPT-2 XL config, rather than the original GPT-2 4*d_model MLP.
GPT2_SWIGLU_CONFIGS: dict[str, dict[str, int]] = {
    "GPT-2 small": {"num_layers": 12, "d_model": 768, "num_heads": 12, "d_ff": 2048},
    "GPT-2 medium": {"num_layers": 24, "d_model": 1024, "num_heads": 16, "d_ff": 2752},
    "GPT-2 large": {"num_layers": 36, "d_model": 1280, "num_heads": 20, "d_ff": 3456},
    "GPT-2 XL": {"num_layers": 48, "d_model": 1600, "num_heads": 25, "d_ff": 4288},
}


def build_experiment_cases(
    *,
    batch_size: int = 1,
    base_context_length: int = 1024,
    max_context_length: int = 16_384,
    context_step: int = 1024,
    vocab_size: int = 50_257,
) -> pd.DataFrame:
    """
    Define all experiment cases.

    Returns one tidy case table. Each row is a single profiler run.
    """
    cases: list[dict[str, int | str]] = []

    # Experiment A: vary model scale at fixed context length.
    for model_idx, model_name in enumerate(MODEL_ORDER):
        cfg = GPT2_SWIGLU_CONFIGS[model_name]
        cases.append(
            {
                "experiment": "model_scale",
                "case_name": model_name,
                "model_name": model_name,
                "model_order": model_idx,
                "batch_size": batch_size,
                "context_length": base_context_length,
                "vocab_size": vocab_size,
                **cfg,
            }
        )

    # Experiment B: vary context length at fixed GPT-2 XL model shape.
    xl_cfg = GPT2_SWIGLU_CONFIGS["GPT-2 XL"]
    for context_length in range(base_context_length, max_context_length + 1, context_step):
        cases.append(
            {
                "experiment": "context_sweep",
                "case_name": f"C={context_length}",
                "model_name": "GPT-2 XL",
                "model_order": MODEL_ORDER.index("GPT-2 XL"),
                "batch_size": batch_size,
                "context_length": context_length,
                "vocab_size": vocab_size,
                **xl_cfg,
            }
        )

    return pd.DataFrame(cases)


def run_experiments(cases: pd.DataFrame) -> pd.DataFrame:
    """
    Run the static profiler for each case and aggregate op-level FLOPs into
    assignment-level model components.
    """
    rows: list[dict[str, int | float | str]] = []

    for _, case in cases.iterrows():
        records, total_flops = profile_gpt_matmul_flops(
            batch_size=int(case["batch_size"]),
            context_length=int(case["context_length"]),
            num_layers=int(case["num_layers"]),
            d_model=int(case["d_model"]),
            num_heads=int(case["num_heads"]),
            d_ff=int(case["d_ff"]),
            vocab_size=int(case["vocab_size"]),
        )
        op_summary = summarize_by_op(records)

        unknown_ops = sorted(set(op_summary) - set(COMPONENT_MAP))
        if unknown_ops:
            raise ValueError(f"Missing component mapping for ops: {unknown_ops}")

        component_flops: dict[str, int] = {name: 0 for name in COMPONENT_ORDER}
        for op_name, flops in op_summary.items():
            component_flops[COMPONENT_MAP[op_name]] += flops

        for component_idx, component in enumerate(COMPONENT_ORDER):
            flops = component_flops[component]
            rows.append(
                {
                    "experiment": case["experiment"],
                    "case_name": case["case_name"],
                    "model_name": case["model_name"],
                    "model_order": int(case["model_order"]),
                    "context_length": int(case["context_length"]),
                    "batch_size": int(case["batch_size"]),
                    "num_layers": int(case["num_layers"]),
                    "d_model": int(case["d_model"]),
                    "num_heads": int(case["num_heads"]),
                    "d_ff": int(case["d_ff"]),
                    "vocab_size": int(case["vocab_size"]),
                    "component": component,
                    "component_order": component_idx,
                    "flops": int(flops),
                    "flops_tflops": flops / 1e12,
                    "total_flops": int(total_flops),
                    "total_tflops": total_flops / 1e12,
                    "proportion": flops / total_flops,
                }
            )

    df = pd.DataFrame(rows)
    df["component"] = pd.Categorical(df["component"], categories=COMPONENT_ORDER, ordered=True)
    df["model_name"] = pd.Categorical(df["model_name"], categories=MODEL_ORDER, ordered=True)
    df["case_name"] = pd.Categorical(
        df["case_name"],
        categories=list(dict.fromkeys(cases["case_name"].tolist())),
        ordered=True,
    )
    return df.sort_values(["experiment", "model_order", "context_length", "component_order"])


def plot_results(df: pd.DataFrame, output_dir: str | Path = "figures") -> list[Path]:
    """
    Generate the four figures needed for parts (d) and (e).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_df = df[df["experiment"] == "model_scale"].copy()
    context_df = df[df["experiment"] == "context_sweep"].copy()

    saved_paths: list[Path] = []

    common_theme = theme_bw() + theme(
        figure_size=(9, 5),
        axis_text_x=element_text(rotation=0, ha="center"),
        legend_title=element_text(size=9),
        legend_position="right",
    )

    # 1. Model scale: component proportion.
    p = (
        ggplot(model_df, aes(x="model_name", y="proportion", fill="component"))
        + geom_col()
        + scale_y_continuous(labels=percent_format())
        + labs(
            title="Component FLOPs share across GPT-2 model scales",
            x="Model",
            y="Share of total matmul FLOPs",
            fill="Component",
        )
        + common_theme
    )
    path = output_dir / "fig1_model_scale_component_proportion.png"
    p.save(path, dpi=180, verbose=False)
    saved_paths.append(path)

    # 2. Model scale: absolute FLOPs.
    p = (
        ggplot(model_df, aes(x="model_name", y="flops_tflops", fill="component"))
        + geom_col()
        + labs(
            title="Absolute component FLOPs across GPT-2 model scales",
            x="Model",
            y="Component FLOPs (TFLOPs)",
            fill="Component",
        )
        + common_theme
    )
    path = output_dir / "fig2_model_scale_absolute_flops.png"
    p.save(path, dpi=180, verbose=False)
    saved_paths.append(path)

    # 3. Context sweep: component proportion.
    p = (
        ggplot(context_df, aes(x="context_length", y="proportion", fill="component"))
        + geom_area()
        + scale_x_continuous(breaks=[1024, 4096, 8192, 12288, 16384])
        + scale_y_continuous(labels=percent_format())
        + labs(
            title="Component FLOPs share as context length increases",
            x="Context length",
            y="Share of total matmul FLOPs",
            fill="Component",
        )
        + common_theme
    )
    path = output_dir / "fig3_context_sweep_component_proportion.png"
    p.save(path, dpi=180, verbose=False)
    saved_paths.append(path)

    # 4. Context sweep: absolute FLOPs.
    p = (
        ggplot(context_df, aes(x="context_length", y="flops_tflops", color="component"))
        + geom_line(size=1.1)
        + geom_point(size=1.8)
        + scale_x_continuous(breaks=[1024, 4096, 8192, 12288, 16384])
        + labs(
            title="Absolute component FLOPs as context length increases",
            x="Context length",
            y="Component FLOPs (TFLOPs)",
            color="Component",
        )
        + common_theme
    )
    path = output_dir / "fig4_context_sweep_absolute_flops.png"
    p.save(path, dpi=180, verbose=False)
    saved_paths.append(path)

    return saved_paths


def make_report_tables(df: pd.DataFrame, output_dir: str | Path = "outputs") -> dict[str, Path]:
    """
    Create compact tables for assignment parts (d) and (e).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_df = df[df["experiment"] == "model_scale"].copy()
    context_df = df[df["experiment"] == "context_sweep"].copy()

    # Part d: component proportions by model scale.
    part_d_prop = (
        model_df.pivot_table(
            index="model_name",
            columns="component",
            values="proportion",
            observed=True,
        )
        .reindex(MODEL_ORDER)
        .reindex(columns=COMPONENT_ORDER)
    )
    part_d_total = (
        model_df.groupby("model_name", observed=True)["total_flops"]
        .first()
        .reindex(MODEL_ORDER)
    )
    part_d_table = part_d_prop.copy()
    for col in COMPONENT_ORDER:
        part_d_table[col] = part_d_table[col].map(lambda x: f"{x:.2%}")
    part_d_table["Total FLOPs"] = part_d_total.map(format_flops)

    # Part e: compare the two endpoints of the context sweep.
    endpoint_lengths = [int(context_df["context_length"].min()), int(context_df["context_length"].max())]
    part_e_base = context_df[context_df["context_length"].isin(endpoint_lengths)].copy()
    part_e_prop = (
        part_e_base.pivot_table(
            index="context_length",
            columns="component",
            values="proportion",
            observed=True,
        )
        .reindex(endpoint_lengths)
        .reindex(columns=COMPONENT_ORDER)
    )
    part_e_total = (
        part_e_base.groupby("context_length", observed=True)["total_flops"]
        .first()
        .reindex(endpoint_lengths)
    )
    part_e_table = part_e_prop.copy()
    for col in COMPONENT_ORDER:
        part_e_table[col] = part_e_table[col].map(lambda x: f"{x:.2%}")
    part_e_table["Total FLOPs"] = part_e_total.map(format_flops)

    # Extra summary for part e: total FLOPs growth factor.
    total_start = int(part_e_total.iloc[0])
    total_end = int(part_e_total.iloc[-1])
    growth_factor = total_end / total_start
    growth_table = pd.DataFrame(
        [
            {
                "Start context": endpoint_lengths[0],
                "End context": endpoint_lengths[1],
                "Start total FLOPs": format_flops(total_start),
                "End total FLOPs": format_flops(total_end),
                "Growth factor": f"{growth_factor:.2f}x",
            }
        ]
    )

    paths = {
        "part_d_markdown": output_dir / "part_d_model_scale_table.md",
        "part_e_markdown": output_dir / "part_e_context_endpoint_table.md",
        "context_growth_markdown": output_dir / "part_e_context_growth_summary.md",
        "component_csv": output_dir / "flops_component_results.csv",
        "cases_csv": output_dir / "flops_experiment_cases.csv",
    }

    paths["part_d_markdown"].write_text(part_d_table.to_markdown(), encoding="utf-8")
    paths["part_e_markdown"].write_text(part_e_table.to_markdown(), encoding="utf-8")
    paths["context_growth_markdown"].write_text(growth_table.to_markdown(index=False), encoding="utf-8")
    df.to_csv(paths["component_csv"], index=False)
    # Reconstruct cases from unique config columns for reproducibility.
    case_cols = [
        "experiment",
        "case_name",
        "model_name",
        "model_order",
        "batch_size",
        "context_length",
        "num_layers",
        "d_model",
        "num_heads",
        "d_ff",
        "vocab_size",
    ]
    df[case_cols].drop_duplicates().to_csv(paths["cases_csv"], index=False)

    return paths


def main() -> None:
    root = Path(__file__).resolve().parent
    output_dir = root / "flops_outputs"
    figure_dir = output_dir / "figures"
    table_dir = output_dir / "tables"

    cases = build_experiment_cases()
    df = run_experiments(cases)
    figure_paths = plot_results(df, figure_dir)
    table_paths = make_report_tables(df, table_dir)

    print("Generated figures:")
    for path in figure_paths:
        print(f"  - {path}")

    print("\nGenerated tables / data:")
    for name, path in table_paths.items():
        print(f"  - {name}: {path}")

    print("\nPart (d) table:")
    print((table_dir / "part_d_model_scale_table.md").read_text(encoding="utf-8"))

    print("\nPart (e) endpoint table:")
    print((table_dir / "part_e_context_endpoint_table.md").read_text(encoding="utf-8"))

    print("\nPart (e) total growth:")
    print((table_dir / "part_e_context_growth_summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
