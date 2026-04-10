"""Benchmark runner and paper comparison reports for SER experiments."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from src.config import REPORTS_PATH, RUNS_PATH, ensure_directories
from src.train import train_model


PAPER_BENCHMARKS = [
    {
        "id": 1,
        "author_year": "Ouyang (2025)",
        "title": "Speech Emotion Detection based on MFCC and CNN-LSTM Architecture",
        "reported_accuracy": 0.6134,
        "drawback": "Accuracy varies significantly across emotion classes.",
        "source": "https://arxiv.org/abs/2501.10666",
        "comparable": True,
    },
    {
        "id": 2,
        "author_year": "Salian et al. (2021)",
        "title": "Speech Emotion Recognition Using Time Distributed CNN and LSTM",
        "reported_accuracy": 0.8926,
        "drawback": "Requires large dataset and high training time.",
        "source": "https://doi.org/10.1051/itmconf/20214003006",
        "comparable": True,
    },
    {
        "id": 3,
        "author_year": "Zhao et al. (2018)",
        "title": "Speech Emotion Recognition Using Deep 1D & 2D CNN-LSTM Networks",
        "reported_accuracy": None,
        "drawback": "Black-box behavior and significant training-data requirements.",
        "source": "https://doi.org/10.1016/j.neucom.2017.10.005",
        "comparable": False,
    },
    {
        "id": 4,
        "author_year": "Ullah et al. (2023)",
        "title": "Speech Emotion Recognition Using CNN and Multi-head Convolutional Transformer",
        "reported_accuracy": 0.8231,
        "drawback": "Performance drops with noisy speech and speaker variations.",
        "source": "https://doi.org/10.3390/s23136212",
        "comparable": True,
    },
    {
        "id": 5,
        "author_year": "Madanian et al. (2023)",
        "title": "Speech Emotion Recognition Using Machine Learning - A Review",
        "reported_accuracy": None,
        "drawback": "Many systems fail to generalize to real-world environments.",
        "source": "https://www.sciencedirect.com/science/article/pii/S2667305323000911",
        "comparable": False,
    },
    {
        "id": 6,
        "author_year": "Bhanbhro et al. (2025)",
        "title": "Comparative Analysis of CNN-LSTM and Attention-Enhanced CNN-LSTM Models",
        "reported_accuracy": 0.9601,
        "drawback": "Attention models increase computational complexity.",
        "source": "https://doi.org/10.3390/signals6020022",
        "comparable": True,
    },
]


DRAWBACK_MITIGATION_MAP = {
    1: "Added richer feature bundle + per-class metrics + UAR monitoring to reduce class-wise instability.",
    2: "Added lightweight variant and protocol controls to reduce compute when needed.",
    3: "Added benchmark report with explicit per-dataset and per-protocol diagnostics for interpretability.",
    4: "Added waveform augmentation (noise/shift/speed/pitch) and SpecAugment for robustness.",
    5: "Added speaker-independent protocol as first-class metric for real-world generalization.",
    6: "Added attention-enhanced and lightweight models to balance accuracy and complexity.",
}


def _latest_run_file() -> Optional[Path]:
    run_files = sorted(RUNS_PATH.glob("*.json"), key=lambda p: p.stat().st_mtime)
    return run_files[-1] if run_files else None


def load_run_summary(run_id: Optional[str] = None) -> dict:
    """Load run summary from outputs/runs."""
    ensure_directories()
    if run_id:
        run_path = RUNS_PATH / f"{run_id}.json"
        if not run_path.exists():
            raise FileNotFoundError(f"Run summary not found: {run_path}")
        return json.loads(run_path.read_text(encoding="utf-8"))

    latest = _latest_run_file()
    if latest is None:
        raise FileNotFoundError("No run summaries found in outputs/runs.")
    return json.loads(latest.read_text(encoding="utf-8"))


def _format_pct(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    return f"{value * 100:.2f}%"


def _paper_comparison_rows(run_summary: dict) -> List[dict]:
    rows: List[dict] = []

    best_accuracy = None
    best_protocol = None
    for protocol_name, result in run_summary.get("results", {}).items():
        acc = result.get("metrics", {}).get("accuracy")
        if isinstance(acc, (float, int)) and (best_accuracy is None or acc > best_accuracy):
            best_accuracy = float(acc)
            best_protocol = protocol_name

    for paper in PAPER_BENCHMARKS:
        reported = paper["reported_accuracy"]
        beats = None
        if reported is not None and best_accuracy is not None:
            beats = best_accuracy > reported

        rows.append(
            {
                "paper_id": paper["id"],
                "paper": paper["author_year"],
                "reported_accuracy": reported,
                "our_best_accuracy": best_accuracy,
                "our_best_protocol": best_protocol,
                "beats_paper": beats,
                "comparable": paper["comparable"],
                "drawback": paper["drawback"],
                "mitigation": DRAWBACK_MITIGATION_MAP.get(paper["id"], "N/A"),
                "source": paper["source"],
            }
        )

    return rows


def build_benchmark_payload(run_summary: dict) -> dict:
    """Build deterministic benchmark payload from a run summary."""
    rows = _paper_comparison_rows(run_summary)

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "run_id": run_summary.get("run_id"),
        "datasets": run_summary.get("datasets", []),
        "protocol": run_summary.get("protocol"),
        "model_variant": run_summary.get("model_variant"),
        "feature_bundle": run_summary.get("feature_bundle"),
        "results": run_summary.get("results", {}),
        "paper_comparison": rows,
    }
    return payload


def render_benchmark_markdown(payload: dict) -> str:
    """Render benchmark payload to markdown report."""
    lines: List[str] = []
    lines.append("# SER Benchmark Report")
    lines.append("")
    lines.append(f"- Run ID: `{payload.get('run_id')}`")
    lines.append(f"- Generated: {payload.get('generated_at')}")
    lines.append(f"- Datasets: `{','.join(payload.get('datasets', []))}`")
    lines.append(f"- Protocol selector: `{payload.get('protocol')}`")
    lines.append(f"- Model variant: `{payload.get('model_variant')}`")
    lines.append(f"- Feature bundle: `{payload.get('feature_bundle')}`")
    lines.append("")

    lines.append("## Per-Protocol Metrics")
    lines.append("")
    lines.append("| Protocol | Accuracy | Macro-F1 | UAR | Weighted F1 |")
    lines.append("|---|---:|---:|---:|---:|")

    for protocol_name in sorted(payload.get("results", {}).keys()):
        metrics = payload["results"][protocol_name].get("metrics", {})
        lines.append(
            f"| {protocol_name} | {_format_pct(metrics.get('accuracy'))} | "
            f"{_format_pct(metrics.get('macro_f1'))} | {_format_pct(metrics.get('uar'))} | "
            f"{_format_pct(metrics.get('f1_score'))} |"
        )

    lines.append("")
    lines.append("## Per-Dataset Metrics")
    lines.append("")
    lines.append("| Protocol | Dataset | Samples | Accuracy | Macro-F1 | UAR |")
    lines.append("|---|---|---:|---:|---:|---:|")

    for protocol_name in sorted(payload.get("results", {}).keys()):
        per_dataset = payload["results"][protocol_name].get("per_dataset_metrics", {})
        for dataset_id in sorted(per_dataset.keys()):
            metrics = per_dataset[dataset_id]
            lines.append(
                f"| {protocol_name} | {dataset_id} | {metrics.get('num_samples', 0)} | "
                f"{_format_pct(metrics.get('accuracy'))} | {_format_pct(metrics.get('macro_f1'))} | "
                f"{_format_pct(metrics.get('uar'))} |"
            )

    lines.append("")
    lines.append("## Paper Comparison (First 6)")
    lines.append("")
    lines.append("| # | Paper | Reported Accuracy | Our Best (Protocol) | Beats? | Comparable |")
    lines.append("|---:|---|---:|---:|---|---|")

    for row in payload.get("paper_comparison", []):
        best_str = "N/A"
        if row["our_best_accuracy"] is not None:
            best_str = f"{_format_pct(row['our_best_accuracy'])} ({row['our_best_protocol']})"
        beats = "N/A" if row["beats_paper"] is None else ("Yes" if row["beats_paper"] else "No")
        lines.append(
            f"| {row['paper_id']} | {row['paper']} | {_format_pct(row['reported_accuracy'])} | "
            f"{best_str} | {beats} | {'Yes' if row['comparable'] else 'No'} |"
        )

    lines.append("")
    lines.append("## Drawback Mitigation Mapping")
    lines.append("")
    lines.append("| # | Drawback from Paper | Our Mitigation | Source |")
    lines.append("|---:|---|---|---|")

    for row in payload.get("paper_comparison", []):
        lines.append(
            f"| {row['paper_id']} | {row['drawback']} | {row['mitigation']} | [link]({row['source']}) |"
        )

    return "\n".join(lines) + "\n"


def run_benchmark(
    papers: str = "first6",
    protocol: str = "dual",
    run_id: Optional[str] = None,
    datasets: Optional[Sequence[str]] = None,
    train_if_missing: bool = False,
    model_variant: str = "attention",
    feature_bundle: str = "robust",
    epochs: int = 15,
    batch_size: int = 32,
) -> dict:
    """Run benchmark report generation, optionally auto-training if missing."""
    if papers.strip().lower() != "first6":
        raise ValueError("Only papers='first6' is currently supported.")

    ensure_directories()

    try:
        run_summary = load_run_summary(run_id=run_id)
    except FileNotFoundError:
        if not train_if_missing:
            raise
        selected_datasets = list(datasets) if datasets else ["ravdess", "crema_d", "tess", "savee"]
        run_summary = train_model(
            datasets=selected_datasets,
            protocol=protocol,
            model_variant=model_variant,
            feature_bundle=feature_bundle,
            epochs=epochs,
            batch_size=batch_size,
            use_focal_loss=True,
            use_augmentation=True,
        )

    payload = build_benchmark_payload(run_summary)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_base = f"benchmark_{payload.get('run_id', 'run')}_{stamp}"

    report_markdown = render_benchmark_markdown(payload)
    report_path = REPORTS_PATH / f"{report_base}.md"
    payload_path = REPORTS_PATH / f"{report_base}.json"

    report_path.write_text(report_markdown, encoding="utf-8")
    payload_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    print(f"Benchmark report saved: {report_path}")
    print(f"Benchmark payload saved: {payload_path}")

    return {
        "report_path": str(report_path),
        "payload_path": str(payload_path),
        "payload": payload,
    }
