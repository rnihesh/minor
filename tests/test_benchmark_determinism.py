"""Determinism checks for benchmark report rendering."""

from __future__ import annotations

from src.benchmark import build_benchmark_payload, render_benchmark_markdown


def test_benchmark_render_is_stable_for_same_payload():
    run_summary = {
        "run_id": "smoke_run",
        "datasets": ["ravdess"],
        "protocol": "dual",
        "model_variant": "attention",
        "feature_bundle": "robust",
        "results": {
            "random_stratified": {
                "metrics": {
                    "accuracy": 0.91,
                    "macro_f1": 0.89,
                    "uar": 0.88,
                    "f1_score": 0.90,
                },
                "per_dataset_metrics": {
                    "ravdess": {
                        "num_samples": 100,
                        "accuracy": 0.91,
                        "macro_f1": 0.89,
                        "uar": 0.88,
                    }
                },
            },
            "speaker_independent": {
                "metrics": {
                    "accuracy": 0.82,
                    "macro_f1": 0.80,
                    "uar": 0.79,
                    "f1_score": 0.81,
                },
                "per_dataset_metrics": {
                    "ravdess": {
                        "num_samples": 120,
                        "accuracy": 0.82,
                        "macro_f1": 0.80,
                        "uar": 0.79,
                    }
                },
            },
        },
    }

    payload = build_benchmark_payload(run_summary)
    payload["generated_at"] = "2026-04-10T11:30:00"

    first = render_benchmark_markdown(payload)
    second = render_benchmark_markdown(payload)

    assert first == second
    assert "Paper Comparison (First 6)" in first
    assert first.count("| 1 | Ouyang (2025) |") == 1
