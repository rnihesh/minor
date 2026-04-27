"""Evaluation metric and plotting regression tests."""

from __future__ import annotations

import numpy as np

from src.evaluate import compute_classification_metrics, plot_per_class_metrics


def test_per_class_plot_handles_missing_canonical_classes(tmp_path):
    y_true = np.array([0, 2, 3, 4, 5, 6, 7])
    y_pred = y_true.copy()

    metrics = compute_classification_metrics(y_true, y_pred)
    save_path = tmp_path / "per_class_metrics.png"
    plot_per_class_metrics(y_true, y_pred, save_path=str(save_path))

    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 0.875
    assert save_path.exists()
