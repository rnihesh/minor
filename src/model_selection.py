"""Helpers for selecting saved SER models from run summaries."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from src.config import MODEL_PATH, RUNS_PATH


@dataclass(frozen=True)
class ModelRecord:
    """Metadata for a saved model candidate."""

    model_path: str
    accuracy: float
    modified_time: float
    run_id: str
    protocol_name: str
    datasets: tuple[str, ...]


def _protocol_name(protocol: Optional[str]) -> Optional[str]:
    if protocol is None:
        return None
    key = protocol.strip().lower()
    if key == "random":
        return "random_stratified"
    if key == "speaker":
        return "speaker_independent"
    if key in {"random_stratified", "speaker_independent"}:
        return key
    return None


def _records_from_run_summaries() -> list[ModelRecord]:
    if not RUNS_PATH.exists():
        return []

    records: list[ModelRecord] = []
    for run_path in sorted(RUNS_PATH.glob("*.json")):
        try:
            payload = json.loads(run_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue

        run_id = str(payload.get("run_id") or run_path.stem)
        datasets = tuple(str(name) for name in payload.get("datasets", []))
        results = payload.get("results", {})
        if not isinstance(results, dict):
            continue

        for protocol_name, result in results.items():
            if not isinstance(result, dict):
                continue
            model_path = result.get("best_model_path")
            accuracy = result.get("metrics", {}).get("accuracy")
            if not isinstance(model_path, str) or not isinstance(accuracy, (float, int)):
                continue
            path = Path(model_path)
            if not path.exists():
                continue
            records.append(
                ModelRecord(
                    model_path=model_path,
                    accuracy=float(accuracy),
                    modified_time=path.stat().st_mtime,
                    run_id=run_id,
                    protocol_name=str(protocol_name),
                    datasets=datasets,
                )
            )
    return records


def find_best_model_record(
    prefer_protocol: Optional[str] = None,
    datasets: Optional[Sequence[str]] = None,
) -> Optional[ModelRecord]:
    """Return the best recorded model, optionally constrained by protocol/datasets."""
    records = _records_from_run_summaries()
    if not records:
        return None

    protocol_name = _protocol_name(prefer_protocol)
    if protocol_name:
        protocol_matches = [record for record in records if record.protocol_name == protocol_name]
        if protocol_matches:
            records = protocol_matches

    if datasets:
        requested = tuple(name.strip().lower() for name in datasets if name.strip())
        dataset_matches = [
            record
            for record in records
            if tuple(name.lower() for name in record.datasets) == requested
        ]
        if dataset_matches:
            records = dataset_matches

    return max(records, key=lambda record: (record.accuracy, record.modified_time))


def find_best_model_path(
    prefer_protocol: Optional[str] = None,
    datasets: Optional[Sequence[str]] = None,
) -> str:
    """Find the strongest saved model path, falling back to newest model file."""
    record = find_best_model_record(prefer_protocol=prefer_protocol, datasets=datasets)
    if record is not None:
        return record.model_path

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model directory not found: {MODEL_PATH}")

    model_files = [
        MODEL_PATH / filename
        for filename in os.listdir(MODEL_PATH)
        if filename.endswith("_best.keras")
    ]
    if not model_files:
        model_files = [
            MODEL_PATH / filename
            for filename in os.listdir(MODEL_PATH)
            if filename.endswith(".keras")
        ]
    if not model_files:
        raise FileNotFoundError(f"No model found in {MODEL_PATH}")

    return str(max(model_files, key=lambda path: path.stat().st_mtime))
