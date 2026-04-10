"""Unified dataset ingestion for multi-dataset SER training."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
import soundfile as sf

from src.config import EMOTIONS, EMOTION_TO_ID, SUPPORTED_DATASETS, resolve_dataset_root


METADATA_COLUMNS = ["dataset_id", "speaker_id", "emotion_id", "sr", "duration", "path"]

_CREMA_EMOTION_MAP = {
    "ANG": "angry",
    "DIS": "disgust",
    "FEA": "fearful",
    "HAP": "happy",
    "NEU": "neutral",
    "SAD": "sad",
}

_TESS_EMOTION_MAP = {
    "angry": "angry",
    "disgust": "disgust",
    "fear": "fearful",
    "fearful": "fearful",
    "happy": "happy",
    "neutral": "neutral",
    "sad": "sad",
    "ps": "surprised",
    "pleasant_surprise": "surprised",
    "surprised": "surprised",
}

_SAVEE_EMOTION_MAP = {
    "a": "angry",
    "d": "disgust",
    "f": "fearful",
    "h": "happy",
    "n": "neutral",
    "sa": "sad",
    "su": "surprised",
    "s": "surprised",
}


class DatasetError(RuntimeError):
    """Raised when dataset ingestion fails in strict mode."""


def _audio_info(path: Path) -> Tuple[Optional[int], Optional[float]]:
    """Return (sample_rate, duration_seconds)."""
    try:
        info = sf.info(path)
        duration = float(info.frames) / float(info.samplerate) if info.samplerate else None
        return int(info.samplerate), duration
    except Exception:
        return None, None


def _build_record(dataset_id: str, speaker_raw: str, emotion_name: str, path: Path) -> Optional[dict]:
    """Build one normalized metadata record."""
    if emotion_name not in EMOTION_TO_ID:
        return None

    sr, duration = _audio_info(path)
    return {
        "dataset_id": dataset_id,
        "speaker_id": f"{dataset_id}:{speaker_raw}",
        "emotion_id": EMOTION_TO_ID[emotion_name],
        "sr": sr,
        "duration": duration,
        "path": str(path.resolve()),
    }


def _scan_ravdess(root: Path) -> List[dict]:
    records: List[dict] = []
    for actor_dir in sorted(root.glob("Actor_*")):
        if not actor_dir.is_dir():
            continue
        for wav_path in sorted(actor_dir.glob("*.wav")):
            parts = wav_path.stem.split("-")
            if len(parts) != 7:
                continue
            modality, vocal_channel, emotion_code = parts[0], parts[1], parts[2]
            if modality != "03" or vocal_channel != "01":
                continue
            emotion_name = EMOTIONS.get(emotion_code)
            if emotion_name is None:
                continue
            rec = _build_record("ravdess", parts[6], emotion_name, wav_path)
            if rec:
                records.append(rec)
    return records


def _scan_crema_d(root: Path) -> List[dict]:
    records: List[dict] = []
    for wav_path in sorted(root.rglob("*.wav")):
        parts = wav_path.stem.split("_")
        if len(parts) < 3:
            continue
        speaker, emotion_code = parts[0], parts[2].upper()
        emotion_name = _CREMA_EMOTION_MAP.get(emotion_code)
        if emotion_name is None:
            continue
        rec = _build_record("crema_d", speaker, emotion_name, wav_path)
        if rec:
            records.append(rec)
    return records


def _parse_tess_emotion(path: Path) -> Optional[str]:
    parent = path.parent.name.lower()
    stem = path.stem.lower()

    candidates = []
    if "_" in parent:
        candidates.append(parent.split("_")[-1])
    if "_" in stem:
        candidates.append(stem.split("_")[-1])
    candidates.extend([parent, stem])

    for candidate in candidates:
        if candidate in _TESS_EMOTION_MAP:
            return _TESS_EMOTION_MAP[candidate]
        for key, value in _TESS_EMOTION_MAP.items():
            if key in candidate:
                return value
    return None


def _scan_tess(root: Path) -> List[dict]:
    records: List[dict] = []
    for wav_path in sorted(root.rglob("*.wav")):
        emotion_name = _parse_tess_emotion(wav_path)
        if emotion_name is None:
            continue
        stem_parts = wav_path.stem.split("_")
        speaker = stem_parts[0] if stem_parts else wav_path.parent.name.split("_")[0]
        rec = _build_record("tess", speaker, emotion_name, wav_path)
        if rec:
            records.append(rec)
    return records


def _extract_savee_code(code_blob: str) -> Optional[str]:
    lower = code_blob.lower()
    for code in ("sa", "su", "a", "d", "f", "h", "n", "s"):
        if lower.startswith(code):
            return code
    match = re.match(r"([a-z]+)", lower)
    if match:
        guessed = match.group(1)
        return guessed if guessed in _SAVEE_EMOTION_MAP else None
    return None


def _scan_savee(root: Path) -> List[dict]:
    records: List[dict] = []
    for wav_path in sorted(root.rglob("*.wav")):
        stem = wav_path.stem
        if "_" not in stem:
            continue
        speaker, code_blob = stem.split("_", 1)
        code = _extract_savee_code(code_blob)
        if code is None:
            continue
        emotion_name = _SAVEE_EMOTION_MAP.get(code)
        if emotion_name is None:
            continue
        rec = _build_record("savee", speaker, emotion_name, wav_path)
        if rec:
            records.append(rec)
    return records


_SCANNERS = {
    "ravdess": _scan_ravdess,
    "crema_d": _scan_crema_d,
    "tess": _scan_tess,
    "savee": _scan_savee,
}


def scan_dataset(dataset_id: str, root: Path) -> List[dict]:
    """Scan a dataset root and return normalized metadata records."""
    if dataset_id not in _SCANNERS:
        raise ValueError(f"Unsupported dataset: {dataset_id}")
    return _SCANNERS[dataset_id](root)


def validate_metadata_schema(metadata: pd.DataFrame) -> None:
    """Validate required columns and core constraints."""
    missing = [column for column in METADATA_COLUMNS if column not in metadata.columns]
    if missing:
        raise ValueError(f"Metadata missing required columns: {missing}")

    if metadata.empty:
        raise ValueError("Metadata is empty. Provide at least one dataset path with WAV files.")

    invalid_emotions = metadata.loc[~metadata["emotion_id"].isin(list(range(len(EMOTION_TO_ID)))), "emotion_id"]
    if not invalid_emotions.empty:
        raise ValueError("Metadata contains invalid emotion IDs.")


def load_unified_metadata(
    datasets: Sequence[str],
    dataset_paths: Optional[Dict[str, str]] = None,
    strict: bool = False,
) -> pd.DataFrame:
    """Load metadata from selected datasets using a unified schema."""
    selected = [dataset.strip().lower() for dataset in datasets if dataset.strip()]
    if not selected:
        raise ValueError("At least one dataset must be provided.")

    for dataset_id in selected:
        if dataset_id not in SUPPORTED_DATASETS:
            raise ValueError(f"Unsupported dataset '{dataset_id}'. Supported: {SUPPORTED_DATASETS}")

    records: List[dict] = []
    missing_roots: List[str] = []

    for dataset_id in selected:
        root = resolve_dataset_root(dataset_id, dataset_paths)
        if root is None or not root.exists():
            missing_roots.append(dataset_id)
            continue

        dataset_records = scan_dataset(dataset_id, root)
        records.extend(dataset_records)

    if missing_roots and strict:
        raise DatasetError(
            "Missing dataset roots for: " + ", ".join(missing_roots)
        )

    if not records:
        available = ", ".join(selected)
        raise DatasetError(
            f"No audio samples were discovered for selected datasets: {available}."
        )

    metadata = pd.DataFrame.from_records(records, columns=METADATA_COLUMNS)
    validate_metadata_schema(metadata)
    return metadata


def metadata_summary(metadata: pd.DataFrame) -> dict:
    """Return compact metadata summary for logging/reporting."""
    validate_metadata_schema(metadata)
    summary = {
        "num_samples": int(len(metadata)),
        "datasets": metadata["dataset_id"].value_counts().to_dict(),
        "speakers": metadata["speaker_id"].nunique(),
        "emotion_distribution": metadata["emotion_id"].value_counts().sort_index().to_dict(),
    }
    return summary
