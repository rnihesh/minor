"""Dataset download helpers for open SER datasets."""

from __future__ import annotations

import shutil
import tarfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

from src.config import DATA_ROOT


DATASET_SOURCES = {
    "ravdess": {
        "description": "RAVDESS emotional speech audio",
        "urls": [
            "https://zenodo.org/records/1188976/files/Audio_Speech_Actors_01-24.zip?download=1",
        ],
        "manual": "If direct download fails, place extracted Actor_01..Actor_24 folders under data/ravdess or use existing archive/.",
    },
    "crema_d": {
        "description": "CREMA-D audio WAV",
        "urls": [
            "https://zenodo.org/records/1188976/files/AudioWAV.zip?download=1",
            "https://github.com/CheyneyComputerScience/CREMA-D/raw/master/AudioWAV.zip",
        ],
        "manual": "If blocked, download CREMA-D AudioWAV manually and extract into data/crema_d/.",
    },
    "tess": {
        "description": "Toronto emotional speech set (TESS)",
        "urls": [
            "https://tspace.library.utoronto.ca/bitstream/1807/24487/1/TESS_Toronto_emotional_speech_set_data.zip",
        ],
        "manual": "If blocked, download TESS manually and extract into data/tess/.",
    },
    "savee": {
        "description": "Surrey SAVEE dataset",
        "urls": [
            "https://datarepository.wolframcloud.com/resources/Surrey-Audio-Visual-Expressed-Emotion-SAVEE-Database",
        ],
        "manual": "SAVEE may require manual access; place extracted WAV files under data/savee/.",
    },
}


def _download_file(url: str, output_file: Path, timeout: int = 120) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response, output_file.open("wb") as f:
        shutil.copyfileobj(response, f)


def _extract_archive(archive_path: Path, target_dir: Path) -> None:
    suffixes = "".join(archive_path.suffixes).lower()
    if suffixes.endswith(".zip"):
        with zipfile.ZipFile(archive_path, "r") as zip_ref:
            zip_ref.extractall(target_dir)
        return
    if suffixes.endswith(".tar.gz") or suffixes.endswith(".tgz"):
        with tarfile.open(archive_path, "r:gz") as tar_ref:
            tar_ref.extractall(target_dir)
        return
    raise ValueError(f"Unsupported archive format: {archive_path.name}")


def _download_dataset(dataset_id: str, root: Path) -> dict:
    dataset_dir = root / dataset_id
    dataset_dir.mkdir(parents=True, exist_ok=True)

    source = DATASET_SOURCES[dataset_id]
    urls = source["urls"]
    errors: List[str] = []

    for idx, url in enumerate(urls, start=1):
        archive_path = dataset_dir / f"download_{idx}.bin"
        try:
            print(f"Attempting download for {dataset_id}: {url}")
            _download_file(url, archive_path)

            # Best-effort archive extraction.
            guessed_name = url.split("?")[0].split("/")[-1] or archive_path.name
            target_archive = dataset_dir / guessed_name
            archive_path.rename(target_archive)

            try:
                _extract_archive(target_archive, dataset_dir)
                print(f"Extracted {target_archive.name} to {dataset_dir}")
            except Exception as extract_exc:
                errors.append(f"Extraction failed for {target_archive.name}: {extract_exc}")
                continue

            return {
                "dataset": dataset_id,
                "status": "downloaded",
                "path": str(dataset_dir),
                "source_url": url,
            }
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
            errors.append(str(exc))
            if archive_path.exists():
                archive_path.unlink(missing_ok=True)

    return {
        "dataset": dataset_id,
        "status": "manual_required",
        "path": str(dataset_dir),
        "manual_instructions": source["manual"],
        "errors": errors,
    }


def download_datasets(pack: str = "open4", root: Optional[str] = None) -> dict:
    """Download datasets for the requested pack."""
    pack_key = pack.strip().lower()
    if pack_key != "open4":
        raise ValueError("Only pack='open4' is currently supported.")

    target_root = Path(root).expanduser().resolve() if root else DATA_ROOT
    target_root.mkdir(parents=True, exist_ok=True)

    results = {}
    for dataset_id in ["ravdess", "crema_d", "tess", "savee"]:
        results[dataset_id] = _download_dataset(dataset_id, target_root)

    return {
        "pack": "open4",
        "root": str(target_root),
        "results": results,
    }
