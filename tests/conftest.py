"""Shared pytest fixtures for SER pipeline tests."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pytest
import soundfile as sf


def _write_wave(path: Path, sr: int = 22050, duration: float = 0.35, freq: float = 220.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    signal = 0.2 * np.sin(2 * np.pi * freq * t)
    sf.write(path, signal.astype(np.float32), sr)


@pytest.fixture()
def mini_dataset_paths(tmp_path: Path) -> Dict[str, str]:
    """
    Build a tiny but valid multi-dataset layout used by tests.

    Includes:
      - RAVDESS-like audio-only speech files
      - CREMA-D naming pattern
      - TESS naming pattern
      - SAVEE naming pattern
    """
    base = tmp_path / "datasets"
    ravdess = base / "ravdess"
    crema_d = base / "crema_d"
    tess = base / "tess"
    savee = base / "savee"

    # RAVDESS: enough samples for stratified + speaker splits.
    for actor_idx in range(1, 5):
        actor = f"{actor_idx:02d}"
        actor_dir = ravdess / f"Actor_{actor}"
        for emotion_idx in range(1, 9):
            emotion = f"{emotion_idx:02d}"
            for repetition in ("01", "02"):
                file_name = f"03-01-{emotion}-01-01-{repetition}-{actor}.wav"
                _write_wave(actor_dir / file_name, freq=180 + (emotion_idx * 25))

    # CREMA-D
    crema_samples = [
        "1001_DFA_ANG_XX.wav",
        "1002_ITS_DIS_XX.wav",
        "1003_IEO_FEA_MD.wav",
        "1004_TIE_HAP_HI.wav",
        "1005_IWW_NEU_MD.wav",
        "1006_TSI_SAD_LO.wav",
    ]
    for idx, file_name in enumerate(crema_samples):
        _write_wave(crema_d / file_name, freq=300 + (idx * 20))

    # TESS
    tess_samples = [
        ("OAF_angry", "OAF_back_angry.wav"),
        ("OAF_happy", "OAF_back_happy.wav"),
        ("YAF_neutral", "YAF_back_neutral.wav"),
        ("YAF_ps", "YAF_back_ps.wav"),
    ]
    for idx, (folder, file_name) in enumerate(tess_samples):
        _write_wave(tess / folder / file_name, freq=380 + (idx * 20))

    # SAVEE
    savee_samples = [
        "DC_a01.wav",
        "JE_d02.wav",
        "JK_f03.wav",
        "KL_h04.wav",
        "DC_n05.wav",
        "JE_sa06.wav",
        "JK_su07.wav",
    ]
    for idx, file_name in enumerate(savee_samples):
        _write_wave(savee / file_name, freq=460 + (idx * 20))

    return {
        "ravdess": str(ravdess),
        "crema_d": str(crema_d),
        "tess": str(tess),
        "savee": str(savee),
    }
