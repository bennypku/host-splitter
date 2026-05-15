"""Audio extraction and vocal separation."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from .config import CFG


def extract_audio(video_path: Path, out_wav: Path) -> Path:
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn", "-ac", "1", "-ar", str(CFG.sample_rate),
        "-acodec", "pcm_s16le",
        str(out_wav),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    return out_wav


def separate_vocals(wav_path: Path, out_dir: Path, model: str = "htdemucs") -> Path:
    """Run Demucs and return the vocals stem path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "python", "-m", "demucs.separate",
        "-n", model,
        "--two-stems", "vocals",
        "-o", str(out_dir),
        str(wav_path),
    ]
    subprocess.run(cmd, check=True)
    vocals = out_dir / model / wav_path.stem / "vocals.wav"
    if not vocals.exists():
        raise FileNotFoundError(f"Demucs vocals stem not found: {vocals}")
    return vocals


def prepare_audio(video_path: Path, work_dir: Path, do_demucs: bool = True) -> Path:
    """Full preprocess: video -> 16k mono wav -> (optional) vocal stem.

    Returns the path of the audio to feed into embedding extraction.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    raw_wav = work_dir / f"{video_path.stem}.16k.wav"
    if not raw_wav.exists():
        extract_audio(video_path, raw_wav)

    if not do_demucs:
        return raw_wav

    if shutil.which("python") is None:
        raise RuntimeError("python not on PATH; cannot run demucs")

    demucs_root = work_dir / "demucs"
    vocals = demucs_root / "htdemucs" / raw_wav.stem / "vocals.wav"
    if not vocals.exists():
        separate_vocals(raw_wav, demucs_root)
    return vocals
