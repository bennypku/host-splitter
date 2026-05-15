"""Sliding-window speaker embedding extraction via 3D-Speaker ERes2Net."""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import soundfile as sf

from .config import CFG

_MODEL = None
_DEVICE = None


def _get_model():
    """Load the underlying ERes2Net model directly (bypassing pipeline wrapper)."""
    global _MODEL, _DEVICE
    if _MODEL is not None:
        return _MODEL, _DEVICE

    import torch
    from modelscope.pipelines import pipeline
    from modelscope.utils.constant import Tasks

    p = pipeline(task=Tasks.speaker_verification, model=CFG.embedding_model)
    _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _MODEL = p.model
    _MODEL.eval()
    _MODEL.to(_DEVICE)
    return _MODEL, _DEVICE


def extract_embeddings(
    wav_path: Path,
    batch_size: int = 32,
    max_windows: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sliding-window embeddings over a wav file. Streams windows from disk."""
    import traceback
    import torch

    with sf.SoundFile(str(wav_path)) as f:
        sr = f.samplerate
        total_frames = len(f)
        channels = f.channels
        if sr != CFG.sample_rate:
            raise ValueError(f"Expected {CFG.sample_rate} Hz, got {sr}")

        win = int(CFG.window_sec * sr)
        hop = int(CFG.hop_sec * sr)
        if total_frames < win:
            return np.zeros((0, 192), dtype=np.float32), np.zeros((0,), dtype=np.float32)

        starts = np.arange(0, total_frames - win + 1, hop, dtype=np.int64)
        if max_windows is not None:
            starts = starts[:max(0, max_windows)]
        n = len(starts)
        times = ((starts + win / 2) / sr).astype(np.float32)

        model, device = _get_model()
        emb_dim = 192
        embs = np.zeros((n, emb_dim), dtype=np.float32)
        rms_threshold = 1e-3
        progress_every = max(1, n // 20)

        with torch.no_grad():
            for i, s in enumerate(starts):
                if i % progress_every == 0:
                    print(f"  embedding {i}/{n}", flush=True)
                f.seek(int(s))
                chunk = f.read(win, dtype="float32", always_2d=False)
                if channels > 1:
                    chunk = chunk.mean(axis=1)
                chunk = np.ascontiguousarray(chunk, dtype=np.float32)
                if float(np.sqrt(np.mean(chunk ** 2))) < rms_threshold:
                    continue
                wav_t = torch.from_numpy(chunk).to(device)
                try:
                    out = model(wav_t)
                except Exception as e:
                    print(f"  !! model() failed at window {i}: {type(e).__name__}: {e}", flush=True)
                    traceback.print_exc()
                    raise
                v = out.detach().cpu().reshape(-1).numpy().astype(np.float32)
                del out, wav_t
                norm = float(np.linalg.norm(v)) + 1e-9
                embs[i] = v / norm

    return embs, times
