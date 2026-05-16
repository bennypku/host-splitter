"""Sliding-window speaker embedding extraction via 3D-Speaker models."""
from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Tuple

import numpy as np
import soundfile as sf

from .config import CFG

_MODEL = None
_DEVICE = None


def _get_model():
    """Load the underlying speaker verification model directly."""
    global _MODEL, _DEVICE
    if _MODEL is not None:
        return _MODEL, _DEVICE

    import torch
    from modelscope.pipelines import pipeline
    from modelscope.utils.constant import Tasks

    t0 = perf_counter()
    p = pipeline(task=Tasks.speaker_verification, model=CFG.embedding_model)
    _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _MODEL = p.model
    _MODEL.eval()
    _MODEL.to(_DEVICE)
    if _DEVICE.type == "cuda":
        props = torch.cuda.get_device_properties(_DEVICE)
        total_gb = props.total_memory / (1024 ** 3)
        reserved_gb = torch.cuda.memory_reserved(_DEVICE) / (1024 ** 3)
        allocated_gb = torch.cuda.memory_allocated(_DEVICE) / (1024 ** 3)
        print(
            f"  model loaded in {perf_counter() - t0:.2f}s on cuda: "
            f"{props.name}, total={total_gb:.2f}GiB, "
            f"allocated={allocated_gb:.2f}GiB, reserved={reserved_gb:.2f}GiB",
            flush=True,
        )
    else:
        print(f"  model loaded in {perf_counter() - t0:.2f}s on cpu", flush=True)
    return _MODEL, _DEVICE


def _cuda_memory_line(torch, device) -> str:
    if device.type != "cuda":
        return ""
    allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)
    reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)
    peak_allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    peak_reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 2)
    return (
        f", cuda_alloc={allocated:.0f}MiB, cuda_reserved={reserved:.0f}MiB, "
        f"cuda_peak_alloc={peak_allocated:.0f}MiB, cuda_peak_reserved={peak_reserved:.0f}MiB"
    )


def extract_embeddings(
    wav_path: Path,
    batch_size: int | None = None,
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

        print(
            f"  audio: sr={sr}, channels={channels}, duration={total_frames / sr:.1f}s, "
            f"window={CFG.window_sec:.1f}s, hop={CFG.hop_sec:.1f}s, windows={n}",
            flush=True,
        )
        model, device = _get_model()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        batch_size = max(1, int(batch_size or CFG.embedding_batch_size))
        emb_dim = CFG.embedding_dim
        embs = np.zeros((n, emb_dim), dtype=np.float32)
        rms_threshold = 1e-3
        progress_every = max(1, n // 20)
        total_infer_sec = 0.0

        with torch.no_grad():
            for batch_start in range(0, n, batch_size):
                batch_t0 = perf_counter()
                batch_end = min(n, batch_start + batch_size)
                batch_chunks = []
                batch_indices = []
                for i in range(batch_start, batch_end):
                    s = starts[i]
                    f.seek(int(s))
                    chunk = f.read(win, dtype="float32", always_2d=False)
                    if channels > 1:
                        chunk = chunk.mean(axis=1)
                    chunk = np.ascontiguousarray(chunk, dtype=np.float32)
                    if float(np.sqrt(np.mean(chunk ** 2))) < rms_threshold:
                        print(f"  embedding {i + 1}/{n}: skipped silence", flush=True)
                        continue
                    batch_chunks.append(chunk)
                    batch_indices.append(i)

                if not batch_chunks:
                    continue

                wav_t = torch.from_numpy(np.stack(batch_chunks, axis=0)).to(device)
                try:
                    infer_t0 = perf_counter()
                    out = model(wav_t)
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    infer_sec = perf_counter() - infer_t0
                    total_infer_sec += infer_sec
                except Exception as e:
                    print(f"  !! model() failed at window {i}: {type(e).__name__}: {e}", flush=True)
                    traceback.print_exc()
                    raise
                batch_out = out.detach().cpu().numpy().astype(np.float32)
                del out, wav_t
                if batch_out.ndim == 1:
                    batch_out = batch_out.reshape(1, -1)
                if batch_out.shape[1] != emb_dim:
                    raise ValueError(
                        f"Expected embedding dim {emb_dim}, got {batch_out.shape[1]}. "
                        "Update CFG.embedding_dim or use a compatible model."
                    )
                norms = np.linalg.norm(batch_out, axis=1, keepdims=True) + 1e-9
                batch_out = batch_out / norms
                for row, i in enumerate(batch_indices):
                    embs[i] = batch_out[row]

                if batch_start % progress_every == 0 or batch_end == n:
                    elapsed = perf_counter() - batch_t0
                    rtf = infer_sec / (CFG.window_sec * len(batch_indices))
                    print(
                        f"  embedding {batch_indices[0] + 1}-{batch_indices[-1] + 1}/{n}: "
                        f"batch={len(batch_indices)}, infer={infer_sec:.2f}s, "
                        f"batch_total={elapsed:.2f}s, rtf={rtf:.2f}"
                        f"{_cuda_memory_line(torch, device)}",
                        flush=True,
                    )

        print(
            f"  embedding summary: windows={n}, total_infer={total_infer_sec:.2f}s, "
            f"avg_infer={total_infer_sec / max(1, n):.2f}s/window"
            f"{_cuda_memory_line(torch, device)}",
            flush=True,
        )

    return embs, times
