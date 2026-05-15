"""Match per-window embeddings against the host DB and smooth the label track."""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
from scipy.signal import medfilt

from .config import CFG
from .db import HostDB

UNKNOWN = "unknown"


def match_embeddings(embs: np.ndarray, db: HostDB) -> Tuple[List[str], np.ndarray]:
    """Per-window match. Returns (labels, max_sims)."""
    ids, M = db.centroid_matrix()
    n = embs.shape[0]
    labels: List[str] = []
    max_sims = np.zeros(n, dtype=np.float32)

    if not ids:
        return [UNKNOWN] * n, max_sims

    # rows of embs are already L2-normalized (or zeroed for silence)
    norms = np.linalg.norm(embs, axis=1)
    sims = embs @ M.T  # (N, K)

    for i in range(n):
        if norms[i] < 1e-6:
            labels.append(UNKNOWN)
            continue
        row = sims[i]
        order = np.argsort(row)[::-1]
        top1 = row[order[0]]
        top2 = row[order[1]] if len(order) > 1 else -1.0
        max_sims[i] = top1
        if top1 > CFG.sim_threshold and (top1 - top2) > CFG.margin_threshold:
            labels.append(ids[order[0]])
        else:
            labels.append(UNKNOWN)
    return labels, max_sims


def smooth_labels(labels: List[str]) -> List[str]:
    """Median-filter the label track over a window of CFG.smooth_window_sec seconds.

    Hop is 1s so window size in samples == seconds.
    """
    if not labels:
        return labels
    unique = sorted(set(labels))
    if len(unique) == 1:
        return list(labels)
    mapping = {lab: i for i, lab in enumerate(unique)}
    inv = {i: lab for lab, i in mapping.items()}
    arr = np.array([mapping[l] for l in labels], dtype=np.int32)

    k = max(1, int(round(CFG.smooth_window_sec / CFG.hop_sec)))
    if k % 2 == 0:
        k += 1
    if k <= 1 or k > len(arr):
        return list(labels)

    # do a per-class majority vote rather than median (median doesn't make sense on labels)
    K = len(unique)
    half = k // 2
    padded = np.pad(arr, (half, half), mode="edge")
    out = np.empty_like(arr)
    for i in range(len(arr)):
        window = padded[i:i + k]
        counts = np.bincount(window, minlength=K)
        out[i] = int(np.argmax(counts))
    return [inv[int(x)] for x in out]
