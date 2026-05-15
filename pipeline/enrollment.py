"""Auto-enroll new hosts from long unknown spans via clustering."""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
from sklearn.cluster import AgglomerativeClustering

from .config import CFG
from .db import HostDB
from .segmenting import find_long_unknown_spans


def _cluster_all(embs: np.ndarray):
    """Cluster window embeddings; return list of (centroid, size, fraction) per cluster, descending."""
    norms = np.linalg.norm(embs, axis=1)
    valid = embs[norms > 1e-6]
    if len(valid) < 2:
        c = embs.mean(axis=0)
        return [(c / (np.linalg.norm(c) + 1e-9), len(embs), 1.0)]

    clusterer = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=CFG.enroll_cluster_distance,
        metric="cosine",
        linkage="average",
    )
    labels = clusterer.fit_predict(valid)
    out = []
    total = len(labels)
    for cid in range(int(labels.max()) + 1):
        mask = labels == cid
        size = int(mask.sum())
        if size == 0:
            continue
        c = valid[mask].mean(axis=0)
        c = c / (np.linalg.norm(c) + 1e-9)
        out.append((c, size, size / total))
    out.sort(key=lambda x: -x[1])
    return out


def auto_enroll_from_track(
    embs: np.ndarray,
    labels: List[str],
    times: np.ndarray,
    db: HostDB,
) -> List[str]:
    """Scan smoothed label track for long unknown spans and enroll dominant clusters.

    Strategy: for each unknown span >= auto_enroll_min_sec, cluster embeddings; each
    cluster whose share of the span is >= min_segment_sec gets registered (or merged
    into an existing host if cosine similarity is high enough).
    """
    touched: List[str] = []
    for i_start, i_end, t_start, t_end in find_long_unknown_spans(labels, times):
        span_embs = embs[i_start:i_end]
        duration = t_end - t_start
        clusters = _cluster_all(span_embs)
        print(f"  enroll-debug: span {t_start:.0f}-{t_end:.0f}s "
              f"({duration/60:.1f}min, {len(span_embs)} windows), "
              f"clusters={[(s, f'{frac:.2f}') for _, s, frac in clusters[:5]]}",
              flush=True)
        for centroid, size, frac in clusters:
            cluster_duration = frac * duration
            if cluster_duration < CFG.min_segment_sec:
                continue
            existing_id, sim = db.find_best_match(centroid)
            if existing_id is not None and sim > CFG.merge_existing_threshold:
                db.update_host(existing_id, centroid, cluster_duration)
                touched.append(existing_id)
                print(f"    -> merged into {existing_id} (sim={sim:.3f}, "
                      f"share={frac:.2f}, ~{cluster_duration/60:.0f}min)", flush=True)
            else:
                new_id = db.add_host(centroid, cluster_duration, sample_emb=centroid)
                touched.append(new_id)
                print(f"    -> registered {new_id} (share={frac:.2f}, "
                      f"~{cluster_duration/60:.0f}min)", flush=True)
    return touched
