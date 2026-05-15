"""Speaker voiceprint database."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import CFG


@dataclass
class HostEntry:
    host_id: str
    centroid_path: str
    registered_at: float
    samples_count: int = 0
    total_duration_sec: float = 0.0


class HostDB:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "hosts.json"
        self.hosts: Dict[str, HostEntry] = {}
        self._centroids: Dict[str, np.ndarray] = {}
        self._load()

    def _load(self):
        if not self.index_path.exists():
            return
        data = json.loads(self.index_path.read_text(encoding="utf-8"))
        for hid, rec in data.items():
            self.hosts[hid] = HostEntry(**rec)
            cpath = self.root / rec["centroid_path"]
            if cpath.exists():
                self._centroids[hid] = np.load(cpath)

    def _save_index(self):
        data = {hid: asdict(e) for hid, e in self.hosts.items()}
        self.index_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def list_ids(self) -> List[str]:
        return list(self.hosts.keys())

    def centroid_matrix(self) -> Tuple[List[str], np.ndarray]:
        """Return (ids, M) where M is (K, D) stacked centroids. Empty if no hosts."""
        ids = list(self._centroids.keys())
        if not ids:
            return [], np.zeros((0, 192), dtype=np.float32)
        M = np.stack([self._centroids[i] for i in ids], axis=0)
        return ids, M

    def _next_id(self) -> str:
        n = len(self.hosts) + 1
        while True:
            candidate = f"host_{n:03d}"
            if candidate not in self.hosts:
                return candidate
            n += 1

    def add_host(self, centroid: np.ndarray, duration_sec: float,
                 sample_emb: Optional[np.ndarray] = None) -> str:
        """Register a brand new host."""
        hid = self._next_id()
        host_dir = self.root / hid
        host_dir.mkdir(parents=True, exist_ok=True)
        samples_dir = host_dir / "samples"
        samples_dir.mkdir(exist_ok=True)

        c = centroid.astype(np.float32)
        c = c / (np.linalg.norm(c) + 1e-9)
        np.save(host_dir / "centroid.npy", c)
        if sample_emb is not None:
            stamp = int(time.time())
            np.save(samples_dir / f"sample_{stamp}.npy", sample_emb.astype(np.float32))

        entry = HostEntry(
            host_id=hid,
            centroid_path=f"{hid}/centroid.npy",
            registered_at=time.time(),
            samples_count=1 if sample_emb is not None else 0,
            total_duration_sec=duration_sec,
        )
        self.hosts[hid] = entry
        self._centroids[hid] = c
        self._save_index()
        return hid

    def update_host(self, hid: str, sample_centroid: np.ndarray, duration_sec: float):
        """EMA-update the centroid of an existing host."""
        if hid not in self.hosts:
            raise KeyError(hid)
        old = self._centroids[hid]
        new = sample_centroid.astype(np.float32)
        new = new / (np.linalg.norm(new) + 1e-9)
        merged = CFG.centroid_ema_old * old + CFG.centroid_ema_new * new
        merged = merged / (np.linalg.norm(merged) + 1e-9)

        host_dir = self.root / hid
        np.save(host_dir / "centroid.npy", merged)
        stamp = int(time.time())
        np.save(host_dir / "samples" / f"sample_{stamp}.npy", new)

        self._centroids[hid] = merged
        entry = self.hosts[hid]
        entry.samples_count += 1
        entry.total_duration_sec += duration_sec
        self._save_index()

    def find_best_match(self, centroid: np.ndarray) -> Tuple[Optional[str], float]:
        ids, M = self.centroid_matrix()
        if not ids:
            return None, 0.0
        c = centroid / (np.linalg.norm(centroid) + 1e-9)
        sims = M @ c
        idx = int(np.argmax(sims))
        return ids[idx], float(sims[idx])
