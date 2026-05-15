"""End-to-end pipeline CLI."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .config import CFG
from .preprocess import prepare_audio
from .embedding import extract_embeddings
from .db import HostDB
from .matching import match_embeddings, smooth_labels
from .enrollment import auto_enroll_from_track
from .segmenting import (
    finalize_segments,
    labels_to_segments,
    bridge_short_unknowns,
)
from .cutting import cut_segments


def log(msg: str):
    print(f"[pipeline] {msg}", flush=True)


def run(video_path: Path, work_dir: Path, db_dir: Path, output_dir: Path,
        dry_run: bool, demucs: bool, max_windows: int | None = None) -> int:
    log(f"video: {video_path}")
    log("step 1/6: preprocess audio")
    audio_path = prepare_audio(video_path, work_dir, do_demucs=demucs)

    log("step 2/6: extract embeddings")
    cache = work_dir / f"{video_path.stem}.embs.npz"
    if cache.exists():
        d = np.load(cache)
        embs, times = d["embs"], d["times"]
        log(f"  loaded cache: {embs.shape[0]} windows")
    else:
        embs, times = extract_embeddings(audio_path, max_windows=max_windows)
        np.savez(cache, embs=embs, times=times)
        log(f"  extracted {embs.shape[0]} windows")

    if embs.shape[0] == 0:
        log("no audio windows; aborting")
        return 1

    db = HostDB(db_dir)
    log(f"step 3/6: pass-1 match against {len(db.list_ids())} known hosts")
    labels1, _ = match_embeddings(embs, db)
    labels1 = smooth_labels(labels1)

    log("step 4/6: auto-enroll long unknown spans")
    touched = auto_enroll_from_track(embs, labels1, times, db)
    if touched:
        log(f"  registered/updated: {touched}")
    else:
        log("  no new hosts enrolled")

    log(f"step 5/6: pass-2 match against {len(db.list_ids())} hosts")
    labels2, _ = match_embeddings(embs, db)
    labels2 = smooth_labels(labels2)

    raw_segs = bridge_short_unknowns(labels_to_segments(labels2, times))
    log(f"  raw segments (pre-filter): {len(raw_segs)}")
    for s in raw_segs:
        log(f"    [raw] {s.label}: {s.start:.1f} -> {s.end:.1f} ({s.duration/60:.1f} min)")

    segments = finalize_segments(labels2, times)
    log(f"  final segments (>=1h, transition-trimmed): {len(segments)}")
    for s in segments:
        log(f"    {s.label}: {s.start:.1f} -> {s.end:.1f} ({s.duration/60:.1f} min)")

    summary = {
        "video": str(video_path),
        "raw_segments": [
            {"start": s.start, "end": s.end, "label": s.label, "duration_sec": s.duration}
            for s in raw_segs
        ],
        "segments": [
            {"start": s.start, "end": s.end, "label": s.label, "duration_sec": s.duration}
            for s in segments
        ],
    }
    (work_dir / f"{video_path.stem}.segments.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    if dry_run:
        log("dry-run: skipping cut and DB persistence (DB writes already happened during enroll)")
        return 0

    log("step 6/6: cut video")
    written = cut_segments(video_path, segments, output_dir)
    for w in written:
        log(f"  wrote {w}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Split livestream recording by speaker.")
    p.add_argument("video", type=Path)
    p.add_argument("--work-dir", type=Path, default=Path("work"))
    p.add_argument("--db-dir", type=Path, default=Path("host_db"))
    p.add_argument("--output-dir", type=Path, default=Path("output"))
    p.add_argument("--dry-run", action="store_true",
                   help="Predict only; do not cut video.")
    p.add_argument("--demucs", action="store_true",
                   help="Enable Demucs vocal separation (slower; useful when BGM is heavy).")
    p.add_argument("--max-windows", type=int, default=None,
                   help="Debug only: process at most this many embedding windows.")
    args = p.parse_args(argv)

    args.work_dir.mkdir(parents=True, exist_ok=True)
    return run(args.video, args.work_dir, args.db_dir, args.output_dir,
               args.dry_run, args.demucs, args.max_windows)


if __name__ == "__main__":
    sys.exit(main())
