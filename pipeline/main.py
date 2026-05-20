"""End-to-end pipeline CLI."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from time import perf_counter

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


VIDEO_EXTS = {".ts", ".mp4", ".flv", ".mkv", ".mov", ".avi"}


def log(msg: str):
    print(f"[pipeline] {msg}", flush=True)


def run(video_path: Path, work_dir: Path, db_dir: Path, output_dir: Path,
        dry_run: bool, demucs: bool, max_windows: int | None = None,
        batch_size: int | None = None) -> int:
    log(f"video: {video_path}")
    run_t0 = perf_counter()
    log("step 1/6: preprocess audio")
    step_t0 = perf_counter()
    audio_path = prepare_audio(video_path, work_dir, do_demucs=demucs)
    log(f"  audio ready: {audio_path} ({perf_counter() - step_t0:.2f}s)")

    log("step 2/6: extract embeddings")
    step_t0 = perf_counter()
    cache = work_dir / f"{video_path.stem}.embs.npz"
    if cache.exists():
        d = np.load(cache)
        embs, times = d["embs"], d["times"]
        log(f"  loaded cache: {embs.shape[0]} windows ({perf_counter() - step_t0:.2f}s)")
    else:
        embs, times = extract_embeddings(
            audio_path,
            batch_size=batch_size,
            max_windows=max_windows,
        )
        np.savez(cache, embs=embs, times=times)
        log(f"  extracted {embs.shape[0]} windows ({perf_counter() - step_t0:.2f}s)")

    if embs.shape[0] == 0:
        log("no audio windows; aborting")
        return 1

    db = HostDB(db_dir)
    log(f"step 3/6: pass-1 match against {len(db.list_ids())} known hosts")
    step_t0 = perf_counter()
    labels1, _ = match_embeddings(embs, db)
    labels1 = smooth_labels(labels1)
    log(f"  pass-1 done ({perf_counter() - step_t0:.2f}s)")

    log("step 4/6: auto-enroll long unknown spans")
    step_t0 = perf_counter()
    touched = auto_enroll_from_track(embs, labels1, times, db)
    if touched:
        log(f"  registered/updated: {touched}")
    else:
        log("  no new hosts enrolled")
    log(f"  auto-enroll done ({perf_counter() - step_t0:.2f}s)")

    log(f"step 5/6: pass-2 match against {len(db.list_ids())} hosts")
    step_t0 = perf_counter()
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
    log(f"  pass-2 and segmenting done ({perf_counter() - step_t0:.2f}s)")

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
        log(f"done in {perf_counter() - run_t0:.2f}s")
        return 0

    log("step 6/6: cut video")
    step_t0 = perf_counter()
    written = cut_segments(video_path, segments, output_dir)
    for w in written:
        log(f"  wrote {w}")
    log(f"  cut done ({perf_counter() - step_t0:.2f}s)")
    log(f"done in {perf_counter() - run_t0:.2f}s")
    return 0


def discover_videos(folder: Path) -> list[Path]:
    videos: list[Path] = []
    skip_dirs = {".host_meta"}
    for path in sorted(folder.iterdir(), key=lambda p: p.name):
        if path.is_dir() and path.name in skip_dirs:
            continue
        if not path.is_file():
            continue
        if path.suffix.lower() not in VIDEO_EXTS:
            continue
        videos.append(path)
    return videos


def archive_processed(video_path: Path, processed_dir: Path) -> Path:
    processed_dir.mkdir(parents=True, exist_ok=True)
    dest = processed_dir / video_path.name
    if dest.exists():
        raise FileExistsError(f"processed archive already exists: {dest}")
    shutil.move(str(video_path), str(dest))
    return dest


def run_folder(folder: Path, dry_run: bool, demucs: bool,
               max_windows: int | None = None,
               batch_size: int | None = None) -> int:
    folder = folder.resolve()
    if not folder.is_dir():
        raise NotADirectoryError(folder)

    meta_dir = folder / ".host_meta"
    work_root = meta_dir / "work"
    db_dir = meta_dir / "host_db"
    processed_dir = folder / "host-splitted"
    output_dir = folder
    meta_dir.mkdir(parents=True, exist_ok=True)
    work_root.mkdir(parents=True, exist_ok=True)

    videos = discover_videos(folder)
    log(f"input dir: {folder}")
    log(f"videos found: {len(videos)}")
    if not videos:
        return 0

    failed = 0
    for video in videos:
        log(f"batch item: {video.name}")
        item_work = work_root / video.stem
        item_work.mkdir(parents=True, exist_ok=True)
        try:
            rc = run(video, item_work, db_dir, output_dir, dry_run, demucs, max_windows, batch_size)
            if rc != 0:
                failed += 1
                log(f"  failed with exit code {rc}; source stays in place")
                continue
            if dry_run:
                log("  dry-run: source stays in place")
                continue
            archived = archive_processed(video, processed_dir)
            log(f"  moved source to {archived}")
        except Exception as exc:
            failed += 1
            log(f"  ERROR: {exc}; source stays in place")

    log(f"batch done: success={len(videos) - failed}, failed={failed}")
    return 1 if failed else 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Split livestream recording by speaker.")
    p.add_argument("video", type=Path, nargs="?", help="Single video file to process.")
    p.add_argument("--input-dir", type=Path,
                   help="Process all livestream recordings in this folder.")
    p.add_argument("--work-dir", type=Path, default=Path("work"))
    p.add_argument("--db-dir", type=Path, default=Path("host_db"))
    p.add_argument("--output-dir", type=Path, default=Path("output"))
    p.add_argument("--dry-run", action="store_true",
                   help="Predict only; do not cut video.")
    p.add_argument("--demucs", action="store_true",
                   help="Enable Demucs vocal separation (slower; useful when BGM is heavy).")
    p.add_argument("--max-windows", type=int, default=None,
                   help="Debug only: process at most this many embedding windows.")
    p.add_argument("--batch-size", type=int, default=None,
                   help="Embedding batch size. Defaults to CFG.embedding_batch_size.")
    args = p.parse_args(argv)

    if args.input_dir is not None:
        return run_folder(args.input_dir, args.dry_run, args.demucs, args.max_windows, args.batch_size)
    if args.video is None:
        p.error("video is required unless --input-dir is used")

    args.work_dir.mkdir(parents=True, exist_ok=True)
    return run(args.video, args.work_dir, args.db_dir, args.output_dir,
               args.dry_run, args.demucs, args.max_windows, args.batch_size)


if __name__ == "__main__":
    sys.exit(main())
