"""Keyframe-aligned lossless video cutting."""
from __future__ import annotations

import json
import re
import subprocess
from bisect import bisect_right
from pathlib import Path
from typing import List

from .segmenting import Segment


_HOST_SUFFIX_RE = re.compile(r"(\d+)$")


def get_keyframe_times(video_path: Path) -> List[float]:
    """Return ascending list of I-frame PTS times (seconds)."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-skip_frame", "nokey",
        "-show_entries", "frame=pts_time",
        "-of", "json",
        str(video_path),
    ]
    out = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
    data = json.loads(out)
    times = []
    for f in data.get("frames", []):
        t = f.get("pts_time")
        if t is None:
            continue
        try:
            times.append(float(t))
        except ValueError:
            pass
    times.sort()
    return times


def align_to_prev_keyframe(t: float, keyframes: List[float]) -> float:
    if not keyframes:
        return t
    idx = bisect_right(keyframes, t) - 1
    if idx < 0:
        return keyframes[0]
    return keyframes[idx]


def host_output_name(host_id: str) -> str:
    """Map internal host ids like host_001 to output dirs like host01."""
    m = _HOST_SUFFIX_RE.search(host_id)
    if not m:
        return host_id
    return f"host{int(m.group(1)):02d}"


def format_duration_tag(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}h{m:02d}m{s:02d}s"


def cut_segment(video_path: Path, start: float, end: float, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start:.3f}",
        "-to", f"{end:.3f}",
        "-i", str(video_path),
        "-c", "copy",
        "-avoid_negative_ts", "make_zero",
        str(out_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def cut_segments(
    video_path: Path,
    segments: List[Segment],
    output_root: Path,
) -> List[Path]:
    """Cut all segments. End times stay as decoded (no realignment) — only starts snap
    backwards to the nearest keyframe to keep -c copy clean."""
    keyframes = get_keyframe_times(video_path)
    written: List[Path] = []
    stem = video_path.stem
    for global_idx, seg in enumerate(segments, start=1):
        start_kf = align_to_prev_keyframe(seg.start, keyframes)
        host_dir = output_root / host_output_name(seg.label)
        start_tag = format_duration_tag(seg.start)
        end_tag = format_duration_tag(seg.end)
        out_path = host_dir / f"{stem}_{start_tag}_{end_tag}_part{global_idx:02d}{video_path.suffix}"
        cut_segment(video_path, start_kf, seg.end, out_path)
        written.append(out_path)
    return written
