"""Segment merging, unknown bridging, length filter, transition drop."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from .config import CFG
from .matching import UNKNOWN


@dataclass
class Segment:
    start: float
    end: float
    label: str

    @property
    def duration(self) -> float:
        return self.end - self.start


def labels_to_segments(labels: List[str], times: np.ndarray) -> List[Segment]:
    """Convert per-window labels into contiguous segments using window-center times."""
    if not labels:
        return []
    # segment boundary = midpoint between consecutive window centers
    hop = CFG.hop_sec
    half = CFG.window_sec / 2
    segs: List[Segment] = []
    cur_label = labels[0]
    cur_start = float(max(0.0, times[0] - half))
    for i in range(1, len(labels)):
        if labels[i] != cur_label:
            boundary = float((times[i - 1] + times[i]) / 2)
            segs.append(Segment(cur_start, boundary, cur_label))
            cur_label = labels[i]
            cur_start = boundary
    last_end = float(times[-1] + half)
    segs.append(Segment(cur_start, last_end, cur_label))
    return segs


def bridge_short_unknowns(segs: List[Segment]) -> List[Segment]:
    """Merge unknown segments shorter than CFG.unknown_bridge_max_sec when surrounded
    by the same known host."""
    out: List[Segment] = []
    i = 0
    while i < len(segs):
        s = segs[i]
        if (
            s.label == UNKNOWN
            and s.duration < CFG.unknown_bridge_max_sec
            and out
            and i + 1 < len(segs)
            and out[-1].label == segs[i + 1].label
            and out[-1].label != UNKNOWN
        ):
            # extend previous, absorb this unknown plus the next same-host segment
            prev = out[-1]
            nxt = segs[i + 1]
            out[-1] = Segment(prev.start, nxt.end, prev.label)
            i += 2
        else:
            out.append(s)
            i += 1
    return out


def drop_unknown(segs: List[Segment]) -> List[Segment]:
    return [s for s in segs if s.label != UNKNOWN]


def filter_min_length(segs: List[Segment]) -> List[Segment]:
    return [s for s in segs if s.duration >= CFG.min_segment_sec]


def apply_transition_drop(segs: List[Segment]) -> List[Segment]:
    """Trim CFG.transition_drop_sec from both sides of every boundary between segments."""
    out: List[Segment] = []
    pad = CFG.transition_drop_sec
    n = len(segs)
    for i, s in enumerate(segs):
        start = s.start + pad if i > 0 else s.start
        end = s.end - pad if i < n - 1 else s.end
        if end - start >= CFG.min_segment_sec:
            out.append(Segment(start, end, s.label))
    return out


def finalize_segments(labels: List[str], times: np.ndarray) -> List[Segment]:
    segs = labels_to_segments(labels, times)
    segs = bridge_short_unknowns(segs)
    segs = drop_unknown(segs)
    segs = filter_min_length(segs)
    segs = apply_transition_drop(segs)
    return segs


def find_long_unknown_spans(labels: List[str], times: np.ndarray):
    """Yield (i_start, i_end, t_start, t_end) for unknown runs >= CFG.auto_enroll_min_sec."""
    n = len(labels)
    i = 0
    half = CFG.window_sec / 2
    while i < n:
        if labels[i] != UNKNOWN:
            i += 1
            continue
        j = i
        while j < n and labels[j] == UNKNOWN:
            j += 1
        t_start = float(max(0.0, times[i] - half))
        t_end = float(times[j - 1] + half)
        if (t_end - t_start) >= CFG.auto_enroll_min_sec:
            yield i, j, t_start, t_end
        i = j
