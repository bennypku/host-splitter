"""Repair interrupted livestream recordings by losslessly merging adjacent files."""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path


VIDEO_EXTS = {".ts", ".mp4", ".flv", ".mkv", ".mov"}
TIMESTAMP_RE = re.compile(r"^(?P<account>.+?)(?P<stamp>\d{14})$")
MAX_GAP = timedelta(minutes=30)
LIVE_DAY_CUTOFF = time(3, 0, 0)
MIN_CONCAT_DURATION_SEC = 6.0


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


@dataclass(frozen=True)
class Recording:
    path: Path
    account: str
    start: datetime
    duration: float

    @property
    def end(self) -> datetime:
        return self.start + timedelta(seconds=self.duration)

    @property
    def live_day(self) -> date:
        if self.start.time() < LIVE_DAY_CUTOFF:
            return self.start.date() - timedelta(days=1)
        return self.start.date()


def log(message: str) -> None:
    print(f"[merge] {message}", flush=True)


def parse_recording_name(path: Path) -> tuple[str, datetime] | None:
    if path.stem.endswith("_merge"):
        return None
    match = TIMESTAMP_RE.match(path.stem)
    if not match:
        return None
    stamp = match.group("stamp")
    try:
        start = datetime.strptime(stamp, "%Y%m%d%H%M%S")
    except ValueError:
        return None
    return match.group("account"), start


def probe_duration(path: Path) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or f"ffprobe failed for {path}")
    data = json.loads(proc.stdout)
    duration = float(data["format"]["duration"])
    if duration <= 0:
        raise RuntimeError(f"non-positive duration for {path}")
    return duration


def discover_recordings(folder: Path, today: date) -> list[Recording]:
    recordings: list[Recording] = []
    skipped_today = 0
    skipped_name = 0
    skipped_ext = 0

    for path in sorted(folder.iterdir(), key=lambda p: p.name):
        if not path.is_file():
            continue
        if path.suffix.lower() not in VIDEO_EXTS:
            skipped_ext += 1
            continue
        parsed = parse_recording_name(path)
        if parsed is None:
            skipped_name += 1
            continue
        account, start = parsed
        if start.date() == today:
            skipped_today += 1
            continue
        duration = probe_duration(path)
        recordings.append(Recording(path=path, account=account, start=start, duration=duration))

    log(f"loaded {len(recordings)} recordings")
    if skipped_today:
        log(f"skipped today's recordings: {skipped_today}")
    if skipped_name:
        log(f"skipped unmatched names: {skipped_name}")
    if skipped_ext:
        log(f"skipped unsupported extensions: {skipped_ext}")
    return sorted(recordings, key=lambda r: r.path.name)


def same_group(prev: Recording, cur: Recording) -> bool:
    if prev.account != cur.account:
        return False
    if prev.live_day != cur.live_day:
        return False
    gap = cur.start - prev.end
    return gap < MAX_GAP


def group_recordings(recordings: list[Recording]) -> list[list[Recording]]:
    groups: list[list[Recording]] = []
    current: list[Recording] = []
    for rec in recordings:
        if not current:
            current = [rec]
            continue
        if same_group(current[-1], rec):
            current.append(rec)
        else:
            groups.append(current)
            current = [rec]
    if current:
        groups.append(current)
    return [group for group in groups if len(group) > 1]


def concat_list_line(path: Path) -> str:
    escaped = str(path.resolve()).replace("'", "'\\''")
    return f"file '{escaped}'\n"


def output_path_for(group: list[Recording], folder: Path) -> Path:
    first = group[0].path
    return folder / f"{first.stem}_merge{first.suffix}"


def merge_group(group: list[Recording], folder: Path, archive_dir: Path, dry_run: bool) -> Path:
    output = output_path_for(group, folder)
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")

    names = ", ".join(rec.path.name for rec in group)
    log(f"group: {names} -> {output.name}")
    concat_inputs = [rec for rec in group if rec.duration >= MIN_CONCAT_DURATION_SEC]
    dropped = [rec for rec in group if rec.duration < MIN_CONCAT_DURATION_SEC]
    if dropped:
        log("  drop from concat (<6s): " + ", ".join(rec.path.name for rec in dropped))
    if not concat_inputs:
        raise RuntimeError(f"all recordings in group are shorter than {MIN_CONCAT_DURATION_SEC:g}s")
    if len(concat_inputs) == 1:
        raise RuntimeError(
            "only one recording remains after dropping <6s inputs; "
            f"refusing to create merge output: {concat_inputs[0].path.name}"
        )
    if dry_run:
        return output

    with tempfile.TemporaryDirectory(prefix="merge_recordings_") as tmp:
        list_path = Path(tmp) / "concat.txt"
        list_path.write_text("".join(concat_list_line(rec.path) for rec in concat_inputs), encoding="utf-8")
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_path),
            "-c",
            "copy",
            str(output),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if proc.returncode != 0:
            if output.exists():
                output.unlink()
            raise RuntimeError(proc.stderr.strip() or f"ffmpeg failed for {output}")

    if not output.exists() or output.stat().st_size <= 0:
        raise RuntimeError(f"merge output missing or empty: {output}")

    archive_dir.mkdir(parents=True, exist_ok=True)
    for rec in group:
        dest = archive_dir / rec.path.name
        if dest.exists():
            raise FileExistsError(f"archive target already exists: {dest}")
        shutil.move(str(rec.path), str(dest))
    return output


def run(folder: Path, archive_name: str, today: date, dry_run: bool) -> int:
    folder = folder.resolve()
    if not folder.is_dir():
        raise NotADirectoryError(folder)

    recordings = discover_recordings(folder, today)
    groups = group_recordings(recordings)
    if not groups:
        log("no merge groups found")
        return 0

    archive_dir = folder / archive_name
    log(f"merge groups found: {len(groups)}")
    for group in groups:
        merge_group(group, folder, archive_dir, dry_run)

    if dry_run:
        log("dry-run complete; no files were merged or moved")
    else:
        log("done")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Losslessly merge interrupted livestream recordings.")
    parser.add_argument("folder", type=Path, help="素材文件夹")
    parser.add_argument("--archive-name", default="_merged_sources", help="归档原始素材的子文件夹名称")
    parser.add_argument("--today", type=lambda s: datetime.strptime(s, "%Y%m%d").date(),
                        default=date.today(), help="按 yyyyMMdd 指定今天日期，默认使用系统日期")
    parser.add_argument("--dry-run", action="store_true", help="只打印合并计划，不生成或移动文件")
    args = parser.parse_args(argv)
    return run(args.folder, args.archive_name, args.today, args.dry_run)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"[merge] ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
