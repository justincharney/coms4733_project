#!/usr/bin/env python3
"""
Add rolling-mean scalar series to existing TensorBoard runs.

This does NOT require retraining. It reads existing `events.out.tfevents*` files
and writes a new event file (in the same run directory) containing additional
scalar tags such as:

  - Episode/total_reward_ma10
  - Episode/grasp_efficiency_ma10
  - Episode/length_ma10
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Sequence
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing `tensorboard` dependency. Install requirements and retry."
    ) from exc


DEFAULT_TAGS: tuple[str, ...] = (
    "Episode/total_reward",
    "Episode/grasp_efficiency",
    "Episode/length",
)


def rolling_mean(values: Sequence[float], window: int) -> list[float]:
    if window <= 0:
        raise ValueError("window must be > 0")
    out: list[float] = [0.0] * len(values)
    running_sum = 0.0
    for idx, value in enumerate(values):
        running_sum += float(value)
        if idx >= window:
            running_sum -= float(values[idx - window])
            out[idx] = running_sum / window
        else:
            out[idx] = running_sum / (idx + 1)
    return out


def find_run_dirs(logdir: Path) -> list[Path]:
    run_dirs = [p for p in logdir.iterdir() if p.is_dir()]
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found under {logdir}")
    return run_dirs


def find_latest_run_dir(logdir: Path) -> Path:
    return max(find_run_dirs(logdir), key=lambda p: p.stat().st_mtime)


def load_scalar_series(run_dir: Path, tag: str) -> tuple[list[int], list[float], list[float]]:
    acc = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    acc.Reload()
    if tag not in acc.Tags().get("scalars", []):
        return [], [], []
    events = acc.Scalars(tag)
    steps = [int(e.step) for e in events]
    values = [float(e.value) for e in events]
    wall_times = [float(e.wall_time) for e in events]
    return steps, values, wall_times


def add_rolling_means(
    run_dir: Path,
    tags: Iterable[str],
    windows: Sequence[int],
    *,
    suffix_template: str = "_ma{window}",
    writer_suffix: str = "_postprocess",
    overwrite: bool = False,
) -> None:
    run_dir = run_dir.resolve()
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)
    if not windows:
        raise ValueError("windows must be non-empty")

    acc = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    acc.Reload()
    existing_tags = set(acc.Tags().get("scalars", []))

    writer = SummaryWriter(log_dir=str(run_dir), filename_suffix=writer_suffix)
    try:
        for tag in tags:
            steps, values, wall_times = load_scalar_series(run_dir, tag)
            if not steps:
                continue

            for window in windows:
                out_tag = f"{tag}{suffix_template.format(window=window)}"
                if not overwrite and out_tag in existing_tags:
                    continue

                smoothed = rolling_mean(values, window=window)
                for step, val, wall_time in zip(steps, smoothed, wall_times):
                    writer.add_scalar(out_tag, val, global_step=step, walltime=wall_time)
    finally:
        writer.flush()
        writer.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add rolling-mean scalar series to TensorBoard runs without retraining."
    )
    parser.add_argument("--logdir", type=Path, default=Path("runs"), help="TensorBoard log root (default: runs/)")
    parser.add_argument("--run", type=Path, default=None, help="Specific run directory under --logdir (default: latest)")
    parser.add_argument("--all-runs", action="store_true", help="Process all run directories under --logdir.")
    parser.add_argument(
        "--window",
        type=int,
        nargs="+",
        default=[10],
        help="Rolling window size(s), e.g. `--window 10 50` (default: 10).",
    )
    parser.add_argument(
        "--tags",
        nargs="*",
        default=list(DEFAULT_TAGS),
        help="Scalar tags to smooth (default: episode reward/length/efficiency).",
    )
    parser.add_argument(
        "--suffix-template",
        type=str,
        default="_ma{window}",
        help="Suffix template appended to output tags (default: _ma{window}).",
    )
    parser.add_argument(
        "--writer-suffix",
        type=str,
        default="_postprocess",
        help="Filename suffix for the new event file (default: _postprocess).",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing smoothed tags if present.")
    args = parser.parse_args()

    logdir = args.logdir
    if not logdir.exists():
        raise FileNotFoundError(logdir)

    if args.all_runs:
        run_dirs = sorted(find_run_dirs(logdir))
    else:
        run_dirs = [(logdir / args.run) if args.run is not None else find_latest_run_dir(logdir)]

    for run_dir in run_dirs:
        add_rolling_means(
            run_dir=run_dir,
            tags=args.tags,
            windows=args.window,
            suffix_template=args.suffix_template,
            writer_suffix=args.writer_suffix,
            overwrite=args.overwrite,
        )
        print(f"Wrote rolling-mean scalars into {run_dir}")


if __name__ == "__main__":
    main()

