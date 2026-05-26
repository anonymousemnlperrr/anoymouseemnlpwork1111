#!/usr/bin/env python3
"""
预缓存 VTLA RGB 帧, 避免训练时每个 sample 都打开 mp4 seek/decode。

示例:
  # 单个 task
  python -m VTLA.data_processing.cache_rgb_frames \
    --data_root data/inboxpicking-01

  # 多个 task
  python -m VTLA.data_processing.cache_rgb_frames \
    --data_root data/inboxpicking-* \
               data/grasp-blueberry-*
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
from PIL import Image

try:
    import av
except ImportError:
    av = None

from VTLA.dataset.vtla_dataset import SRC_FPS


CAMERAS = [
    "observation.images.realsense_rgb",
    "observation.images.depth",
    "observation.images.side",
]


def discover_task_roots(patterns: list[str]) -> list[Path]:
    roots: list[Path] = []
    for pattern in patterns:
        path = Path(pattern)
        if path.is_dir():
            roots.append(path)
            continue
        roots.extend(sorted(p for p in Path(".").glob(pattern) if p.is_dir()))
    return sorted(set(roots))


def load_sampled_frame_requests(data_root: Path, target_fps: int) -> dict[int, set[int]]:
    data_files = sorted((data_root / "data").glob("chunk-*/*.parquet"))
    if not data_files:
        data_files = sorted((data_root / "data").glob("*.parquet"))
    if not data_files:
        raise FileNotFoundError(f"No parquet files in {data_root / 'data'}")

    frames_df = pd.concat([pd.read_parquet(path) for path in data_files], ignore_index=True)
    stride = max(1, round(SRC_FPS / target_fps))
    sampled = frames_df.iloc[list(range(0, len(frames_df), stride))]

    requests: dict[int, set[int]] = defaultdict(set)
    for row in sampled.itertuples(index=False):
        requests[int(row.episode_index)].add(int(row.frame_index))
    return requests


def discover_camera_videos(data_root: Path, camera_name: str) -> dict[int, Path]:
    cam_dir = data_root / "videos" / camera_name
    if not cam_dir.exists():
        return {}

    videos: dict[int, Path] = {}
    for mp4 in sorted(cam_dir.rglob("*.mp4")):
        stem = mp4.stem
        try:
            ep_offset = int(stem.split("-")[-1])
        except ValueError:
            ep_offset = 0
        videos[ep_offset] = mp4
    return videos


def resolve_video_path(video_map: dict[int, Path], episode_idx: int) -> Path | None:
    if episode_idx in video_map:
        return video_map[episode_idx]
    if 0 in video_map:
        return video_map[0]
    return None


def cache_frame_path(cache_root: Path, camera_name: str, episode_idx: int, frame_idx: int) -> Path:
    return cache_root / camera_name / f"episode-{episode_idx:06d}" / f"frame-{frame_idx:06d}.jpg"


def prepare_image(image: Image.Image, rgb_size: tuple[int, int]) -> Image.Image:
    if image.size != (rgb_size[1], rgb_size[0]):
        image = image.resize((rgb_size[1], rgb_size[0]), Image.BILINEAR)
    return image.convert("RGB")


def extract_video_targets(
    mp4_path: Path,
    frame_targets: dict[int, list[Path]],
    rgb_size: tuple[int, int],
) -> tuple[int, int]:
    if av is None:
        raise RuntimeError("PyAV is not installed; cannot pre-cache RGB frames")

    target_indices = set(frame_targets)
    if not target_indices:
        return 0, 0

    written = 0
    decoded = 0
    max_target = max(target_indices)

    with av.open(str(mp4_path)) as container:
        for frame_idx, frame in enumerate(container.decode(video=0)):
            decoded += 1
            if frame_idx > max_target and not target_indices:
                break
            if frame_idx not in target_indices:
                continue

            image = prepare_image(frame.to_image(), rgb_size)
            for output_path in frame_targets[frame_idx]:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                image.save(output_path, format="JPEG", quality=90)
                written += 1
            target_indices.remove(frame_idx)
            if not target_indices:
                break

    return written, decoded


def cache_task(
    data_root: Path,
    target_fps: int,
    cache_dir_name: str,
    rgb_size: tuple[int, int],
    overwrite: bool,
) -> None:
    frame_requests = load_sampled_frame_requests(data_root, target_fps)
    cache_root = data_root / cache_dir_name
    print(f"[{data_root.name}] samples={sum(len(v) for v in frame_requests.values())} cache={cache_root}")

    for camera_name in CAMERAS:
        video_map = discover_camera_videos(data_root, camera_name)
        if not video_map:
            print(f"  - {camera_name}: skip (no video dir)")
            continue

        targets_by_video: dict[Path, dict[int, list[Path]]] = defaultdict(lambda: defaultdict(list))
        for episode_idx, frame_ids in frame_requests.items():
            video_path = resolve_video_path(video_map, episode_idx)
            if video_path is None:
                continue
            for frame_idx in sorted(frame_ids):
                output_path = cache_frame_path(cache_root, camera_name, episode_idx, frame_idx)
                if output_path.exists() and not overwrite:
                    continue
                targets_by_video[video_path][frame_idx].append(output_path)

        if not targets_by_video:
            print(f"  - {camera_name}: already cached")
            continue

        camera_written = 0
        for video_path, frame_targets in sorted(targets_by_video.items()):
            written, _ = extract_video_targets(video_path, frame_targets, rgb_size)
            camera_written += written
        print(f"  - {camera_name}: wrote {camera_written} cached frames")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-cache VTLA RGB frames for Stage B / eval")
    parser.add_argument("--data_root", nargs="+", required=True, help="Task root(s), supports shell-expanded globs")
    parser.add_argument("--target_fps", type=int, default=10, help="Sampling FPS used by the dataset")
    parser.add_argument("--cache_dir_name", type=str, default="rgb_cache_480x640", help="Cache dir name under each task root")
    parser.add_argument("--rgb_height", type=int, default=480, help="Cached frame height")
    parser.add_argument("--rgb_width", type=int, default=640, help="Cached frame width")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing cached frames")
    args = parser.parse_args()

    roots = discover_task_roots(args.data_root)
    if not roots:
        print("No task directories found. Check --data_root.", file=sys.stderr)
        sys.exit(1)
    if av is None:
        print("PyAV is required for RGB caching. Install `av` in the training environment.", file=sys.stderr)
        sys.exit(1)

    rgb_size = (args.rgb_height, args.rgb_width)
    print(f"=== Pre-caching RGB frames for {len(roots)} task roots ===")
    for root in roots:
        cache_task(
            data_root=root,
            target_fps=args.target_fps,
            cache_dir_name=args.cache_dir_name,
            rgb_size=rgb_size,
            overwrite=args.overwrite,
        )
    print("Done.")


if __name__ == "__main__":
    main()