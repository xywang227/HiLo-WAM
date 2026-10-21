#!/usr/bin/env python3
"""Convert a local LeRobot v2.1 dataset to v3.0 with index validation.

This is a thin, local-only wrapper around LeRobot's v2.1 -> v3.0 converter.
It fixes the upstream data-file rollover bookkeeping in the upstream
LeRobot v0.4.0 converter and verifies every episode-to-parquet
mapping after conversion.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def _bootstrap_lerobot_source() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--lerobot-src",
        default=os.environ.get("LEROBOT_V3_SRC"),
        help="Directory containing the v3-compatible lerobot Python package.",
    )
    args, remaining = parser.parse_known_args()
    if args.lerobot_src:
        source = Path(args.lerobot_src).expanduser().resolve()
        if not (source / "lerobot").is_dir():
            raise FileNotFoundError(f"No lerobot package found below --lerobot-src={source}")
        sys.path.insert(0, str(source))
    return args, remaining


BOOTSTRAP_ARGS, REMAINING_ARGS = _bootstrap_lerobot_source()

def _load_conversion_dependencies() -> None:
    global pq, lerobot_utils, converter
    try:
        import pyarrow.parquet as pq
        from lerobot.datasets import utils as lerobot_utils
        from lerobot.datasets.v30 import convert_dataset_v21_to_v30 as converter
    except ImportError as exc:
        raise SystemExit(
            "A LeRobot checkout containing lerobot.datasets.v30 is required. "
            "Pass --lerobot-src /path/to/lerobot/src or set LEROBOT_V3_SRC. "
            f"Original import error: {exc}"
        ) from exc


def convert_data_with_correct_rollover(
    root: Path,
    new_root: Path,
    data_file_size_in_mb: int,
) -> list[dict]:
    """Consolidate episode parquet files without mis-indexing rollover episodes."""
    episode_paths = sorted((root / "data").glob("*/*.parquet"))
    image_keys = converter.get_image_keys(root)

    chunk_index = 0
    file_index = 0
    current_size_mb = 0.0
    total_frames = 0
    paths_to_concat: list[Path] = []
    episodes_metadata: list[dict] = []

    converter.logging.info("Converting data files from %d episodes", len(episode_paths))
    for episode_index, episode_path in enumerate(
        converter.tqdm.tqdm(episode_paths, desc="convert data files")
    ):
        episode_size_mb = converter.get_parquet_file_size_in_mb(episode_path)
        episode_num_frames = converter.get_parquet_num_frames(episode_path)

        # Flush before assigning the current episode to a file.  The vendored
        # converter assigned its metadata first, so every rollover episode
        # incorrectly referenced the preceding parquet file.
        if paths_to_concat and current_size_mb + episode_size_mb >= data_file_size_in_mb:
            converter.concat_data_files(
                paths_to_concat,
                new_root,
                chunk_index,
                file_index,
                image_keys,
            )
            chunk_index, file_index = converter.update_chunk_file_indices(
                chunk_index,
                file_index,
                lerobot_utils.DEFAULT_CHUNK_SIZE,
            )
            current_size_mb = 0.0
            paths_to_concat = []

        episodes_metadata.append(
            {
                "episode_index": episode_index,
                "data/chunk_index": chunk_index,
                "data/file_index": file_index,
                "dataset_from_index": total_frames,
                "dataset_to_index": total_frames + episode_num_frames,
            }
        )
        paths_to_concat.append(episode_path)
        current_size_mb += episode_size_mb
        total_frames += episode_num_frames

    if paths_to_concat:
        converter.concat_data_files(
            paths_to_concat,
            new_root,
            chunk_index,
            file_index,
            image_keys,
        )

    return episodes_metadata


def convert_videos_in_parallel(
    root: Path,
    new_root: Path,
    video_file_size_in_mb: int,
) -> list[dict] | None:
    """Convert independent camera streams concurrently while preserving key order."""
    video_keys = sorted(converter.get_video_keys(root))
    if not video_keys:
        return None

    with ThreadPoolExecutor(max_workers=min(len(video_keys), 3)) as executor:
        futures = [
            executor.submit(
                converter.convert_videos_of_camera,
                root,
                new_root,
                video_key,
                video_file_size_in_mb,
            )
            for video_key in video_keys
        ]
        metadata_per_camera = [future.result() for future in futures]

    episode_counts = {len(camera_metadata) for camera_metadata in metadata_per_camera}
    if len(episode_counts) != 1:
        raise ValueError(f"All cameras must have the same episode count, got {episode_counts}")

    merged_metadata: list[dict] = []
    for episode_index in range(episode_counts.pop()):
        merged = {"episode_index": episode_index}
        for camera_metadata in metadata_per_camera:
            row = camera_metadata[episode_index]
            if int(row["episode_index"]) != episode_index:
                raise ValueError(
                    f"Camera episode mismatch at {episode_index}: {row['episode_index']}"
                )
            merged.update(row)
        merged_metadata.append(merged)
    return merged_metadata


def convert_episodes_metadata_chunked(
    root: Path,
    new_root: Path,
    episodes_metadata: list[dict],
    episodes_video_metadata: list[dict] | None = None,
) -> None:
    """Write large v3 episode metadata as multiple standard parquet files."""
    from datasets import Dataset

    legacy_episodes = converter.legacy_load_episodes(root)
    episode_stats = converter.legacy_load_episodes_stats(root)
    episode_count = len(episodes_metadata)
    counts = {len(legacy_episodes), len(episode_stats), episode_count}
    if episodes_video_metadata is not None:
        counts.add(len(episodes_video_metadata))
    if len(counts) != 1:
        raise ValueError(f"Episode metadata counts differ: {counts}")

    rows_per_file = 2500
    output_dir = new_root / lerobot_utils.EPISODES_DIR / "chunk-000"
    output_dir.mkdir(parents=True, exist_ok=True)
    legacy_rows = list(legacy_episodes.values())
    stats_rows = list(episode_stats.values())

    for file_index, start in enumerate(range(0, episode_count, rows_per_file)):
        rows = []
        stop = min(start + rows_per_file, episode_count)
        for episode_index in range(start, stop):
            legacy_row = legacy_rows[episode_index]
            data_row = episodes_metadata[episode_index]
            video_row = (
                {} if episodes_video_metadata is None else episodes_video_metadata[episode_index]
            )
            ids = {
                int(legacy_row["episode_index"]),
                int(data_row["episode_index"]),
                episode_index,
            }
            if video_row:
                ids.add(int(video_row["episode_index"]))
            if len(ids) != 1:
                raise ValueError(f"Episode index mismatch at {episode_index}: {ids}")
            row = {
                **data_row,
                **video_row,
                **legacy_row,
                **converter.flatten_dict({"stats": stats_rows[episode_index]}),
                "meta/episodes/chunk_index": 0,
                "meta/episodes/file_index": file_index,
            }
            rows.append(row)

        output_path = output_dir / f"file-{file_index:03d}.parquet"
        Dataset.from_list(rows).to_parquet(output_path)

    aggregate_stats = converter.aggregate_stats(stats_rows)
    converter.write_stats(aggregate_stats, new_root)


def reconstruct_data_metadata(staging_root: Path, legacy_root: Path) -> list[dict]:
    """Rebuild episode-to-data mappings from already converted parquet files."""
    legacy_episodes = converter.legacy_load_episodes(legacy_root)
    lengths = {index: int(row["length"]) for index, row in legacy_episodes.items()}
    locations: dict[int, tuple[int, int]] = {}
    for path in sorted((staging_root / "data").glob("chunk-*/file-*.parquet")):
        chunk_index = int(path.parent.name.removeprefix("chunk-"))
        file_index = int(path.stem.removeprefix("file-"))
        episode_indices = (
            pq.read_table(path, columns=["episode_index"])
            .column("episode_index")
            .unique()
            .to_pylist()
        )
        for episode_index in episode_indices:
            locations[int(episode_index)] = (chunk_index, file_index)

    metadata = []
    frame_offset = 0
    for episode_index in range(len(legacy_episodes)):
        if episode_index not in locations:
            raise ValueError(f"No staged data parquet contains episode {episode_index}")
        chunk_index, file_index = locations[episode_index]
        length = lengths[episode_index]
        metadata.append(
            {
                "episode_index": episode_index,
                "data/chunk_index": chunk_index,
                "data/file_index": file_index,
                "dataset_from_index": frame_offset,
                "dataset_to_index": frame_offset + length,
            }
        )
        frame_offset += length
    return metadata


def reconstruct_video_metadata(
    staging_root: Path,
    legacy_root: Path,
    video_file_size_in_mb: int,
) -> list[dict] | None:
    """Rebuild v3 video offsets without remuxing already converted videos."""
    video_keys = sorted(converter.get_video_keys(legacy_root))
    if not video_keys:
        return None
    per_camera: list[list[dict]] = []
    for video_key in video_keys:
        chunk_index = 0
        file_index = 0
        size_mb = 0.0
        duration_s = 0.0
        rows = []
        paths = sorted((legacy_root / "videos").glob(f"*/{video_key}/*.mp4"))
        for episode_index, path in enumerate(paths):
            episode_size_mb = converter.get_file_size_in_mb(path)
            episode_duration_s = converter.get_video_duration_in_s(path)
            if size_mb + episode_size_mb >= video_file_size_in_mb and size_mb > 0:
                chunk_index, file_index = converter.update_chunk_file_indices(
                    chunk_index,
                    file_index,
                    lerobot_utils.DEFAULT_CHUNK_SIZE,
                )
                size_mb = 0.0
                duration_s = 0.0
            target = staging_root / lerobot_utils.DEFAULT_VIDEO_PATH.format(
                video_key=video_key,
                chunk_index=chunk_index,
                file_index=file_index,
            )
            if not target.is_file():
                raise FileNotFoundError(f"Missing staged video file: {target}")
            rows.append(
                {
                    "episode_index": episode_index,
                    f"videos/{video_key}/chunk_index": chunk_index,
                    f"videos/{video_key}/file_index": file_index,
                    f"videos/{video_key}/from_timestamp": duration_s,
                    f"videos/{video_key}/to_timestamp": duration_s + episode_duration_s,
                }
            )
            size_mb += episode_size_mb
            duration_s += episode_duration_s
        per_camera.append(rows)

    merged_rows = []
    for episode_index in range(len(per_camera[0])):
        row = {"episode_index": episode_index}
        for camera_rows in per_camera:
            row.update(camera_rows[episode_index])
        merged_rows.append(row)
    return merged_rows


def resume_existing_staging(dataset_root: Path) -> None:
    """Finish metadata and directory swap after a late converter failure."""
    staging_root = dataset_root.with_name(f"{dataset_root.name}_v30")
    backup_root = dataset_root.with_name(f"{dataset_root.name}_old")
    if not dataset_root.is_dir() or not staging_root.is_dir():
        raise FileNotFoundError("Both the v2.1 dataset and v3 staging directories are required")
    if backup_root.exists():
        raise FileExistsError(f"Refusing to replace existing backup: {backup_root}")

    with (staging_root / "meta/info.json").open("r", encoding="utf-8") as stream:
        info = json.load(stream)
    data_metadata = reconstruct_data_metadata(staging_root, dataset_root)
    video_metadata = reconstruct_video_metadata(
        staging_root,
        dataset_root,
        int(info["video_files_size_in_mb"]),
    )
    episodes_dir = staging_root / lerobot_utils.EPISODES_DIR
    if episodes_dir.exists():
        shutil.rmtree(episodes_dir)
    convert_episodes_metadata_chunked(
        dataset_root,
        staging_root,
        data_metadata,
        video_metadata,
    )
    validate_v30_dataset(staging_root)
    shutil.move(str(dataset_root), str(backup_root))
    shutil.move(str(staging_root), str(dataset_root))
    print(f"Resumed and finalized v3.0 dataset at {dataset_root}")
    print(f"The original v2.1 dataset was retained at {backup_root}")


def _read_episode_rows(dataset_root: Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted((dataset_root / "meta" / "episodes").glob("chunk-*/file-*.parquet")):
        rows.extend(pq.read_table(path).to_pylist())
    return sorted(rows, key=lambda row: int(row["episode_index"]))


def validate_v30_dataset(dataset_root: Path) -> None:
    """Validate metadata version and all data/video file references."""
    with (dataset_root / "meta" / "info.json").open("r", encoding="utf-8") as stream:
        info = json.load(stream)
    if info.get("codebase_version") != "v3.0":
        raise ValueError(
            f"Expected codebase_version='v3.0', got {info.get('codebase_version')!r}"
        )

    episode_rows = _read_episode_rows(dataset_root)
    if len(episode_rows) != int(info["total_episodes"]):
        raise ValueError(
            f"Episode metadata count mismatch: {len(episode_rows)} != {info['total_episodes']}"
        )

    episodes_by_data_file: dict[Path, set[int]] = {}
    episodes_by_video_file: dict[Path, list[tuple[float, float, int]]] = {}
    video_keys = [
        key for key, feature in info["features"].items() if feature.get("dtype") == "video"
    ]
    for expected_index, row in enumerate(episode_rows):
        episode_index = int(row["episode_index"])
        if episode_index != expected_index:
            raise ValueError(
                f"Episode indices are not contiguous at position {expected_index}: {episode_index}"
            )
        data_path = dataset_root / info["data_path"].format(
            chunk_index=int(row["data/chunk_index"]),
            file_index=int(row["data/file_index"]),
        )
        episodes_by_data_file.setdefault(data_path, set()).add(episode_index)

        for video_key in video_keys:
            video_path = dataset_root / info["video_path"].format(
                video_key=video_key,
                chunk_index=int(row[f"videos/{video_key}/chunk_index"]),
                file_index=int(row[f"videos/{video_key}/file_index"]),
            )
            if not video_path.is_file():
                raise FileNotFoundError(f"Missing video referenced by episode {episode_index}: {video_path}")
            from_timestamp = float(row[f"videos/{video_key}/from_timestamp"])
            to_timestamp = float(row[f"videos/{video_key}/to_timestamp"])
            if to_timestamp <= from_timestamp:
                raise ValueError(f"Invalid video timestamp range for episode {episode_index}/{video_key}")
            episodes_by_video_file.setdefault(video_path, []).append(
                (from_timestamp, to_timestamp, episode_index)
            )

    for data_path, expected_episodes in episodes_by_data_file.items():
        if not data_path.is_file():
            raise FileNotFoundError(f"Missing data parquet: {data_path}")
        actual_episodes = {
            int(value)
            for value in pq.read_table(data_path, columns=["episode_index"])
            .column("episode_index")
            .unique()
            .to_pylist()
        }
        if expected_episodes != actual_episodes:
            missing = sorted(expected_episodes - actual_episodes)
            unexpected = sorted(actual_episodes - expected_episodes)
            raise ValueError(
                f"Incorrect episode-to-parquet mapping in {data_path}: "
                f"missing={missing[:10]}, unexpected={unexpected[:10]}"
            )

    # Container durations may be rounded to the stream time base.  Two frames
    # is strict enough to catch a bad concatenation without rejecting normal
    # MP4 duration rounding.
    timestamp_tolerance_s = max(1e-3, 2.0 / float(info["fps"]))
    for video_path, ranges in episodes_by_video_file.items():
        ranges.sort()
        previous_to = 0.0
        for from_timestamp, to_timestamp, episode_index in ranges:
            if abs(from_timestamp - previous_to) > timestamp_tolerance_s:
                raise ValueError(
                    f"Non-contiguous timestamps in {video_path} before episode {episode_index}: "
                    f"{previous_to} -> {from_timestamp}"
                )
            previous_to = to_timestamp
        actual_duration = converter.get_video_duration_in_s(video_path)
        if abs(actual_duration - previous_to) > timestamp_tolerance_s:
            raise ValueError(
                f"Video duration mismatch in {video_path}: metadata={previous_to}, "
                f"actual={actual_duration}"
            )

    print(
        f"Validated LeRobot v3.0 dataset at {dataset_root}: "
        f"{len(episode_rows)} episodes, {len(episodes_by_data_file)} data files, "
        f"{len(video_keys)} video keys."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Safely convert a local LeRobot v2.1 dataset to v3.0."
    )
    parser.add_argument("--lerobot-src", default=BOOTSTRAP_ARGS.lerobot_src)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--root", required=True, help="Parent directory containing REPO_ID.")
    parser.add_argument("--data-file-size-in-mb", type=int, default=None)
    parser.add_argument("--video-file-size-in-mb", type=int, default=None)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--resume-staging",
        action="store_true",
        help="Finish an existing REPO_ID_v30 staging tree without remuxing videos.",
    )
    return parser.parse_args(REMAINING_ARGS)


def main() -> None:
    args = parse_args()
    _load_conversion_dependencies()
    dataset_root = Path(args.root).expanduser().resolve() / args.repo_id
    if args.validate_only:
        validate_v30_dataset(dataset_root)
        return
    if args.resume_staging:
        resume_existing_staging(dataset_root)
        return

    backup_root = dataset_root.with_name(f"{dataset_root.name}_old")
    staging_root = dataset_root.with_name(f"{dataset_root.name}_v30")
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Missing extracted v2.1 dataset: {dataset_root}")
    if backup_root.exists() or staging_root.exists():
        raise FileExistsError(
            "Refusing to delete or replace an existing conversion directory. "
            f"Move it aside first: {backup_root} or {staging_root}"
        )

    # Patch only this process; the source LeRobot checkout remains untouched.
    converter.convert_data = convert_data_with_correct_rollover
    converter.convert_videos = convert_videos_in_parallel
    converter.convert_episodes_metadata = convert_episodes_metadata_chunked
    converter.convert_dataset(
        repo_id=args.repo_id,
        root=args.root,
        data_file_size_in_mb=args.data_file_size_in_mb,
        video_file_size_in_mb=args.video_file_size_in_mb,
        push_to_hub=False,
        force_conversion=True,
    )
    validate_v30_dataset(dataset_root)
    print(f"The original v2.1 dataset was retained at {backup_root}")


if __name__ == "__main__":
    main()
