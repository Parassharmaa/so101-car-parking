import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

import pyarrow as pa
import pyarrow.parquet as pq

from so101_car_parking.data.export_data import merge_parts
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def normalize_locators(root):
    rows = 0
    for path in sorted((root / "meta/episodes").rglob("*.parquet")):
        table = pq.read_table(path)
        for name, value in (
            ("meta/episodes/chunk_index", int(path.parent.name.split("-")[-1])),
            ("meta/episodes/file_index", int(path.stem.split("-")[-1])),
        ):
            index = table.schema.get_field_index(name)
            values = pa.array([value] * len(table), type=table.schema.field(index).type)
            table = table.set_column(index, table.schema.field(index), values)
        temporary = path.with_suffix(".tmp")
        pq.write_table(table, temporary)
        temporary.replace(path)
        rows += len(table)
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Merge extra training episodes while preserving the original validation files."
    )
    parser.add_argument("original", type=Path)
    parser.add_argument("extra", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if (args.output / "lerobot/train").exists():
        parser.error("Training output already exists; use a fresh destination")
    collections = [args.original, args.extra]
    records = [
        json.loads(p.read_text())
        for root in collections
        for p in sorted((root / "raw").glob("*.json"))
    ]
    if len({r["task"]["seed"] for r in records}) != len(records):
        raise RuntimeError("Scene seeds overlap")
    train = [r for r in records if r["split"] == "train"]
    parts = []
    for i, collection in enumerate(collections):
        source = collection / "lerobot"
        part = args.output / "aggregation-sources" / str(i)
        for name in ("data", "videos"):
            target = part / "train" / name
            if not target.exists():
                shutil.copytree(source / "train" / name, target, copy_function=os.link)
        shutil.copytree(source / "train/meta", part / "train/meta", dirs_exist_ok=True)
        shutil.copyfile(source / "episode_map.json", part / "episode_map.json")
        # LeRobot 0.6.1 can retain source-file locators after packing metadata.
        # Correct only private metadata copies; numeric data and videos stay untouched.
        normalize_locators(part / "train")
        parts.append(part)
    native = args.output / "lerobot"
    merge_parts(parts, native, train)
    normalize_locators(native / "train")
    shutil.copytree(
        args.original / "lerobot/validation",
        native / "validation",
        copy_function=os.link,
    )
    mapping = json.loads((native / "episode_map.json").read_text())
    mapping["validation"] = json.loads(
        (args.original / "lerobot/episode_map.json").read_text()
    )["validation"]
    (native / "episode_map.json").write_text(json.dumps(mapping, indent=2) + "\n")
    raw = args.output / "raw"
    raw.mkdir(exist_ok=True)
    for collection in collections:
        for path in (collection / "raw").iterdir():
            if not (raw / path.name).exists():
                (raw / path.name).symlink_to(path.resolve())
    counts = {}
    for split in ("train", "validation"):
        data = LeRobotDataset(
            f"local/parking-300-{split}", root=native / split, video_backend="pyav"
        )
        counts[split] = dict(episodes=data.num_episodes, frames=data.num_frames)
    if counts["train"]["episodes"] != len(train) or counts["validation"][
        "episodes"
    ] != len(mapping["validation"]):
        raise RuntimeError("Merged episode count differs")
    for source in (args.original / "lerobot/validation").rglob("*"):
        if source.is_file():
            target = (
                native
                / "validation"
                / source.relative_to(args.original / "lerobot/validation")
            )
            if (
                hashlib.file_digest(source.open("rb"), "sha256").digest()
                != hashlib.file_digest(target.open("rb"), "sha256").digest()
            ):
                raise RuntimeError("Validation files differ")
    result = dict(
        counts=counts,
        validation_identical=True,
        unique_scene_seeds=len(records),
        sources=[str(root) for root in collections],
        replay_verified_during_export=True,
    )
    (args.output / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
