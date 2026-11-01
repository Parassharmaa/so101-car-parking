import argparse
import hashlib
import json
from pathlib import Path

from lerobot.configs.video import RGBEncoderConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import mujoco
import numpy as np

from parking import load_task
from replay import replay
from teacher import DT


JOINTS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


def merge_parts(parts, output, episodes):
    from lerobot.datasets.aggregate import aggregate_datasets

    mapping = {}
    for split in ("train", "validation"):
        if not any(episode["split"] == split for episode in episodes):
            continue
        roots = [part for part in parts if (part / split / "meta/info.json").exists()]
        sources = [
            LeRobotDataset(
                f"local/source-{split}-{i}", root=part / split, video_backend="pyav"
            )
            for i, part in enumerate(roots)
        ]
        entries = [
            json.loads((part / "episode_map.json").read_text())[split] for part in roots
        ]
        if any(
            len(items) != source.num_episodes for items, source in zip(entries, sources)
        ):
            raise RuntimeError("Export part metadata is incomplete")
        mapping[split] = [
            dict(item, episode_index=i)
            for i, item in enumerate(item for items in entries for item in items)
        ]
        ids = [item["raw_episode_id"] for item in mapping[split]]
        expected = {
            episode["episode_id"] for episode in episodes if episode["split"] == split
        }
        if len(ids) != len(set(ids)) or set(ids) != expected:
            raise RuntimeError("Export parts do not cover the split exactly once")
        root = output / split
        if not (root / "meta/info.json").exists():
            aggregate_datasets(
                repo_ids=[source.repo_id for source in sources],
                roots=[source.root for source in sources],
                aggr_repo_id=f"local/so101-parking-{split}",
                aggr_root=root,
                concatenate_videos=False,
                concatenate_data=False,
            )
        merged = LeRobotDataset(
            f"local/so101-parking-{split}", root=root, video_backend="pyav"
        )
        if merged.num_episodes != len(ids) or merged.num_frames != sum(
            source.num_frames for source in sources
        ):
            raise RuntimeError("Merged dataset counts differ")
        print(
            f"Merged {split}: {merged.num_episodes} episodes, {merged.num_frames} frames",
            flush=True,
        )
    (output / "episode_map.json").write_text(json.dumps(mapping, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Replay accepted episodes into local LeRobot datasets."
    )
    parser.add_argument("collection", type=Path)
    parser.add_argument(
        "--limit", type=int, help="Export an accepted prefix for a pipeline pilot."
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Independent output root for a parallel export part.",
    )
    parser.add_argument("--split", choices=["train", "validation"])
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int)
    parser.add_argument("--tag", default="")
    parser.add_argument("--merge-parts", nargs="+", type=Path)
    args = parser.parse_args()
    plan = json.loads((args.collection / "plan.json").read_text())
    names = ["scene.xml", "sim.py", "parking.py", "teacher.py", "collect.py"]
    source_hash = hashlib.sha256(
        b"".join(Path(__file__).with_name(name).read_bytes() for name in names)
    ).hexdigest()
    if source_hash != plan["source_hash"]:
        raise RuntimeError("Simulator/teacher source differs from the collection.")
    episodes = [
        json.loads(path.read_text())
        for path in sorted((args.collection / "raw").glob("*.json"))
    ]
    if (not args.limit and len(episodes) != len(plan["requests"])) or not all(
        e["outcome"]["success"] for e in episodes
    ):
        raise RuntimeError(
            "Collection is incomplete or includes unsuccessful episodes."
        )
    if args.limit:
        prefix = {episode["episode_id"]: episode for episode in episodes}
        if any(i not in prefix for i in range(args.limit)):
            raise RuntimeError("Requested episode prefix is not complete yet.")
        episodes = [prefix[i] for i in range(args.limit)]
    output = args.output or args.collection / "lerobot"
    if args.merge_parts:
        merge_parts(args.merge_parts, output, episodes)
        return
    features = {
        "observation.state": dict(dtype="float32", shape=(6,), names=JOINTS),
        "action": dict(dtype="float32", shape=(6,), names=JOINTS),
        **{
            f"observation.images.{camera}": dict(
                dtype="video", shape=(384, 512, 3), names=["height", "width", "channel"]
            )
            for camera in ("overhead", "wrist")
        },
    }
    encoder = RGBEncoderConfig(
        vcodec="h264", pix_fmt="yuv420p", crf=23, preset="fast", g=20
    )
    mapping_path = output / "episode_map.json"
    previous_mapping = (
        json.loads(mapping_path.read_text()) if mapping_path.exists() else {}
    )
    mapping = {split: entries.copy() for split, entries in previous_mapping.items()}
    for split in [args.split] if args.split else ["train", "validation"]:
        selected = [episode for episode in episodes if episode["split"] == split][
            args.start : args.stop
        ]
        root = output / split
        repo_id = f"local/so101-parking-{split}" + (f"-{args.tag}" if args.tag else "")
        mapping[split] = []
        for index, episode in enumerate(selected):
            options = dict(
                root=root,
                video_backend="pyav",
                rgb_encoder=encoder,
                encoder_threads=2,
                streaming_encoding=True,
                encoder_queue_maxsize=8,
            )
            if (root / "meta/info.json").exists():
                dataset = LeRobotDataset.resume(repo_id, **options)
            else:
                dataset = LeRobotDataset.create(
                    repo_id,
                    fps=round(1 / DT),
                    features=features,
                    robot_type="so101_mujoco",
                    metadata_buffer_size=1,
                    **options,
                )
            entry = dict(
                episode_index=index,
                raw_episode_id=episode["episode_id"],
                seed=episode["task"]["seed"],
                action=episode["task"]["action"],
                instruction=episode["task"]["instruction"],
            )
            mapping[split].append(entry)
            if index < dataset.num_episodes:
                previous = previous_mapping.get(split, [])
                if (
                    index >= len(previous)
                    or previous[index]["raw_episode_id"] != episode["episode_id"]
                ):
                    dataset.finalize()
                    raise RuntimeError(
                        "Existing dataset episode mapping differs from the raw collection."
                    )
                dataset.finalize()
                continue
            with np.load(
                args.collection / "raw" / f"episode_{episode['episode_id']:03d}.npz"
            ) as archive:
                arrays = {key: archive[key] for key in archive.files}
            model, data = load_task(episode["task"])
            try:
                with mujoco.Renderer(model, height=384, width=512) as renderer:
                    for frame in replay(model, data, arrays):
                        row = {
                            "observation.state": arrays["state"][frame],
                            "action": arrays["action"][frame],
                            "task": episode["task"]["instruction"],
                        }
                        for key, camera in (
                            ("overhead", "overhead"),
                            ("wrist", "wrist_cam"),
                        ):
                            renderer.update_scene(data, camera=camera)
                            row[f"observation.images.{key}"] = renderer.render()
                        dataset.add_frame(row)
                dataset.save_episode()
            finally:
                dataset.finalize()
            mapping_path.write_text(json.dumps(mapping, indent=2) + "\n")
            print(
                f"{split} {index + 1}/{len(selected)}: replay verified, {len(arrays['action'])} frames",
                flush=True,
            )
        root.parent.mkdir(parents=True, exist_ok=True)
        mapping_path.write_text(json.dumps(mapping, indent=2) + "\n")
    print(
        "Export complete. Use training statistics for both policy training and validation."
    )


if __name__ == "__main__":
    main()
