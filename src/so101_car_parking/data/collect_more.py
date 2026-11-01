from so101_car_parking.data.provenance import simulation_source_hash
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path

import numpy as np

from so101_car_parking.data.collect import bay_region, make_plan, run_episode
from so101_car_parking.simulation.parking import make_task


def collect_one(item):
    episode_id, request, output = item
    path = output / "raw" / f"episode_{episode_id:03d}.json"
    if path.exists() and path.with_suffix(".npz").exists():
        return json.loads(path.read_text())
    for attempt in range(500):
        seed = 10_000_000 + episode_id * 5000 + attempt
        entry = dict(episode_id=episode_id, attempt=attempt, seed=seed)
        try:
            task = make_task(
                seed,
                request["num_cars"],
                request["action"],
                request["car_color"],
                request["bay_color"],
            )
            if bay_region(task) != request["region"]:
                entry["reason"] = "region_filter"
            else:
                arrays, result = run_episode(task)
                entry["result"] = result
                if result["success"]:
                    metadata = dict(
                        episode_id=episode_id, split="train", task=task, outcome=result
                    )
                    with path.with_suffix(".npz.tmp").open("wb") as stream:
                        np.savez_compressed(stream, **arrays)
                    path.with_suffix(".npz.tmp").replace(path.with_suffix(".npz"))
                    path.with_suffix(".json.tmp").write_text(
                        json.dumps(metadata, indent=2) + "\n"
                    )
                    path.with_suffix(".json.tmp").replace(path)
        except (ValueError, RuntimeError) as error:
            entry["reason"] = str(error)
        with (output / f"attempts-{episode_id}.jsonl").open("a") as stream:
            stream.write(json.dumps(entry) + "\n")
        if path.exists():
            return metadata
    raise RuntimeError(f"No successful demonstration for episode {episode_id}")


def main():
    parser = argparse.ArgumentParser(
        description="Collect extra training data without changing the held-out set."
    )
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--output", type=Path, default=Path("data/parking-extra-200"))
    parser.add_argument("--workers", type=int, default=3, choices=[1, 2, 3, 4])
    args = parser.parse_args()
    if args.count <= 0 or args.count % 8:
        parser.error("--count must be a positive multiple of 8")
    requests = [r for r in make_plan(args.count * 5 // 4) if r["split"] == "train"]
    if len(requests) != args.count:
        raise RuntimeError("Training plan count differs")
    digest = simulation_source_hash()
    plan = dict(
        source_hash=digest,
        generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        requests=requests,
        first_episode_id=1000,
        seed_offset=10_000_000,
        fps=20,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "raw").mkdir(exist_ok=True)
    path = args.output / "plan.json"
    if path.exists() and json.loads(path.read_text()) != plan:
        raise RuntimeError("Collection configuration changed; choose a new directory")
    path.write_text(json.dumps(plan, indent=2) + "\n")
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = [
            pool.submit(collect_one, (1000 + i, request, args.output))
            for i, request in enumerate(requests)
        ]
        for count, job in enumerate(as_completed(jobs), 1):
            record = job.result()
            print(
                f"{count}/{args.count}: {record['task']['instruction']}, seed {record['task']['seed']}",
                flush=True,
            )


if __name__ == "__main__":
    main()
