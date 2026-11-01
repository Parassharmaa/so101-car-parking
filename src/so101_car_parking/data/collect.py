from so101_car_parking.data.provenance import simulation_source_hash
import argparse
from collections import Counter
import hashlib
import itertools
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import mujoco
import numpy as np

from so101_car_parking.simulation.parking import COLORS, load_task, make_task
from so101_car_parking.simulation.teacher import DT, grasp_contacts, pick_and_place, push_to_bay


def bay_region(task):
    bay = next(bay for bay in task["bays"] if bay["color"] == task["target_bay"])
    x, y = bay["center"]
    return (
        "near_corner"
        if x < 0.18
        else "far_corner"
        if x > 0.31 and abs(y) > 0.12
        else "side"
    )


def make_plan(count):
    rng = np.random.default_rng(2026)
    colors = list(COLORS)
    pairs = [
        (colors[i], colors[(i + shift) % 4]) for shift in range(4) for i in range(4)
    ]
    requests = []
    for action in ("push", "pick"):
        for split, number, offset in (
            ("train", count * 4 // 10, 0),
            ("validation", count // 10, 2),
        ):
            goals = [pairs[i % len(pairs)] for i in range(number)]
            regions = [
                ["near_corner", "side", "far_corner"][i % 3] for i in range(number)
            ]
            counts = [2 + (i + offset) % 3 for i in range(number)]
            rng.shuffle(goals)
            rng.shuffle(regions)
            rng.shuffle(counts)
            for pair, region, cars in zip(goals, regions, counts):
                requests.append(
                    dict(
                        action=action,
                        split=split,
                        car_color=pair[0],
                        bay_color=pair[1],
                        region=region,
                        num_cars=cars,
                    )
                )
    rng.shuffle(requests)
    return requests


def outcome(model, data, task, metrics):
    car = data.body(f"{task['target_car']}_car")
    rotation = car.xmat.reshape(3, 3)
    corners = np.array(
        list(itertools.product((-0.040, 0.044), (-0.027, 0.027), (-0.012, 0.028)))
    )
    footprint = (car.xpos + corners @ rotation.T)[:, :2]
    bay = next(bay for bay in task["bays"] if bay["color"] == task["target_bay"])
    inside = bool(
        np.all(
            np.abs(footprint - bay["center"]) <= np.array(task["bay_size"]) / 2 - 0.001
        )
    )
    velocity = data.joint(f"{task['target_car']}_car_free").qvel
    checks = {
        "inside_bay": inside,
        "upright": bool(rotation[2, 2] > 0.99),
        "settled": bool(
            np.linalg.norm(velocity[:3]) < 0.01 and np.linalg.norm(velocity[3:]) < 0.15
        ),
        "distractors_unchanged": metrics["max_distractor_displacement_m"] <= 0.005,
        "robot_contact": metrics["robot_contact_steps"] > 0,
        "requested_action": metrics["grasped_lift"]
        if task["action"] == "pick"
        else metrics["max_lift_m"] <= 0.008,
        "physics_stable": not bool(np.any(data.warning.number)),
    }
    return dict(
        success=all(checks.values()),
        checks=checks,
        metrics=metrics,
        final_car_position=car.xpos.tolist(),
        seconds=float(data.time),
    )


def run_episode(task):
    model, data = load_task(task)
    teacher = pick_and_place if task["action"] == "pick" else push_to_bay
    states, actions, qpos, timestamps = [], [], [], []
    starts = None
    metrics = dict(
        max_distractor_displacement_m=0.0,
        max_lift_m=0.0,
        grasped_lift=False,
        robot_contact_steps=0,
    )
    target = model.body(f"{task['target_car']}_car").id
    robot = model.body_rootid[model.body("base").id]
    error = None
    try:
        for action in teacher(model, data, task):
            if starts is None and data.time >= 1 - 1e-8:
                starts = {
                    car["color"]: data.body(f"{car['color']}_car").xpos.copy()
                    for car in task["cars"]
                }
            action = np.asarray(action, dtype=np.float32)
            states.append(data.qpos[:6].astype(np.float32).copy())
            actions.append(action.copy())
            qpos.append(data.qpos.copy())
            timestamps.append(float(data.time))
            data.ctrl[:] = action
            for _ in range(round(DT / model.opt.timestep)):
                mujoco.mj_step(model, data)
                if starts is None:
                    continue
                height = float(
                    data.body(f"{task['target_car']}_car").xpos[2]
                    - starts[task["target_car"]][2]
                )
                metrics["max_lift_m"] = max(metrics["max_lift_m"], height)
                if (
                    task["action"] == "pick"
                    and height > 0.025
                    and grasp_contacts(model, data, task["target_car"])
                ):
                    metrics["grasped_lift"] = True
                for color, position in starts.items():
                    if color != task["target_car"]:
                        distance = float(
                            np.linalg.norm(data.body(f"{color}_car").xpos - position)
                        )
                        metrics["max_distractor_displacement_m"] = max(
                            metrics["max_distractor_displacement_m"], distance
                        )
                for contact in data.contact:
                    roots = {
                        model.body_rootid[model.geom_bodyid[g]]
                        for g in (contact.geom1, contact.geom2)
                    }
                    if roots == {target, robot}:
                        metrics["robot_contact_steps"] += 1
                        break
            if not np.isfinite(data.qpos).all() or np.any(data.warning.number):
                raise RuntimeError("Unstable physics")
    except RuntimeError as failure:
        error = str(failure)
    result = outcome(model, data, task, metrics)
    if error:
        result["success"] = False
        result["error"] = error
    arrays = dict(
        state=np.asarray(states, np.float32),
        action=np.asarray(actions, np.float32),
        qpos=np.asarray(qpos),
        timestamp=np.asarray(timestamps),
        final_qpos=data.qpos.copy(),
    )
    return arrays, result



def git_revision():
    result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None

def main():
    parser = argparse.ArgumentParser(
        description="Collect successful scripted parking episodes."
    )
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--output", type=Path, default=Path("data/parking-100"))
    parser.add_argument("--attempts", type=int, default=500)
    parser.add_argument("--workers", type=int, choices=[1, 2, 3, 4], default=1)
    parser.add_argument("--worker", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.count < 10 or args.count % 10:
        parser.error("--count must be a positive multiple of 10")
    args.output.mkdir(parents=True, exist_ok=True)
    raw = args.output / "raw"
    raw.mkdir(exist_ok=True)
    source_hash = simulation_source_hash()
    plan_path = args.output / "plan.json"
    if plan_path.exists():
        saved = json.loads(plan_path.read_text())
        if saved["source_hash"] != source_hash or len(saved["requests"]) != args.count:
            raise RuntimeError(
                "Collection source/config changed; use a new output directory."
            )
        plan = saved["requests"]
    else:
        plan = make_plan(args.count)
        plan_path.write_text(
            json.dumps(
                dict(
                    source_hash=source_hash,
                    requests=plan,
                    mujoco=mujoco.__version__,
                    numpy=np.__version__,
                    python=platform.python_version(),
                    fps=round(1 / DT),
                    git_commit=git_revision(),
                ),
                indent=2,
            )
            + "\n"
        )
    if args.workers > 1 and args.worker is None:
        environment = os.environ | {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
        }
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--count",
            str(args.count),
            "--output",
            str(args.output.resolve()),
            "--attempts",
            str(args.attempts),
            "--workers",
            str(args.workers),
        ]
        children = [
            subprocess.Popen(command + ["--worker", str(i)], env=environment)
            for i in range(args.workers)
        ]
        codes = [child.wait() for child in children]
        records = [json.loads(path.read_text()) for path in sorted(raw.glob("*.json"))]
        attempts = [
            json.loads(line)
            for path in sorted(args.output.glob("attempts-*.jsonl"))
            for line in path.read_text().splitlines()
        ]
        (args.output / "attempts.jsonl").write_text(
            "".join(json.dumps(item) + "\n" for item in attempts)
        )
        summary = dict(
            accepted=len(records),
            requested=args.count,
            seed_candidates=len(attempts),
            simulated_attempts=sum("result" in item for item in attempts),
            actions=dict(Counter(item["task"]["action"] for item in records)),
            splits=dict(Counter(item["split"] for item in records)),
        )
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        if any(codes):
            raise RuntimeError(
                "A collector worker stopped; accepted episodes and attempt logs are saved."
            )
        print(json.dumps(summary, indent=2))
        return
    suffix = "" if args.worker is None else f"-{args.worker}"
    log_path = args.output / f"attempts{suffix}.jsonl"
    attempts = (
        [json.loads(line) for line in log_path.read_text().splitlines()]
        if log_path.exists()
        else []
    )
    accepted = []
    summary_path = args.output / f"summary{suffix}.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    started = time.monotonic()
    for episode_id, request in enumerate(plan):
        if args.worker is not None and episode_id % args.workers != args.worker:
            continue
        metadata_path = raw / f"episode_{episode_id:03d}.json"
        array_path = metadata_path.with_suffix(".npz")
        if metadata_path.exists() and array_path.exists():
            accepted.append(json.loads(metadata_path.read_text()))
            continue
        first_attempt = 1 + max(
            (item["attempt"] for item in attempts if item["episode_id"] == episode_id),
            default=-1,
        )
        for attempt in range(first_attempt, first_attempt + args.attempts):
            seed = (
                (1_000_000 if request["split"] == "validation" else 0)
                + episode_id * 5000
                + attempt
            )
            entry = dict(episode_id=episode_id, attempt=attempt, seed=seed)
            try:
                task = make_task(
                    seed,
                    request["num_cars"],
                    request["action"],
                    request["car_color"],
                    request["bay_color"],
                )
            except ValueError:
                entry["reason"] = "requested_color_absent"
            except RuntimeError as failure:
                entry["reason"] = str(failure)
            else:
                if bay_region(task) != request["region"]:
                    entry["reason"] = "region_filter"
                else:
                    arrays, result = run_episode(task)
                    entry["result"] = result
                    if result["success"]:
                        metadata = dict(
                            episode_id=episode_id,
                            split=request["split"],
                            task=task,
                            outcome=result,
                        )
                        temporary = array_path.with_suffix(".tmp")
                        with temporary.open("wb") as stream:
                            np.savez_compressed(stream, **arrays)
                        temporary.replace(array_path)
                        temporary_metadata = metadata_path.with_suffix(".json.tmp")
                        temporary_metadata.write_text(
                            json.dumps(metadata, indent=2) + "\n"
                        )
                        temporary_metadata.replace(metadata_path)
                        accepted.append(metadata)
            attempts.append(entry)
            with log_path.open("a") as stream:
                stream.write(json.dumps(entry) + "\n")
            if metadata_path.exists():
                print(
                    f"{len(list(raw.glob('*.json')))}/{args.count}: {request['split']} {request['action']} "
                    f"{request['num_cars']} cars, {request['car_color']}→{request['bay_color']}, "
                    f"{request['region']}, seed {seed}, {len(arrays['action'])} frames",
                    flush=True,
                )
                break
        else:
            raise RuntimeError(
                f"No accepted episode for request {episode_id}; progress is saved."
            )
        assigned = (
            args.count
            if args.worker is None
            else sum(i % args.workers == args.worker for i in range(args.count))
        )
        summary = dict(
            accepted=len(accepted),
            requested=assigned,
            seed_candidates=len(attempts),
            simulated_attempts=sum("result" in item for item in attempts),
            actions=dict(Counter(item["task"]["action"] for item in accepted)),
            splits=dict(Counter(item["split"] for item in accepted)),
            elapsed_seconds=round(time.monotonic() - started, 1),
        )
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    if args.worker is None:
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
