import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import time

import mujoco
import numpy as np
import torch

from so101_car_parking.data.collect import outcome
from so101_car_parking.simulation.parking import load_task
from so101_car_parking.simulation.teacher import DT, grasp_contacts, robot_contact
from so101_car_parking.policy.vla import EXECUTION_STEPS, load_bundle


def ground_support(model, data, car_id):
    for contact in data.contact:
        a, b = contact.geom1, contact.geom2
        if (
            model.body_rootid[model.geom_bodyid[a]] == car_id
            and model.geom(b).name == "table"
        ) or (
            model.body_rootid[model.geom_bodyid[b]] == car_id
            and model.geom(a).name == "table"
        ):
            return True
    return False


def run_scene(policy, pre, post, task, device, noise_seed, max_steps, progress=None):
    model, data = load_task(task)
    frames, states, actions, raw_actions, timestamps = [], [], [], [], []

    def record(action, raw=None):
        action = np.asarray(action, dtype=np.float32)
        frames.append(data.qpos.copy())
        states.append(data.qpos[:6].astype(np.float32).copy())
        actions.append(action.copy())
        raw_actions.append(
            np.asarray(action if raw is None else raw, dtype=np.float32).copy()
        )
        timestamps.append(float(data.time))
        data.ctrl[:] = action

    for _ in range(20):
        record(data.ctrl)
        mujoco.mj_step(model, data, nstep=round(DT / model.opt.timestep))
    initial = {
        c["color"]: data.body(f"{c['color']}_car").xpos.copy() for c in task["cars"]
    }
    color = task["target_car"]
    car_id = model.body(f"{color}_car").id
    metrics = dict(
        max_distractor_displacement_m=0.0,
        max_lift_m=0.0,
        grasped_lift=False,
        robot_contact_steps=0,
    )
    torch.manual_seed(noise_seed)
    policy.reset()
    policy.eval()
    settled_frames = clipped_steps = 0
    latencies = []
    reason = "timeout"
    started = time.monotonic()
    with (
        mujoco.Renderer(model, height=384, width=512) as renderer,
        torch.inference_mode(),
    ):
        for step in range(max_steps):
            if step % EXECUTION_STEPS == 0:
                observation = {
                    "observation.state": torch.from_numpy(
                        data.qpos[:6].astype(np.float32).copy()
                    ),
                    "task": task["instruction"],
                }
                for name, camera in (("overhead", "overhead"), ("wrist", "wrist_cam")):
                    renderer.update_scene(data, camera=camera)
                    observation[f"observation.images.{name}"] = (
                        torch.from_numpy(renderer.render().copy())
                        .permute(2, 0, 1)
                        .float()
                        / 255
                    )
                batch = pre(observation)
                inference_start = time.monotonic()
            context = (
                torch.autocast("cuda", dtype=torch.bfloat16)
                if device == "cuda"
                else nullcontext()
            )
            with context:
                action = (
                    post(policy.select_action(batch))
                    .detach()
                    .float()
                    .cpu()
                    .numpy()
                    .reshape(-1)
                )
            if step % EXECUTION_STEPS == 0:
                latencies.append(time.monotonic() - inference_start)
            if action.shape != (6,) or not np.isfinite(action).all():
                reason = "invalid_action"
                break
            clipped = np.clip(
                action, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1]
            )
            clipped_steps += int(np.any(action != clipped))
            record(clipped, action)
            failure = None
            for _ in range(round(DT / model.opt.timestep)):
                mujoco.mj_step(model, data)
                height = float(data.body(f"{color}_car").xpos[2] - initial[color][2])
                metrics["max_lift_m"] = max(metrics["max_lift_m"], height)
                if height > 0.025 and grasp_contacts(model, data, color):
                    metrics["grasped_lift"] = True
                metrics["robot_contact_steps"] += int(robot_contact(model, data, color))
                for other, position in initial.items():
                    if other != color:
                        distance = float(
                            np.linalg.norm(data.body(f"{other}_car").xpos - position)
                        )
                        metrics["max_distractor_displacement_m"] = max(
                            metrics["max_distractor_displacement_m"], distance
                        )
                if not np.isfinite(data.qpos).all() or np.any(data.warning.number):
                    failure = "unstable_physics"
                elif metrics["max_distractor_displacement_m"] > 0.005:
                    failure = "distractor_moved"
                elif task["action"] == "push" and metrics["max_lift_m"] > 0.008:
                    failure = "lifted_during_push"
                elif data.body(f"{color}_car").xmat.reshape(3, 3)[2, 2] < 0.85:
                    failure = "target_tipped"
                elif data.body(f"{color}_car").xpos[2] < -0.025:
                    failure = "target_fell"
            if failure:
                reason = failure
                break
            result = outcome(model, data, task, metrics)
            released = not robot_contact(model, data, color) and ground_support(
                model, data, car_id
            )
            settled_frames = settled_frames + 1 if result["success"] and released else 0
            if settled_frames >= 10:
                reason = "success"
                break
            if progress is not None and (step + 1) % 400 == 0:
                progress(step + 1, time.monotonic() - started)
    result = outcome(model, data, task, metrics)
    result.update(
        success=reason == "success",
        termination=reason,
        clipped_steps=clipped_steps,
        control_steps=len(actions) - 20,
        inference_chunks=len(latencies),
        inference_seconds_median=float(np.median(latencies)) if latencies else None,
        wall_seconds=time.monotonic() - started,
        execution_steps=EXECUTION_STEPS,
        initial_settle_seconds=1,
        noise_seed=noise_seed,
    )
    arrays = dict(
        state=np.asarray(states, np.float32),
        action=np.asarray(actions, np.float32),
        raw_action=np.asarray(raw_actions, np.float32),
        qpos=np.asarray(frames),
        timestamp=np.asarray(timestamps),
        final_qpos=data.qpos.copy(),
    )
    return arrays, result


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate only the learned VLA in held-out parking scenes."
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--collection", type=Path, default=Path("data/parking-100"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cuda"], default="cuda")
    parser.add_argument("--max-steps", type=int, default=2400)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--scene-indices",
        type=int,
        nargs="+",
        help="Fixed validation indices for checkpoint comparisons.",
    )
    parser.add_argument(
        "--eager", action="store_true", help="Disable compilation for inference"
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    policy, pre, post = load_bundle(args.checkpoint, args.device)
    policy.requires_grad_(False).eval()
    if args.eager:
        torch.set_float32_matmul_precision("high")
        for name in ("forward", "sample_actions"):
            method = getattr(policy.model, name)
            setattr(
                policy.model,
                name,
                getattr(method, "_torchdynamo_orig_callable", method),
            )
    episodes = [
        json.loads(path.read_text())
        for path in sorted((args.collection / "raw").glob("*.json"))
    ]
    episodes = [episode for episode in episodes if episode["split"] == "validation"]
    if args.limit:
        episodes = episodes[: args.limit]
    if args.scene_indices and any(
        i < 0 or i >= len(episodes) for i in args.scene_indices
    ):
        parser.error("A scene index is outside the validation split")
    records = []
    for i, episode in enumerate(episodes):
        if args.scene_indices is not None and i not in args.scene_indices:
            continue
        path = args.output / f"scene_{i:02d}"
        if path.with_suffix(".json").exists():
            existing = json.loads(path.with_suffix(".json").read_text())
            if existing["checkpoint"] != str(args.checkpoint.resolve()):
                raise RuntimeError("Existing evaluation belongs to another checkpoint")
            records.append(existing)
            continue
        arrays, result = run_scene(
            policy, pre, post, episode["task"], args.device, 70000 + i, args.max_steps
        )
        np.savez_compressed(path.with_suffix(".npz"), **arrays)
        record = dict(
            episode_id=episode["episode_id"],
            task=episode["task"],
            outcome=result,
            checkpoint=str(args.checkpoint.resolve()),
            split="validation",
        )
        path.with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n")
        records.append(record)
        print(
            f"{i + 1}/{len(episodes)}: {episode['task']['action']}, {result['termination']}, {result['control_steps']} steps",
            flush=True,
        )
    summary = dict(
        episodes=len(records),
        successes=sum(r["outcome"]["success"] for r in records),
        checkpoint=str(args.checkpoint),
        execution_steps=EXECUTION_STEPS,
        max_steps=args.max_steps,
        teacher_disabled=True,
        initial_settle_seconds=1,
        results={},
    )
    for action in ("push", "pick"):
        selected = [r for r in records if r["task"]["action"] == action]
        summary["results"][action] = dict(
            episodes=len(selected),
            successes=sum(r["outcome"]["success"] for r in selected),
        )
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
