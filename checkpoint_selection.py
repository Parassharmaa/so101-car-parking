import itertools

import json

from pathlib import Path

import mujoco

import numpy as np

from parking import load_task

from teacher import robot_contact


def load_records(folder):
    records = []
    for path in sorted(Path(folder).glob("scene_*.json")):
        record = json.loads(path.read_text())
        model, data = load_task(record["task"])
        task = record["task"]
        address = model.jnt_qposadr[model.joint(f"{task['target_car']}_car_free").id]
        with np.load(path.with_suffix(".npz")) as trajectory:
            data.qpos[:] = trajectory["final_qpos"]
            positions = np.concatenate(
                (
                    trajectory["qpos"][20:, address : address + 2],
                    trajectory["final_qpos"][None, address : address + 2],
                )
            )
        mujoco.mj_forward(model, data)
        car = data.body(f"{task['target_car']}_car")
        pose_offset = float(
            np.max(np.abs(car.xpos - record["outcome"]["final_car_position"]))
        )
        corners = np.array(
            list(itertools.product((-0.040, 0.044), (-0.027, 0.027), (-0.012, 0.028)))
        )
        footprint = (car.xpos + corners @ car.xmat.reshape(3, 3).T)[:, :2]
        bay = next(bay for bay in task["bays"] if bay["color"] == task["target_bay"])
        distances = np.linalg.norm(positions - bay["center"], axis=1)
        initial_distance, final_distance = float(distances[0]), float(distances[-1])
        edges = (footprint[:, None, :] - footprint[None, :, :]).reshape(-1, 2)
        axes = np.concatenate((np.eye(2), edges[:, ::-1] * [-1, 1]))
        lengths = np.linalg.norm(axes, axis=1)
        axes = axes[lengths > 1e-12] / lengths[lengths > 1e-12, None]
        projected = footprint @ axes.T
        center = np.asarray(bay["center"]) @ axes.T
        radius = np.abs(axes) @ (np.asarray(task["bay_size"]) / 2)
        overlap = bool(
            np.all(projected.max(axis=0) >= center - radius - 1e-9)
            and np.all(projected.min(axis=0) <= center + radius + 1e-9)
        )
        supported = False
        for contact in data.contact:
            a, b = contact.geom1, contact.geom2
            if model.body_rootid[model.geom_bodyid[b]] == car.id:
                a, b = b, a
            if (
                model.body_rootid[model.geom_bodyid[a]] == car.id
                and model.geom(b).name == "table"
            ):
                supported = True
        released = not robot_contact(model, data, task["target_car"])
        record["partial_progress"] = dict(
            footprint_overlaps_bay=overlap,
            table_supported=supported,
            released=released,
            bay_overlap_on_table=overlap and supported,
            recorded_pose_offset_m=pose_offset,
            initial_distance_m=initial_distance,
            final_distance_m=final_distance,
            closest_distance_m=float(distances.min()),
            distance_closed_m=initial_distance - final_distance,
            fraction_distance_closed=(initial_distance - final_distance)
            / max(initial_distance, 1e-6),
        )
        records.append(record)
    return records


def score(records):
    successes = bay_overlaps = pickups = push_progress = contacts = 0
    progress = []
    for record in records:
        task, outcome = record["task"], record["outcome"]
        metrics, checks = outcome["metrics"], outcome["checks"]
        successes += int(outcome["success"])
        safe = (
            checks["physics_stable"]
            and checks["distractors_unchanged"]
            and checks["upright"]
            and (task["action"] != "push" or metrics["max_lift_m"] <= 0.008)
        )
        contacted = safe and metrics["robot_contact_steps"] > 0
        contacts += int(contacted)
        car = next(car for car in task["cars"] if car["color"] == task["target_car"])
        bay = next(bay for bay in task["bays"] if bay["color"] == task["target_bay"])
        initial = np.linalg.norm(np.asarray(car["position"][:2]) - bay["center"])
        final = np.linalg.norm(
            np.asarray(outcome["final_car_position"][:2]) - bay["center"]
        )
        moved = (
            float(np.clip((initial - final) / max(initial, 1e-6), 0, 1))
            if contacted
            else 0.0
        )
        progress.append(moved)
        requested_action = (
            metrics["grasped_lift"] if task["action"] == "pick" else contacted
        )
        bay_overlaps += int(
            safe
            and requested_action
            and record["partial_progress"]["bay_overlap_on_table"]
        )
        pickups += int(safe and task["action"] == "pick" and metrics["grasped_lift"])
        push_progress += int(task["action"] == "push" and contacted and moved >= 0.25)
    distances = {
        key: float(np.mean([record["partial_progress"][key] for record in records]))
        for key in (
            "initial_distance_m",
            "final_distance_m",
            "closest_distance_m",
            "distance_closed_m",
        )
    }
    return dict(
        episodes=len(records),
        successes=successes,
        safe_bay_overlaps=bay_overlaps,
        safe_pickups=pickups,
        safe_push_progress=push_progress,
        safe_subgoals=pickups + push_progress,
        safe_contacts=contacts,
        mean_progress=float(np.mean(progress)),
        **{f"mean_{key}": value for key, value in distances.items()},
    )
