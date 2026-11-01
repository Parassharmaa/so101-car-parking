import argparse
from copy import deepcopy
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import mujoco.viewer
import numpy as np

from so101_car_parking.simulation.sim import load_scene


ROOT = Path(__file__).resolve().parent / "assets"
COLORS = {
    "red": (0.85, 0.12, 0.10),
    "blue": (0.10, 0.40, 0.85),
    "green": (0.12, 0.65, 0.28),
    "yellow": (0.95, 0.70, 0.08),
}
BAY_SIZE = (0.11, 0.076)
ROBOT_MOUNT = (0.12, 0, 0.10)


def make_task(seed, num_cars=2, action="push", car_color=None, bay_color=None):
    if num_cars not in (2, 3, 4) or action not in ("push", "pick"):
        raise ValueError("Use 2–4 cars and action 'push' or 'pick'.")
    rng = np.random.default_rng(seed)
    car_colors = rng.permutation(list(COLORS))[:num_cars].tolist()
    bay_colors = rng.permutation(list(COLORS))[:num_cars].tolist()
    starts = rng.permutation([-0.12, -0.04, 0.04, 0.12])[:num_cars]
    cars = [
        {
            "color": color,
            "position": [
                round(float(rng.uniform(0.225, 0.25)), 4),
                round(float(y + rng.uniform(-0.004, 0.004)), 4),
                0.014,
            ],
        }
        for color, y in zip(car_colors, starts)
    ]
    anchors = [
        (0.12, -0.17),
        (0.245, -0.19),
        (0.37, -0.17),
        (0.37, 0),
        (0.37, 0.17),
        (0.245, 0.19),
        (0.12, 0.17),
    ]
    bays = []
    occupied = [
        (np.array(car["position"][:2]), np.array([0.045, 0.028])) for car in cars
    ]
    half_bay = np.array(BAY_SIZE) / 2
    for color in bay_colors:
        for _ in range(200):
            center = np.round(
                np.array(anchors[rng.integers(len(anchors))])
                + rng.uniform(-0.015, 0.015, 2),
                4,
            )
            if all(
                np.any(np.abs(center - other) >= half_bay + half_size + 0.012)
                for other, half_size in occupied
            ):
                bays.append({"color": color, "center": center.tolist()})
                occupied.append((center, half_bay))
                break
        else:
            raise RuntimeError("Could not place separated parking bays.")
    car_color = car_color or str(rng.choice(car_colors))
    bay_color = bay_color or str(rng.choice(bay_colors))
    if car_color not in car_colors or bay_color not in bay_colors:
        raise ValueError(f"Available cars: {car_colors}; available bays: {bay_colors}")
    verb = "Push" if action == "push" else "Pick up"
    destination = "into" if action == "push" else "and place it in"
    return {
        "seed": seed,
        "action": action,
        "target_car": car_color,
        "target_bay": bay_color,
        "instruction": f"{verb} the {car_color} car {destination} the {bay_color} parking bay. "
        "Leave the other cars where they are.",
        "cars": cars,
        "bays": bays,
        "bay_size": list(BAY_SIZE),
        "robot_mount": list(ROBOT_MOUNT),
    }


def load_task(task):
    root = ET.parse(ROOT / "scene.xml").getroot()
    root.find("include").set("file", str(ROOT / "models/so101/so101.xml"))
    root.find("compiler").set("meshdir", str(ROOT / "models/so101/assets"))
    world = root.find("worldbody")
    mount = task["robot_mount"]
    ET.SubElement(
        world,
        "geom",
        name="pedestal",
        type="box",
        pos=f"{mount[0]} {mount[1]} {mount[2] / 2}",
        size=f"0.035 0.025 {mount[2] / 2}",
        rgba="0.2 0.23 0.27 1",
    )
    template = world.find("body[@name='car']")
    world.remove(template)
    for car in task["cars"]:
        body = deepcopy(template)
        color = car["color"]
        for element in body.iter():
            if "name" in element.attrib:
                element.set("name", f"{color}_{element.get('name')}")
        body.set("pos", " ".join(map(str, car["position"])))
        body.find(f"geom[@name='{color}_chassis']").set(
            "rgba", " ".join(map(str, (*COLORS[color], 1)))
        )
        world.append(body)

    for bay in task["bays"]:
        color = bay["color"]
        x, y = bay["center"]
        hx, hy = np.array(task["bay_size"]) / 2
        rectangles = [
            (0, 0, hx, hy, tuple(0.55 * c + 0.22 for c in COLORS[color])),
            (-hx, 0, 0.0015, hy, COLORS[color]),
            (hx, 0, 0.0015, hy, COLORS[color]),
            (0, -hy, hx, 0.0015, COLORS[color]),
            (0, hy, hx, 0.0015, COLORS[color]),
            (-0.008, 0, 0.001, 0.012, (1, 1, 1)),
            (0, 0.012, 0.008, 0.001, (1, 1, 1)),
            (0, 0, 0.008, 0.001, (1, 1, 1)),
            (0.008, 0.006, 0.001, 0.006, (1, 1, 1)),
        ]
        for i, (dx, dy, sx, sy, rgb) in enumerate(rectangles):
            ET.SubElement(
                world,
                "geom",
                name=f"{color}_bay_{i}",
                type="box",
                pos=f"{x + dx} {y + dy} {0.0002 + i * 0.00005}",
                size=f"{sx} {sy} 0.00005",
                rgba=" ".join(map(str, (*rgb, 1))),
                contype="0",
                conaffinity="0",
            )
    spec = mujoco.MjSpec.from_string(ET.tostring(root, encoding="unicode"))
    for name, position, size in [
        ("gripper", [-0.0075, 0, -0.092], [0.001, 0.014, 0.009]),
        ("moving_jaw_so101_v1", [-0.0105, -0.0686, 0.0192], [0.001, 0.009, 0.014]),
    ]:
        spec.body(name).add_geom(
            name=f"{name}_pad",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=position,
            size=size,
            mass=0.001,
            rgba=[0.12, 0.12, 0.13, 1],
            friction=[1.2, 0.005, 0.0001],
            condim=6,
            priority=2,
            solref=[0.01, 1],
        )
    model = spec.compile()
    model, data = load_scene(model)
    model.body_pos[model.body("base").id] = mount
    data.joint("shoulder_pan").qpos[0] = 1.85
    data.actuator("shoulder_pan").ctrl[0] = 1.85
    mujoco.mj_forward(model, data)
    return model, data


def main():
    parser = argparse.ArgumentParser(description="Seeded car-parking task scenes.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cars", type=int, choices=[2, 3, 4], default=2)
    parser.add_argument("--action", choices=["push", "pick"], default="push")
    parser.add_argument("--car-color", choices=COLORS)
    parser.add_argument("--bay-color", choices=COLORS)
    parser.add_argument(
        "--render",
        type=Path,
        help="Save a PNG and task JSON instead of opening the viewer.",
    )
    parser.add_argument(
        "--camera", default="overhead", choices=["overview", "overhead", "wrist_cam"]
    )
    args = parser.parse_args()
    try:
        task = make_task(
            args.seed, args.cars, args.action, args.car_color, args.bay_color
        )
    except ValueError as error:
        parser.error(str(error))
    model, data = load_task(task)
    print(task["instruction"])
    if args.render:
        import imageio.v2 as imageio

        args.render.parent.mkdir(parents=True, exist_ok=True)
        mujoco.mj_step(model, data, nstep=round(1 / model.opt.timestep))
        with mujoco.Renderer(model, height=720, width=960) as renderer:
            renderer.update_scene(data, camera=args.camera)
            imageio.imwrite(args.render, renderer.render())
        args.render.with_suffix(".json").write_text(json.dumps(task, indent=2) + "\n")
    else:
        mujoco.viewer.launch(model, data)


if __name__ == "__main__":
    main()
