import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from so101_car_parking.simulation.parking import load_task
from so101_car_parking.simulation.teacher import DT


def replay(model, data, arrays):
    for i, (expected, action) in enumerate(zip(arrays["qpos"], arrays["action"])):
        error = float(np.max(np.abs(data.qpos - expected)))
        if error > 1e-7:
            raise RuntimeError(f"Replay diverged at frame {i}: {error}")
        yield i
        data.ctrl[:] = action
        mujoco.mj_step(model, data, nstep=round(DT / model.opt.timestep))
    if np.max(np.abs(data.qpos - arrays["final_qpos"])) > 1e-7:
        raise RuntimeError("Final replay state differs")


def main():
    parser = argparse.ArgumentParser(
        description="Replay recorded joint commands and optionally save video."
    )
    parser.add_argument("episode", type=Path)
    parser.add_argument("--video", type=Path)
    parser.add_argument(
        "--camera", default="overview", choices=["overview", "overhead", "wrist_cam"]
    )
    args = parser.parse_args()
    metadata = json.loads(args.episode.with_suffix(".json").read_text())
    with np.load(args.episode.with_suffix(".npz")) as archive:
        arrays = {key: archive[key] for key in archive.files}
    model, data = load_task(metadata["task"])
    renderer = writer = None
    if args.video:
        import imageio.v2 as imageio

        args.video.parent.mkdir(parents=True, exist_ok=True)
        renderer = mujoco.Renderer(model, height=720, width=960)
        writer = imageio.get_writer(
            args.video, fps=round(1 / DT), codec="libx264", macro_block_size=1
        )
    try:
        for _ in replay(model, data, arrays):
            if writer:
                renderer.update_scene(data, camera=args.camera)
                writer.append_data(renderer.render())
    finally:
        if writer:
            writer.close()
            renderer.close()
    print(metadata["task"]["instruction"])
    print(f"Replayed {len(arrays['action'])} frames with state error <= 1e-7.")


if __name__ == "__main__":
    main()
