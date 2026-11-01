"""Render logged simulator states from a learned rollout without re-integration."""

import argparse
import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np

from so101_car_parking.simulation.parking import load_task


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path)
    parser.add_argument("--video", type=Path, required=True)
    args = parser.parse_args()
    record = json.loads(args.episode.with_suffix(".json").read_text())
    with np.load(args.episode.with_suffix(".npz")) as archive:
        arrays = {key: archive[key] for key in archive.files}
    model, data = load_task(record["task"])
    args.video.parent.mkdir(parents=True, exist_ok=True)
    with (
        mujoco.Renderer(model, height=720, width=960) as renderer,
        imageio.get_writer(
            args.video, fps=20, codec="libx264", macro_block_size=1
        ) as writer,
    ):
        for index, qpos in enumerate(arrays["qpos"]):
            data.qpos[:] = qpos
            data.qvel[:] = 0
            data.time = float(arrays["timestamp"][index])
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera="overview")
            writer.append_data(renderer.render())
    proof = dict(
        method="Recorded-state playback of a learned rollout; no new inference or physical re-integration.",
        frames=len(arrays["qpos"]),
        fps=20,
        task=record["task"],
        source_trace_sha256=hashlib.file_digest(
            args.episode.with_suffix(".npz").open("rb"), "sha256"
        ).hexdigest(),
    )
    args.video.with_suffix(".json").write_text(json.dumps(proof, indent=2) + "\n")
    print(str(args.video), flush=True)


if __name__ == "__main__":
    main()
