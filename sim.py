from pathlib import Path

import mujoco
import mujoco.viewer


def load_scene(model=None):
    if model is None:
        model = mujoco.MjModel.from_xml_path(str(Path(__file__).with_name("scene.xml")))
    data = mujoco.MjData(model)
    home = {
        "shoulder_pan": 0,
        "shoulder_lift": -0.7,
        "elbow_flex": 1.4,
        "wrist_flex": -0.7,
        "wrist_roll": 0,
        "gripper": 0.1,
    }
    for name, angle in home.items():
        data.joint(name).qpos[0] = angle
        data.actuator(name).ctrl[0] = angle
    mujoco.mj_forward(model, data)
    return model, data


def main():
    model, data = load_scene()
    mujoco.viewer.launch(model, data)


if __name__ == "__main__":
    main()
