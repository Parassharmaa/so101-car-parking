from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from so101_car_parking.policy.pi05 import (
    BASE_ID,
    BASE_REVISION,
    TOKENIZER_REVISION,
    load_bundle as load_bundle,
    make_model as make_model,
    save_bundle as save_bundle,
)

EXECUTION_STEPS = 10


def dataset(root, chunks=True):
    return LeRobotDataset(
        "local/so101-parking",
        root=Path(root),
        video_backend="pyav",
        delta_timestamps={"action": [i / 20 for i in range(50)]} if chunks else None,
    )


def model_reference(kind="pi05"):
    return dict(
        base_model=BASE_ID,
        base_revision=BASE_REVISION,
        tokenizer_revision=TOKENIZER_REVISION,
    )
