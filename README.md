# SO-101 car parking

An SO-101 robot arm moves toy cars to colored parking bays in MuJoCo.
A fine-tuned pi0.5 model controls the arm from a text instruction and two camera images.

The task is not solved. The selected model parked 0 of 20 test cars fully inside a bay.
It put 2 of 20 cars partly inside the correct bay.

![Recorded partial placement](docs/media/scene.jpg)

![Partial placement, recorded playback](docs/media/demo.gif)

The 15-second [preview clip](docs/media/demo.mp4) shows a partial placement. The car is not fully inside the bay.
The [full project video](https://huggingface.co/blazeofchi/so101-car-parking-pi05/resolve/main/media/submission.mp4) shows all results.

## Task

Each scene has 2 to 4 cars and the same number of colored bays.
The instruction names one car color, one bay color and one action: push, or pick and place.

The model receives these inputs:

- the instruction
- an overhead camera image
- a wrist camera image
- the six joint positions

The model gives six joint targets. It predicts 50 targets and the arm does the first 10 at 20 Hz. Then the model looks again.
The model does not receive car or bay positions. Only the scripted teacher and the scorer use them.

## Setup

Use Python 3.12. Training and model runs need a CUDA GPU.
Accept the PaliGemma license on Hugging Face before you download the tokenizer.

```sh
python -m pip install -e .
so101-preview --seed 42
```

## Layout

```text
src/so101_car_parking/
  simulation/   scenes, robot assets and scripted teacher
  data/         demonstration collection and export
  policy/       model loading and training
  evaluation/   model runs, scoring and playback
```

## Data and training

A scripted teacher makes the demonstrations. The commands below make 300 episodes: 280 for training and 20 for testing.

```sh
so101-collect --count 100 --output data/base --workers 3
so101-export data/base
so101-collect-more --count 200 --output data/extra --workers 3
so101-export data/extra
so101-merge data/base data/extra data/parking-300
so101-prepare data/parking-300/lerobot
so101-train --data data/parking-300/lerobot --output runs/pi05 --steps 6000 --batch-size 32 --save-every 1000
```

Training updates only the pi0.5 action expert and its action and time layers. The vision-language part stays frozen.
The selected model is checkpoint 3000.

## Evaluation and video

Download the selected checkpoint and the test scenes. The checkpoint weights use the Gemma license.

```sh
hf download blazeofchi/so101-car-parking-pi05 --local-dir checkpoints/003000/policy
hf download blazeofchi/so101-car-parking-validation --repo-type dataset --include "raw/*.json" --local-dir data/evaluation
so101-evaluate checkpoints/003000/policy --collection data/evaluation --output runs/eval --eager
so101-render-recorded runs/eval/scene_00.json --video captures/example.mp4
```

The evaluation uses only the model. There is no teacher help and no object reset.
The video command plays back the logged simulator states.

[Model](https://huggingface.co/blazeofchi/so101-car-parking-pi05) · [Training data](https://huggingface.co/datasets/blazeofchi/so101-car-parking-train) · [Test data](https://huggingface.co/datasets/blazeofchi/so101-car-parking-validation)

## Results

| Run | Full parks | Partial placements |
| --- | --- | --- |
| 20 test scenes, original settings | 0 / 20 | 2 / 20 |
| 4 new random scenes, eager BF16 | 0 / 4 | 0 / 4 |

A partial placement means the car touches the correct bay area but is not fully inside it.
The 20 test scenes also helped to select the checkpoint, so they are not a fully independent test.

Most runs failed early, before the car came near the bay. Only the 2 partial placements ended near the bay.
In the other 18 runs, the car stayed at least 12 cm from the bay center.
Seven runs moved another car, 3 lifted the car during a push, and 8 stopped at the time limit.

## Credits

- Robot model: [MuJoCo Menagerie SO-101](https://github.com/google-deepmind/mujoco_menagerie/tree/a662de603224c47123b0783821c58ad4402ad3e3/robotstudio_so101), Apache-2.0. The license is in `src/so101_car_parking/simulation/assets/models/so101/LICENSE`.
- Simulator: [MuJoCo](https://github.com/google-deepmind/mujoco), Apache-2.0.
- Policy code and base model: [LeRobot](https://github.com/huggingface/lerobot), Apache-2.0, and [`lerobot/pi05_base`](https://huggingface.co/lerobot/pi05_base), based on pi0.5 by Physical Intelligence.
- Tokenizer: [PaliGemma](https://huggingface.co/google/paligemma-3b-pt-224), Gemma license.

AI coding assistants helped a lot with this project. OpenAI Codex helped write most of the code, run the experiments and draft the documents. Claude helped edit the final text.

The project code uses the Apache-2.0 license.
