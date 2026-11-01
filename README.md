# SO-101 car parking

MuJoCo car-placement simulation with a fine-tuned pi0.5 policy.

## Setup

Use Python 3.12 and a CUDA GPU for training or policy inference.
Accept the PaliGemma license on Hugging Face before downloading its tokenizer.

```sh
python -m pip install -r requirements.txt
python parking.py --seed 42
```

## Data and training

```sh
python collect.py --count 100 --output data/base --workers 3
python export_data.py data/base
python collect_more.py --count 200 --output data/extra --workers 3
python export_data.py data/extra
python merge_training.py data/base data/extra data/parking-300
python train_vla.py --data data/parking-300/lerobot --output runs/pi05 --steps 6000 --batch-size 32 --save-every 1000
```

## Evaluation and video

Place the selected checkpoint in `checkpoints/003000/policy`.

```sh
python eval_vla.py checkpoints/003000/policy --collection data/parking-300 --output runs/eval --eager
python replay.py runs/eval/scene_00.json --video captures/example.mp4
```

Robot assets retain their license in `models/so101/LICENSE`.
