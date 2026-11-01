import hashlib
import json
import os
from pathlib import Path
import shutil

from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file
import torch
from transformers.initialization import no_init_weights

from lerobot.configs import FeatureType
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.utils.feature_utils import dataset_to_policy_features


BASE_ID = "lerobot/pi05_base"
BASE_REVISION = "22c30e5b34f9e7ab738b1b34fc2fc5ecfe2d28f0"
TOKENIZER_ID = "google/paligemma-3b-pt-224"
TOKENIZER_REVISION = "35e4f46485b4d07967e7e9935bc3786aad50687c"
CAMERAS = {
    "observation.images.overhead": "observation.images.base_0_rgb",
    "observation.images.wrist": "observation.images.left_wrist_0_rgb",
}


def tokenizer_assets():
    if os.environ.get("PI05_TOKENIZER_DIR"):
        path = Path(os.environ["PI05_TOKENIZER_DIR"]).resolve()
        if not (path / "tokenizer_config.json").exists():
            raise RuntimeError("The supplied tokenizer directory is incomplete")
        return path
    return Path(
        snapshot_download(
            TOKENIZER_ID,
            revision=TOKENIZER_REVISION,
            allow_patterns=["*.json", "*.model", "*.jinja"],
        )
    )


def frozen_digest(policy):
    digest = hashlib.sha256()
    for name, value in policy.named_parameters():
        if not value.requires_grad:
            tensor = value.detach().cpu().contiguous()
            digest.update(f"{name}:{tensor.dtype}:{list(tensor.shape)}".encode())
            digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def load_base(config, device):
    if device != "cuda":
        raise RuntimeError("This Pi05 adapter requires CUDA with bfloat16 autocast")
    base = Path(
        snapshot_download(
            BASE_ID,
            revision=BASE_REVISION,
            allow_patterns=["*.json", "model.safetensors"],
        )
    )
    config.device = "cpu"
    # Reconstruct the original base dtypes before changing the adaptation scope.
    config.train_expert_only = config.freeze_vision_encoder = True
    with no_init_weights():
        policy = PI05Policy(config)
    # Actions use action_out_proj, not the expert's unused text-output head.
    policy.model.paligemma_with_expert.gemma_expert.lm_head.requires_grad_(False)
    for value in policy.parameters():
        if value.requires_grad:
            value.data = value.data.float()
    # The library's from_pretrained catches loading failures. Fail closed here.
    state = policy._fix_pytorch_state_dict_keys(
        load_file(base / "model.safetensors"), config
    )
    state = {
        key if key.startswith("model.") else f"model.{key}": value
        for key, value in state.items()
    }
    policy.load_state_dict(state, strict=True)
    loaded = policy.state_dict()
    for key, value in state.items():
        if not torch.equal(loaded[key], value.to(loaded[key].dtype)):
            raise RuntimeError(f"Pretrained weights differ at {key}")
    del state, loaded
    policy._parking_frozen_digest = frozen_digest(policy)
    config.device = device
    return policy.to(device).eval(), base


def make_model(data, device):
    assets = tokenizer_assets()  # Check gated access before downloading large weights.
    config = PI05Config.from_pretrained(BASE_ID, revision=BASE_REVISION)
    config.device = "cpu"
    config.push_to_hub = False
    config.train_expert_only = True
    config.freeze_vision_encoder = True
    config.gradient_checkpointing = True
    config.compile_model = os.environ.get("PI05_COMPILE", "0") == "1"
    config.compile_mode = os.environ.get("PI05_COMPILE_MODE", "default")
    config.dtype = "bfloat16"
    config.use_amp = device == "cuda"
    config.n_action_steps = 10
    config.empty_cameras = 1
    features = {
        CAMERAS.get(key, key): value
        for key, value in dataset_to_policy_features(data.meta.features).items()
    }
    config.input_features = {
        key: value
        for key, value in features.items()
        if value.type != FeatureType.ACTION
    }
    config.output_features = {
        key: value
        for key, value in features.items()
        if value.type == FeatureType.ACTION
    }
    stats = {CAMERAS.get(key, key): value for key, value in data.meta.stats.items()}
    for key in ("observation.state", "action"):
        if not all(name in stats[key] for name in ("q01", "q99")):
            raise RuntimeError(f"Training-only quantiles missing for {key}")
    policy, base = load_base(config, device)
    pre, post = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=str(base),
        preprocessor_overrides={
            "rename_observations_processor": {"rename_map": CAMERAS},
            "normalizer_processor": {
                "features": features,
                "norm_map": config.normalization_mapping,
                "stats": stats,
            },
            "tokenizer_processor": {"tokenizer_name": str(assets)},
            "device_processor": {"device": device},
        },
        postprocessor_overrides={
            "unnormalizer_processor": {
                "features": config.output_features,
                "norm_map": config.normalization_mapping,
                "stats": stats,
            }
        },
    )
    return policy, pre, post, assets


def save_bundle(path, policy, pre, post, assets, metadata):
    path = Path(path)
    if frozen_digest(policy) != policy._parking_frozen_digest:
        raise RuntimeError(
            "Frozen parameters changed; cannot save a compact checkpoint"
        )
    path.mkdir(parents=True, exist_ok=True)
    policy.config.save_pretrained(path)
    save_file(
        {
            name: value.detach().float().cpu().contiguous()
            for name, value in policy.named_parameters()
            if value.requires_grad
        },
        path / "learned_weights.safetensors",
    )
    pre.save_pretrained(path, config_filename="policy_preprocessor.json")
    post.save_pretrained(path, config_filename="policy_postprocessor.json")
    shutil.copytree(assets, path / "tokenizer", dirs_exist_ok=True)
    metadata = metadata | dict(
        adaptation=getattr(policy, "_parking_adaptation", {"kind": "expert"}),
        frozen_parameters_sha256=policy._parking_frozen_digest,
        learned_weights_sha256=hashlib.file_digest(
            (path / "learned_weights.safetensors").open("rb"), "sha256"
        ).hexdigest(),
    )
    (path / "parking.json").write_text(json.dumps(metadata, indent=2) + "\n")


def load_bundle(path, device):
    path = Path(path)
    metadata = json.loads((path / "parking.json").read_text())
    config = PI05Config.from_pretrained(path)
    config.push_to_hub = False
    if metadata.get("adaptation", {"kind": "expert"})["kind"] != "expert":
        raise RuntimeError("Use the selected expert-only checkpoint")
    policy, _ = load_base(config, device)
    if metadata["frozen_parameters_sha256"] != policy._parking_frozen_digest:
        raise RuntimeError("Frozen base reconstruction differs")
    weights = path / "learned_weights.safetensors"
    if (
        hashlib.file_digest(weights.open("rb"), "sha256").hexdigest()
        != metadata["learned_weights_sha256"]
    ):
        raise RuntimeError("Learned weights checksum differs")
    learned = load_file(weights)
    if set(learned) != {
        name for name, value in policy.named_parameters() if value.requires_grad
    }:
        raise RuntimeError("Trainable parameters differ from the checkpoint")
    policy.load_state_dict(learned, strict=False)
    pre, post = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=str(path),
        preprocessor_overrides={
            "device_processor": {"device": device},
            "tokenizer_processor": {
                "tokenizer_name": str((path / "tokenizer").resolve())
            },
        },
    )
    return policy, pre, post
