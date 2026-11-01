import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import random
import signal
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from vla import dataset, make_model, load_bundle, save_bundle, model_reference


def autocast(device):
    return (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if device == "cuda"
        else nullcontext()
    )


def sample_order(data, count, seed):
    weights = torch.zeros(len(data), dtype=torch.double)
    episodes = data.meta.episodes
    for start, stop in zip(
        episodes["dataset_from_index"], episodes["dataset_to_index"]
    ):
        start, stop = int(start) + 20, int(stop) - 20
        weights[start:stop] = 1 / (stop - start)
    generator = torch.Generator().manual_seed(seed + 10000)
    return torch.multinomial(
        weights, count, replacement=True, generator=generator
    ).tolist()


def validation_loss(policy, processor, loader, device):
    policy.eval()
    losses = []
    devices = [torch.cuda.current_device()] if device == "cuda" else []
    with torch.random.fork_rng(devices=devices), torch.no_grad():
        torch.manual_seed(1337)
        for batch in loader:
            with autocast(device):
                loss, _ = policy(processor(batch))
            losses.append(float(loss))
    policy.train()
    return float(np.mean(losses))


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune a VLA on the parking training split."
    )
    parser.add_argument("--policy-type", choices=["pi05"], default="pi05")
    parser.add_argument("--data", type=Path, default=Path("data/parking-100/lerobot"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--stop-at", type=int)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cuda")
    parser.add_argument("--save-every", type=int, default=5000)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--extend",
        action="store_true",
        help="Extend a pi05 resume while retaining its saved LR schedule.",
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.extend and not args.resume:
        parser.error("--extend requires an explicit checkpoint")
    if args.smoke:
        args.steps, args.batch_size, args.workers, args.save_every = 2, 1, 0, 2
    if (
        args.output.exists()
        and not args.resume
        and any(
            path.name not in {"train.log", "evaluation.log"}
            for path in args.output.iterdir()
        )
    ):
        parser.error(
            "Output is not empty; choose a new directory or an explicit checkpoint to resume."
        )
    args.output.mkdir(parents=True, exist_ok=True)
    stop_requested = False

    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGUSR1, request_stop)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    train = dataset(args.data / "train")
    validation = dataset(args.data / "validation")
    if args.resume:
        previous = json.loads((args.resume / "parking.json").read_text())
        if any(previous[key] != getattr(args, key) for key in ["seed", "batch_size"]):
            raise RuntimeError("Resume seed and batch size must match the checkpoint.")
        if (args.extend and args.steps <= previous["steps"]) or (
            not args.extend and args.steps != previous["steps"]
        ):
            raise RuntimeError(
                "A changed training horizon requires --extend and a larger target."
            )
        policy, pre, post = load_bundle(args.resume, args.device)
        if args.extend and policy.config.type != "pi05":
            raise RuntimeError(
                "Schedule-preserving extension is implemented for pi05 only."
            )
        assets = args.resume / (
            "tokenizer" if policy.config.type == "pi05" else "backbone"
        )
    else:
        policy, pre, post, assets = make_model(train, args.device)
    config = policy.config
    if not args.resume:
        config.scheduler_warmup_steps = min(1000, max(1, args.steps // 20))
        config.scheduler_decay_steps = args.steps
    params = [p for p in policy.parameters() if p.requires_grad]
    optimizer = config.get_optimizer_preset().build(params)
    scheduler = config.get_scheduler_preset().build(
        optimizer, num_training_steps=args.steps
    )
    start_step = 0
    if args.resume:
        state = torch.load(
            args.resume.parent / "training_state.pt",
            map_location="cpu",
            weights_only=False,
        )
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        torch.set_rng_state(state["cpu_rng"])
        if args.device == "cuda":
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        start_step = state["step"]
    order = sample_order(train, args.steps * args.batch_size, args.seed)
    loader = DataLoader(
        Subset(train, order[start_step * args.batch_size :]),
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=args.device == "cuda",
        generator=torch.Generator().manual_seed(args.seed + 20000),
    )
    val_indices = []
    for start, stop in zip(
        validation.meta.episodes["dataset_from_index"],
        validation.meta.episodes["dataset_to_index"],
    ):
        val_indices.extend(
            np.linspace(
                int(start) + 20, int(stop) - 21, 1 if args.smoke else 10, dtype=int
            ).tolist()
        )
    val_loader = DataLoader(
        Subset(validation, val_indices[:2] if args.smoke else val_indices),
        batch_size=args.batch_size,
        num_workers=args.workers,
        generator=torch.Generator().manual_seed(1337),
    )
    metadata = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    metadata.update(model_reference(policy.config.type))
    metadata.update(
        initial_horizon_steps=previous.get("initial_horizon_steps", previous["steps"])
        if args.resume
        else args.steps,
        resumed_step=start_step,
        scheduler_warmup_steps=config.scheduler_warmup_steps,
        scheduler_decay_steps=config.scheduler_decay_steps,
    )
    metadata.update(
        train_episodes=train.num_episodes,
        validation_episodes=validation.num_episodes,
        train_stats_sha256=hashlib.sha256(
            (args.data / "train/meta/stats.json").read_bytes()
        ).hexdigest(),
        trainable_parameters=sum(p.numel() for p in params),
        execution_steps=config.n_action_steps,
        sampling="uniform episodes; skip first/last 20 reset/terminal-hold frames",
        precision="bf16 autocast" if args.device == "cuda" else "fp32",
    )
    (args.output / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata), flush=True)
    before = params[0].detach().clone() if args.smoke else None
    policy.train()
    started = time.monotonic()
    for step, batch in enumerate(loader, start=start_step + 1):
        optimizer.zero_grad(set_to_none=True)
        with autocast(args.device):
            loss, _ = policy(pre(batch))
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite training loss")
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(
            params, config.optimizer_grad_clip_norm
        )
        if not torch.isfinite(gradient):
            raise RuntimeError("Nonfinite gradient")
        optimizer.step()
        scheduler.step()
        if (
            step == start_step + 1
            or step % 100 == 0
            or args.smoke
            or step == args.stop_at
        ):
            record = dict(
                step=step,
                loss=float(loss.detach()),
                grad_norm=float(gradient),
                lr=scheduler.get_last_lr()[0],
                elapsed_seconds=time.monotonic() - started,
            )
            with (args.output / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
        stopping = (
            stop_requested
            or step == args.steps
            or (args.stop_at is not None and step >= args.stop_at)
        )
        if step % args.save_every == 0 or stopping:
            score = validation_loss(policy, pre, val_loader, args.device)
            record = dict(step=step, validation_loss=score)
            with (args.output / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            checkpoint = args.output / "checkpoints" / f"{step:06d}"
            save_bundle(
                checkpoint / "policy", policy, pre, post, assets, metadata | record
            )
            torch.save(
                dict(
                    step=step,
                    optimizer=optimizer.state_dict(),
                    scheduler=scheduler.state_dict(),
                    cpu_rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state_all()
                    if args.device == "cuda"
                    else [],
                ),
                checkpoint / "training_state.pt",
            )
            (checkpoint / ".complete").write_text(str(step) + "\n")
            print(json.dumps(record | {"checkpoint": str(checkpoint)}), flush=True)
        if stopping:
            break
    if args.smoke:
        change = float(torch.max(torch.abs(params[0].detach() - before)))
        if change == 0:
            raise RuntimeError("Smoke run did not update the intended parameters")
        observation = validation[20]
        observation.pop("action", None)
        torch.manual_seed(123)
        policy.eval()
        policy.reset()
        with autocast(args.device):
            original = post(policy.select_action(pre(observation))).detach().cpu()
        restored, restored_pre, restored_post = load_bundle(
            checkpoint / "policy", args.device
        )
        torch.manual_seed(123)
        restored.reset()
        with autocast(args.device):
            recovered = (
                restored_post(restored.select_action(restored_pre(observation)))
                .detach()
                .cpu()
            )
        if (
            original.shape != (1, 6)
            or not torch.isfinite(recovered).all()
            or not torch.allclose(original, recovered, atol=1e-5)
        ):
            raise RuntimeError("Checkpoint reload/inference differs")
        print(
            json.dumps(
                {
                    "smoke_passed": True,
                    "parameter_change": change,
                    "reload_max_error": float((original - recovered).abs().max()),
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
