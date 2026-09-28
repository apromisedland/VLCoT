import json
import math
import random
from contextlib import nullcontext
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from vlcot.checkpoint import restore_weights, save_checkpoint
from vlcot.data.dataset import TrajectoryDataset, collate_samples
from vlcot.data.io import ActionNormalizer, file_hash, fit_normalizer, read_json, resolve_path, write_json
from vlcot.models.backbone import OpenVLABackbone
from vlcot.models.network import VLCoTModel
from vlcot.models.objective import batch_objective


def observation_label_preflight(dataset):
    known_states = 0
    known_visibility = 0
    for episode in dataset.episodes:
        with h5py.File(resolve_path(dataset.manifest_path, episode["path"]), "r") as data:
            if "observation_targets" not in data:
                raise ValueError("Observation targets are missing; run annotation first")
            known_states += int(np.count_nonzero(data["observation_targets/state_valid"]))
            known_visibility += int(np.count_nonzero(data["observation_targets/visibility_valid"]))
    if dataset.config.variant != "base" and (known_states == 0 or known_visibility == 0):
        raise ValueError("Observation supervision requires visible state labels and independently labeled observability")


def train(config, manifest_path, output_directory, device="cuda", resume=None):
    settings = config.training
    if settings.max_steps is None or settings.max_steps < 1:
        raise ValueError("training.max_steps must be supplied explicitly; the paper does not specify it")
    if settings.micro_batch_size < 1 or settings.effective_batch_size % settings.micro_batch_size:
        raise ValueError("Effective batch size must be divisible by positive micro_batch_size")
    if settings.workers != 0:
        raise ValueError("This deterministic single-GPU reference trainer currently requires workers=0")
    if min(settings.save_every, settings.validate_every, settings.validation_batches) < 1:
        raise ValueError("Checkpoint and validation intervals must be positive")
    if settings.bf16 and (not device.startswith("cuda") or not torch.cuda.is_bf16_supported()):
        raise ValueError("BF16 training requires a supported CUDA device; set bf16=false for FP32")
    random.seed(settings.seed)
    np.random.seed(settings.seed)
    torch.manual_seed(settings.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(settings.seed)
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()) and resume is None:
        raise ValueError("Training requires a fresh output directory unless --resume is specified")
    normalizer = (ActionNormalizer(read_json(Path(resume) / "normalizer.json")) if resume else
                  fit_normalizer(manifest_path, settings.recovery_data, seed=settings.seed))
    if normalizer.statistics["manifest_sha256"] != file_hash(manifest_path):
        raise ValueError("Resume dataset differs from the normalization/training manifest")
    training_data = TrajectoryDataset(manifest_path, config.model, normalizer, include_recovery=settings.recovery_data,
                                      seed=settings.seed)
    validation_data = TrajectoryDataset(manifest_path, config.model, normalizer, split="validation",
                                        include_recovery=settings.recovery_data, seed=settings.seed)
    observation_label_preflight(training_data)
    observation_label_preflight(validation_data)
    backbone = OpenVLABackbone(config.model, device=device, training=True)
    model = VLCoTModel(backbone, config.model).to(device)
    if settings.gradient_checkpointing:
        backbone.enable_gradient_checkpointing()
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=settings.learning_rate, weight_decay=settings.weight_decay)

    def schedule(step):
        warmup = min(settings.warmup_steps, settings.max_steps)
        if step < warmup:
            return (step + 1) / max(1, warmup)
        fraction = (step - warmup) / max(1, settings.max_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1, fraction)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    step = 0
    epoch = 0
    cursor = 0
    best_validation = float("inf")
    if resume:
        saved_config = read_json(Path(resume) / "config.json")
        if saved_config != config.to_dict():
            raise ValueError("Resume requires the original full configuration, including max_steps")
        restore_weights(model, resume)
        saved = torch.load(Path(resume) / "training_state.pt", map_location="cpu", weights_only=True)
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        step, epoch, cursor = saved["step"], saved["epoch"], saved["cursor"]
        best_validation = saved["best_validation"]
        torch.set_rng_state(saved["torch_rng"])
        if device.startswith("cuda"):
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
    write_json(output / "config.json", config.to_dict())
    write_json(output / "normalizer.json", normalizer.statistics)
    validation_loader = DataLoader(validation_data, batch_size=settings.micro_batch_size, collate_fn=collate_samples)
    accumulation = settings.effective_batch_size // settings.micro_batch_size
    iterator = None

    def autocast():
        return torch.autocast("cuda", dtype=torch.bfloat16) if settings.bf16 else nullcontext()

    def checkpoint(label):
        save_checkpoint(output / label, model, config, normalizer, {
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "step": step, "epoch": epoch, "cursor": cursor, "best_validation": best_validation,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if device.startswith("cuda") else [],
        }, {"optimizer_steps": step, "manifest_sha256": file_hash(manifest_path), "seed": settings.seed})

    model.train()
    while step < settings.max_steps:
        optimizer.zero_grad(set_to_none=True)
        metrics = {}
        for _ in range(accumulation):
            while True:
                if iterator is None:
                    training_data.set_epoch(epoch)
                    generator = torch.Generator().manual_seed(settings.seed + epoch)
                    loader = DataLoader(training_data, batch_size=settings.micro_batch_size, shuffle=True,
                                        generator=generator, drop_last=True, collate_fn=collate_samples)
                    if not len(loader):
                        raise ValueError("Training split is smaller than micro_batch_size")
                    iterator = iter(loader)
                    for skipped in range(cursor):
                        next(iterator)
                try:
                    batch = next(iterator)
                    cursor += 1
                    break
                except StopIteration:
                    epoch += 1
                    cursor = 0
                    iterator = None
            with autocast():
                losses = batch_objective(model, batch, settings)
            if not all(torch.isfinite(value).item() for value in losses.values()):
                raise FloatingPointError(f"Non-finite loss at optimizer step {step}")
            (losses["total"] / accumulation).backward()
            for name, value in losses.items():
                metrics[name] = metrics.get(name, 0.0) + float(value.detach()) / accumulation
        torch.nn.utils.clip_grad_norm_(parameters, settings.max_grad_norm, error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        step += 1
        entry = {"step": step, "epoch": epoch, "learning_rate": scheduler.get_last_lr()[0], "train": metrics}
        if step % settings.validate_every == 0 or step == settings.max_steps:
            model.eval()
            measurements = []
            with torch.no_grad():
                for batch_index, batch in enumerate(validation_loader):
                    if batch_index >= settings.validation_batches:
                        break
                    with autocast():
                        losses = batch_objective(model, batch, settings)
                    measurements.append(float(losses["total"]))
            validation = float(np.mean(measurements))
            if not math.isfinite(validation):
                raise FloatingPointError("Non-finite validation loss")
            entry["validation_loss"] = validation
            if validation < best_validation:
                best_validation = validation
                checkpoint(f"best-{step:08d}")
                write_json(output / "best.json", {"checkpoint": f"best-{step:08d}", "validation_loss": validation})
            model.train()
        with (output / "training.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(entry, allow_nan=False) + "\n")
        print(json.dumps(entry, allow_nan=False), flush=True)
        if step % settings.save_every == 0 or step == settings.max_steps:
            checkpoint(f"step-{step:08d}")
    return output
