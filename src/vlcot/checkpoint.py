from pathlib import Path

import torch

from vlcot.config import config_from_dict
from vlcot.data.io import ActionNormalizer, file_hash, read_json, write_json


def save_checkpoint(directory, model, config, normalizer, training_state=None, metadata=None):
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=False)
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    weights = {name: value.detach().cpu() for name, value in model.state_dict().items()
               if name in trainable or not name.startswith("backbone.")}
    torch.save(weights, destination / "trainable.pt")
    write_json(destination / "config.json", config.to_dict())
    write_json(destination / "normalizer.json", normalizer.statistics)
    write_json(destination / "metadata.json", {
        "format_version": 1, "backbone_included": False, "weight_sha256": file_hash(destination / "trainable.pt"),
        "config_sha256": file_hash(destination / "config.json"),
        "normalizer_sha256": file_hash(destination / "normalizer.json"),
        **(metadata or {}),
    })
    if training_state is not None:
        torch.save(training_state, destination / "training_state.pt")


def restore_weights(model, directory):
    directory = Path(directory)
    metadata = read_json(directory / "metadata.json")
    if metadata["format_version"] != 1 or file_hash(directory / "trainable.pt") != metadata["weight_sha256"]:
        raise ValueError("Checkpoint format or checksum mismatch")
    for filename, key in (("config.json", "config_sha256"), ("normalizer.json", "normalizer_sha256")):
        if file_hash(directory / filename) != metadata[key]:
            raise ValueError(f"Checkpoint {filename} checksum mismatch")
    state = torch.load(directory / "trainable.pt", map_location="cpu", weights_only=True)
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    if not trainable.issubset(state):
        raise ValueError(f"Checkpoint is missing trainable parameters: {sorted(trainable - state.keys())}")
    result = model.load_state_dict(state, strict=False)
    if result.unexpected_keys:
        raise ValueError(f"Unexpected checkpoint tensors: {result.unexpected_keys}")


def load_policy_model(directory, device="cuda", backbone_path=None):
    from vlcot.models.backbone import OpenVLABackbone
    from vlcot.models.network import VLCoTModel

    directory = Path(directory)
    config = config_from_dict(read_json(directory / "config.json"))
    if backbone_path is not None:
        config.model.backbone = str(Path(backbone_path).resolve())
    backbone = OpenVLABackbone(config.model, device=device)
    model = VLCoTModel(backbone, config.model).to(device)
    restore_weights(model, directory)
    if device.startswith("cuda"):
        model.to(dtype=torch.bfloat16)
    model.eval()
    normalizer = ActionNormalizer(read_json(directory / "normalizer.json"))
    if len(normalizer.low) != config.model.action_dim:
        raise ValueError("Checkpoint action normalization dimensions differ from its model")
    return model, config, normalizer
