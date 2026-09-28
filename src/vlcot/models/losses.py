import torch
from torch.nn import functional as functional


def masked_mean(values: torch.Tensor, mask: torch.Tensor):
    valid = mask.to(dtype=torch.bool).expand_as(values)
    safe = torch.where(valid, values, torch.zeros_like(values))
    return safe.sum() / valid.sum().clamp_min(1)


def masked_bce(logits, targets, mask):
    valid = mask.to(dtype=torch.bool).expand_as(logits)
    safe_targets = torch.where(valid, targets, torch.zeros_like(targets))
    safe_logits = torch.where(valid, logits, torch.zeros_like(logits))
    return masked_mean(functional.binary_cross_entropy_with_logits(safe_logits, safe_targets, reduction="none"), valid)


def stage_loss(decoded, targets):
    valid = targets["valid"].bool()
    safe_target = torch.where(valid, targets["expected"], torch.zeros_like(targets["expected"]))
    elementwise = functional.binary_cross_entropy_with_logits(decoded["expected"], safe_target, reduction="none")
    per_stage = torch.where(valid, elementwise, torch.zeros_like(elementwise)).sum(-1) / valid.sum(-1).clamp_min(1)
    return masked_mean(per_stage, valid.any(-1))


def semantics_loss(decoded, targets):
    losses = []
    for name in ("relevant", "required", "releases"):
        losses.append(masked_bce(decoded[name], targets[name], targets["contract_valid"]))
    losses.append(masked_bce(decoded["required_values"], targets["required_values"],
                             targets["contract_valid"] * targets["required"]))
    losses.append(masked_bce(decoded["stop"], targets["stop"], targets["stop_valid"]))
    return torch.stack(losses).mean()


def observation_loss(predicted, targets):
    state = masked_bce(predicted["state"], targets["state"], targets["state_valid"])
    visibility = masked_bce(predicted["observable"], targets["observable"], targets["visibility_valid"])
    return state, visibility


def action_loss(prediction, target, mask):
    safe_target = torch.where(mask.bool(), target, torch.zeros_like(target))
    safe_prediction = torch.where(mask.bool(), prediction, torch.zeros_like(prediction))
    return masked_mean((safe_prediction - safe_target).abs(), mask)
