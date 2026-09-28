import torch

from vlcot.models.losses import action_loss, masked_bce, observation_loss, semantics_loss, stage_loss


def sample_objective(model, sample, training):
    encoded = [model.encode(observation) for observation in sample["recent"]]
    initial = next((value for value in encoded if value.step == 0), None)
    if initial is None and (sample["prefix_count"] or model.config.initial_frame):
        initial = model.encode(sample["initial"])
    staged = model.config.variant not in {"base", "auxiliary"}
    history = sample["history"] if staged else torch.zeros_like(model.tensor(sample["history"]))
    context = model.context(encoded[-1], history, initial)
    contextual = model.contextual_features(context)
    zero = contextual.sum() * 0
    losses = {name: zero for name in ("stage", "observation", "visibility", "semantics", "auxiliary_state")}
    if staged:
        prefix_count = sample["prefix_count"]
        if prefix_count:
            prefix = model.empty_latents()
            prefix_outputs = []
            for window in sample["prefix_windows"]:
                planned_observation = window["observation"]
                planned_encoded = next((value for value in encoded if value.step == planned_observation.step), None)
                if planned_encoded is None:
                    planned_encoded = initial if planned_observation.step == 0 else model.encode(planned_observation)
                prefix_context = model.context(planned_encoded, window["history"], initial)
                extension, extension_decoded = model.generate_stages(
                    prefix_context, prefix=prefix, stage_limit=window["stage_count"], exact=True)
                prefix = torch.cat((prefix, extension))
                prefix_outputs.append(extension_decoded)
            prefix_decoded = {name: torch.cat([output[name] for output in prefix_outputs]) for name in prefix_outputs[0]}
        else:
            prefix, prefix_decoded = model.empty_latents(), {}
        suffix, suffix_decoded = model.generate_stages(context, prefix, sample["suffix_count"], exact=True)
        decoded = {name: torch.cat((prefix_decoded[name], value), dim=0) if prefix_count else value
                   for name, value in suffix_decoded.items()}
        targets = {name: model.tensor(value) for name, value in sample["stage_targets"].items()}
        losses["stage"] = stage_loss(decoded, targets)
        losses["semantics"] = semantics_loss(decoded, targets)
    else:
        suffix = model.empty_latents()
        if model.config.variant == "auxiliary":
            target = sample["stage_targets"]
            active = sample["prefix_count"]
            predicted = model.auxiliary_state(contextual[:, -1]).squeeze(0)
            losses["auxiliary_state"] = masked_bce(predicted, model.tensor(target["expected"][active]),
                                                    model.tensor(target["valid"][active]))
    if model.config.variant != "base":
        observation = model.estimate(encoded)
        targets = {name: model.tensor(value) for name, value in sample["observation_targets"].items()}
        losses["observation"], losses["visibility"] = observation_loss(observation, targets)
    actions = model.actions(contextual, suffix)
    losses["action"] = action_loss(actions, model.tensor(sample["actions"]), model.tensor(sample["action_mask"]))
    losses["total"] = (losses["action"] + training.stage_weight * (losses["stage"] + losses["auxiliary_state"])
                       + training.observation_weight * losses["observation"]
                       + training.auxiliary_weight * (losses["semantics"] + losses["visibility"]))
    return losses


def batch_objective(model, samples, training):
    losses = [sample_objective(model, sample, training) for sample in samples]
    return {name: torch.stack([sample[name] for sample in losses]).mean() for name in losses[0]}
