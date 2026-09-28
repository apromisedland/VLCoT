from collections import deque

import h5py
import numpy as np
import torch

from vlcot.checkpoint import load_policy_model
from vlcot.control.verification import Decision, StageVerifier
from vlcot.data.io import file_hash, read_json, resolve_path, write_json
from vlcot.evaluation.metrics import verification_f1
from vlcot.types import Observation, StagePlan, StateEstimate, VerifiedHistory


@torch.inference_mode()
def diagnose(checkpoint, diagnostic_manifest, output_file, device="cuda", backbone_path=None):
    model, config, _ = load_policy_model(checkpoint, device, backbone_path)
    if config.model.variant in {"base", "auxiliary"}:
        write_json(output_file, {"f1": None, "reason": "variant_has_no_latent_stage_verification_interface"})
        return
    manifest = read_json(diagnostic_manifest)
    if not manifest.get("cases") or not manifest.get("provenance", {}).get("source"):
        raise ValueError("An independently labeled fixed diagnostic manifest is required")
    predictions, labels, records = [], [], []
    for case in manifest["cases"]:
        indices = case["observation_indices"]
        if not indices or any(left >= right for left, right in zip(indices, indices[1:])):
            raise ValueError("Diagnostic observation indices must be strictly increasing")
        with h5py.File(resolve_path(diagnostic_manifest, case["trajectory"]), "r") as data:
            def observation(frame):
                return Observation({name: np.asarray(data[f"images/{name}"][frame]) for name in config.model.camera_names},
                                   np.asarray(data["proprio"][frame]), case["instruction"], frame)

            planning_frame = int(case.get("planning_frame", 0))
            initial = model.encode(observation(0))
            start = initial if planning_frame == 0 else model.encode(observation(planning_frame))
            empty_history = np.zeros(2 * len(config.model.predicate_names), dtype=np.float32)
            latents, decoded = model.generate_stages(model.context(start, empty_history, initial))
            stage_index = int(case["stage_index"])
            predicted = False
            if stage_index < 0:
                raise ValueError("Diagnostic stage index must be nonnegative")
            if stage_index < len(latents):
                stage_plan = StagePlan(latents[stage_index:stage_index + 1], [model.stage_specs(decoded)[stage_index]])
                verifier = StageVerifier(config.verification)
                recent = deque(maxlen=config.model.observation_window)
                for frame in indices:
                    if frame < planning_frame:
                        raise ValueError("Diagnostic observations cannot precede the planning observation")
                    recent.append(model.encode(observation(frame)))
                    values = model.estimate(list(recent))
                    estimate = StateEstimate(values["state"].sigmoid().float().cpu().numpy(),
                                             values["observable"].sigmoid().float().cpu().numpy(), frame)
                    decision = verifier.check(stage_plan, VerifiedHistory(), estimate)
                    predicted = decision.decision == Decision.COMPLETE
            label = case["completed"]
            if not isinstance(label, bool):
                raise ValueError("Diagnostic completed labels must be independent booleans")
            predictions.append(predicted)
            labels.append(label)
            records.append({"id": case["id"], "prediction": predicted, "completed": label})
    write_json(output_file, {**verification_f1(predictions, labels), "cases": records,
                            "diagnostic_manifest_sha256": file_hash(diagnostic_manifest)})
