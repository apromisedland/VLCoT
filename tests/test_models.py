import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch import nn

from vlcot.checkpoint import restore_weights, save_checkpoint
from vlcot.config import Config, ModelConfig
from vlcot.data.io import ActionNormalizer
from vlcot.models.losses import masked_bce, stage_loss
from vlcot.models.network import VLCoTModel
from vlcot.types import Observation

pytestmark = pytest.mark.model


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_size = 32
        self.images = nn.Linear(3, self.hidden_size)
        self.text = nn.Embedding(257, self.hidden_size)
        layer = nn.TransformerEncoderLayer(self.hidden_size, 4, dim_feedforward=64, dropout=0, batch_first=True)
        self.decoder = nn.TransformerEncoder(layer, 1)

    def visual_embeddings(self, image):
        values = torch.as_tensor(image, dtype=torch.float32).mean((0, 1)) / 255
        return self.images(values).view(1, 1, -1)

    def text_embeddings(self, instruction):
        return self.text(torch.tensor([list(instruction.encode("utf-8"))]))

    def bos_embedding(self):
        return self.text(torch.tensor([[256]]))

    def hidden(self, embeddings):
        length = embeddings.shape[1]
        mask = torch.ones((length, length), dtype=torch.bool).triu(1)
        return self.decoder(embeddings, mask=mask)


def model(variant="full"):
    torch.manual_seed(7)
    config = ModelConfig(action_dim=2, proprio_dim=2, predicate_names=["held", "placed"], camera_names=["camera"],
                          head_dim=32, attention_heads=4, vectors_per_stage=2, max_stages=2, variant=variant)
    return VLCoTModel(TinyBackbone(), config)


def observation(step=0):
    return Observation({"camera": np.full((8, 8, 3), 120, dtype=np.uint8)}, np.zeros(2, np.float32), "place", step)


@pytest.mark.parametrize("variant,expected_gradient", [("full", True), ("no_alignment", False)])
def test_alignment_controls_gradient_route(variant, expected_gradient):
    network = model(variant)
    encoded = network.encode(observation())
    context = network.context(encoded, np.zeros(4))
    _, decoded = network.generate_stages(context, stage_limit=1, exact=True)
    loss = stage_loss(decoded, {"expected": torch.ones(1, 2), "valid": torch.ones(1, 2)})
    loss.backward()
    assert network.stage_decoder.heads["expected"].weight.grad.abs().sum() > 0
    gradient = network.latent_start.grad
    assert (gradient is not None and gradient.abs().sum() > 0) == expected_gradient


def test_actions_depend_on_latents():
    network = model().eval()
    context = network.contextual_features(network.context(network.encode(observation()), np.zeros(4)))
    before = network.actions(context, torch.zeros(1, 2, 32))
    after = network.actions(context, torch.ones(1, 2, 32))
    assert before.shape == (8, 2)
    assert not torch.allclose(before, after)


def test_observation_estimator_does_not_read_latent_predictions():
    network = model().eval()
    encoded = [network.encode(observation())]
    before = network.estimate(encoded)["state"].detach().clone()
    with torch.no_grad():
        network.latent_start.add_(1000)
        network.stage_decoder.heads["expected"].weight.mul_(1000)
    after = network.estimate(encoded)["state"]
    assert torch.equal(before, after)


def test_invalid_targets_are_masked_before_bce():
    logits = torch.tensor([1.0, 2.0], requires_grad=True)
    loss = masked_bce(logits, torch.tensor([float("nan"), float("nan")]), torch.zeros(2))
    loss.backward()
    assert loss.item() == 0
    assert torch.equal(logits.grad, torch.zeros(2))


def test_action_gradient_is_preserved_without_stage_alignment():
    network = model("no_alignment")
    context = network.context(network.encode(observation()), np.zeros(4))
    latents, _ = network.generate_stages(context, stage_limit=1, exact=True)
    action = network.actions(network.contextual_features(context), latents)
    action.abs().mean().backward()
    assert network.latent_start.grad is not None
    assert network.latent_start.grad.abs().sum() > 0


def test_variable_suffix_objective_supports_optimizer_update():
    from vlcot.config import TrainConfig
    from vlcot.models.objective import batch_objective

    network = model()
    targets = {name: np.ones((2, 2), np.float32) for name in
               ("expected", "valid", "relevant", "required", "required_values", "contract_valid")}
    targets["releases"] = np.zeros((2, 2), np.float32)
    targets["stop"] = np.array([0, 1], np.float32)
    targets["stop_valid"] = np.array([0, 1], np.float32)
    sample = {"initial": observation(0), "recent": [observation(0), observation(4)], "history": np.zeros(4),
              "prefix_count": 1, "suffix_count": 1, "stage_targets": targets,
              "prefix_windows": [{"observation": observation(0), "history": np.zeros(4), "stage_count": 1}],
              "actions": np.zeros((8, 2), np.float32), "action_mask": np.ones((8, 2), np.float32),
              "observation_targets": {name: np.ones(2, np.float32) for name in
                                      ("state", "state_valid", "observable", "visibility_valid")}}
    optimizer = torch.optim.AdamW(network.parameters(), lr=1e-3)
    previous = network.latent_start.detach().clone()
    losses = batch_objective(network, [sample], TrainConfig())
    losses["total"].backward()
    optimizer.step()
    assert all(torch.isfinite(value) for value in losses.values())
    assert not torch.equal(previous, network.latent_start)


def test_checkpoint_roundtrip_preserves_trainable_state(tmp_path):
    network = model()
    normalizer = ActionNormalizer({"low": [-1, -1], "high": [1, 1]})
    destination = tmp_path / "checkpoint"
    save_checkpoint(destination, network, Config(model=network.config), normalizer)
    expected = {name: tensor.clone() for name, tensor in network.state_dict().items()}
    with torch.no_grad():
        for parameter in network.parameters():
            parameter.add_(1)
    restore_weights(network, destination)
    for name, tensor in network.state_dict().items():
        assert torch.equal(tensor, expected[name])


def test_repair_preserves_exact_prefix_and_changes_suffix_length():
    from vlcot.control.budget import OPERATIONS, BudgetLedger
    from vlcot.control.policy import VLCoTPolicy
    from vlcot.config import VerificationConfig

    network = model().eval()
    network.config.max_stages = 3
    with torch.no_grad():
        network.stage_decoder.stop.weight.zero_()
        network.stage_decoder.stop.bias.fill_(-10)
    policy = VLCoTPolicy(network, VerificationConfig(), ActionNormalizer({"low": [-1, -1], "high": [1, 1]}))
    policy.reset(BudgetLedger(1000, {name: 1 for name in OPERATIONS}))
    original = policy.plan(observation())
    assert len(original.latents) == 3
    original.active = 1
    prefix = original.latents[:1].clone()
    policy.history.confirm(0, original.stages[0], 0)
    with torch.no_grad():
        network.stage_decoder.stop.bias.fill_(10)
    repaired = policy.repair(observation(1), 1)
    assert torch.equal(repaired.latents[:1], prefix)
    assert len(repaired.latents) == 2
    assert policy.regenerated_vectors == 2
    assert policy.ledger.records["encode"]["calls"] == 2


def test_rolling_extension_does_not_count_as_repair_vectors():
    from vlcot.control.budget import OPERATIONS, BudgetLedger
    from vlcot.control.policy import VLCoTPolicy
    from vlcot.config import VerificationConfig

    network = model().eval()
    policy = VLCoTPolicy(network, VerificationConfig(), ActionNormalizer({"low": [-1, -1], "high": [1, 1]}))
    policy.reset(BudgetLedger(1000, {name: 1 for name in OPERATIONS}))
    current = policy.plan(observation())
    current.active = len(current.stages)
    policy._extend(observation(1))
    assert policy.regenerated_vectors == 0
    assert policy.ledger.records["rolling"]["units"] > 0
