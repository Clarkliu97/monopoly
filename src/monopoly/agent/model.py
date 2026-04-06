from __future__ import annotations

"""Policy model abstractions used for Monopoly RL training and inference."""

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from monopoly.agent.config import PolicyConfig
from monopoly.agent.features import GLOBAL_FEATURE_SIZE, OBSERVATION_PLAYER_COUNT, OBSERVATION_SPACE_COUNT, PLAYER_FEATURE_SIZE, SPACE_FEATURE_SIZE


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    """One sampled or greedy policy action together with its value estimate."""

    action_id: int
    log_probability: float
    value: float


@dataclass(frozen=True, slots=True)
class TrainingExample:
    """Serialized PPO training example collected from one environment transition."""

    observation: list[float]
    action_mask: list[bool]
    heuristic_bias: list[float]
    action_id: int
    discounted_return: float
    advantage: float
    old_log_probability: float
    threat_target: float = 0.0


class _MlpPolicyNetwork(nn.Module):
    """Simple MLP backbone for policy, value, and auxiliary threat prediction."""

    def __init__(self, observation_size: int, action_count: int, hidden_sizes: tuple[int, ...]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        input_size = observation_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            input_size = hidden_size
        self.backbone = nn.Sequential(*layers)
        self.policy_head = nn.Linear(input_size, action_count)
        self.value_head = nn.Linear(input_size, 1)
        self.threat_head = nn.Linear(input_size, 1)

    def forward(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the MLP backbone and return policy logits, values, and threat predictions."""
        hidden = self.backbone(observations)
        logits = self.policy_head(hidden)
        values = self.value_head(hidden).squeeze(-1)
        threats = self.threat_head(hidden).squeeze(-1)
        return logits, values, threats


class _TransformerPolicyNetwork(nn.Module):
    """Token-based transformer backbone for structured board observations."""

    def __init__(
        self,
        action_count: int,
        *,
        input_layout: Mapping[str, int],
        embedding_size: int,
        head_count: int,
        layer_count: int,
    ) -> None:
        super().__init__()
        self.global_size = int(input_layout["global_size"])
        self.player_count = int(input_layout["player_count"])
        self.player_size = int(input_layout["player_size"])
        self.space_count = int(input_layout["space_count"])
        self.space_size = int(input_layout["space_size"])
        self.global_projection = nn.Linear(self.global_size, embedding_size)
        self.player_projection = nn.Linear(self.player_size, embedding_size)
        self.space_projection = nn.Linear(self.space_size, embedding_size)
        token_count = 2 + self.player_count + self.space_count
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_size))
        self.position_embeddings = nn.Parameter(torch.zeros(1, token_count, embedding_size))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_size,
            nhead=head_count,
            dim_feedforward=embedding_size * 4,
            dropout=0.0,
            batch_first=True,
            activation="gelu",
        )
        self.backbone = nn.TransformerEncoder(encoder_layer, num_layers=layer_count)
        self.policy_head = nn.Linear(embedding_size, action_count)
        self.value_head = nn.Linear(embedding_size, 1)
        self.threat_head = nn.Linear(embedding_size, 1)

    def forward(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project flat observations into tokens and return pooled policy/value outputs."""
        batch_size = observations.shape[0]
        global_end = self.global_size
        players_end = global_end + self.player_count * self.player_size
        global_features = observations[:, :global_end]
        player_features = observations[:, global_end:players_end].reshape(batch_size, self.player_count, self.player_size)
        space_features = observations[:, players_end:].reshape(batch_size, self.space_count, self.space_size)
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        global_token = self.global_projection(global_features).unsqueeze(1)
        player_tokens = self.player_projection(player_features)
        space_tokens = self.space_projection(space_features)
        tokens = torch.cat((cls_token, global_token, player_tokens, space_tokens), dim=1)
        tokens = tokens + self.position_embeddings[:, :tokens.shape[1], :]
        hidden = self.backbone(tokens)
        pooled = hidden[:, 0, :]
        logits = self.policy_head(pooled)
        values = self.value_head(pooled).squeeze(-1)
        threats = self.threat_head(pooled).squeeze(-1)
        return logits, values, threats


class TorchPolicyModel:
    """High-level wrapper around policy networks, sampling, and tensor utilities."""

    def __init__(
        self,
        observation_size: int,
        action_count: int,
        seed: int = 7,
        hidden_sizes: tuple[int, ...] = (512, 512, 256, 256),
        device: str = "cpu",
        model_type: str = "mlp",
        transformer_embedding_size: int = 128,
        transformer_heads: int = 8,
        transformer_layers: int = 2,
        input_layout: Mapping[str, int] | None = None,
    ) -> None:
        self.observation_size = observation_size
        self.action_count = action_count
        self.hidden_sizes = hidden_sizes
        self.model_type = model_type
        self.transformer_embedding_size = int(transformer_embedding_size)
        self.transformer_heads = int(transformer_heads)
        self.transformer_layers = int(transformer_layers)
        self.input_layout = {
            "global_size": GLOBAL_FEATURE_SIZE,
            "player_count": OBSERVATION_PLAYER_COUNT,
            "player_size": PLAYER_FEATURE_SIZE,
            "space_count": OBSERVATION_SPACE_COUNT,
            "space_size": SPACE_FEATURE_SIZE,
        } if input_layout is None else {key: int(value) for key, value in input_layout.items()}
        self.device = torch.device(device)
        self._rng = np.random.default_rng(seed)
        self._torch_generator = torch.Generator(device=self.device.type)
        self._torch_generator.manual_seed(seed)
        self._use_pinned_memory = self.device.type == "cuda"
        if self.model_type == "transformer":
            self.network = _TransformerPolicyNetwork(
                action_count,
                input_layout=self.input_layout,
                embedding_size=self.transformer_embedding_size,
                head_count=self.transformer_heads,
                layer_count=self.transformer_layers,
            ).to(self.device)
        elif self.model_type == "mlp":
            self.network = _MlpPolicyNetwork(observation_size, action_count, hidden_sizes).to(self.device)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def act(
        self,
        observation: np.ndarray,
        action_mask: np.ndarray,
        heuristic_bias: np.ndarray | None = None,
        explore: bool = True,
        use_heuristic_bias: bool = True,
    ) -> PolicyDecision:
        """Choose one legal action from a masked policy distribution."""
        observation_tensor = torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        action_mask_tensor = torch.as_tensor(action_mask, dtype=torch.bool, device=self.device).unsqueeze(0)
        heuristic_tensor = None if heuristic_bias is None else torch.as_tensor(heuristic_bias, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            logits, values, _ = self.network(observation_tensor)
            masked_logits = self._apply_mask(logits, action_mask_tensor, heuristic_tensor, use_heuristic_bias=use_heuristic_bias)
            distribution = Categorical(logits=masked_logits)
            if explore:
                probabilities = torch.softmax(masked_logits, dim=-1)
                action_tensor = torch.multinomial(probabilities, num_samples=1, generator=self._torch_generator).squeeze(-1)
            else:
                action_tensor = torch.argmax(masked_logits, dim=-1)
            log_probability = distribution.log_prob(action_tensor)

        return PolicyDecision(
            action_id=int(action_tensor.item()),
            log_probability=float(log_probability.item()),
            value=float(values.squeeze(0).item()),
        )

    def batch_tensors(self, examples: list[TrainingExample]) -> dict[str, torch.Tensor]:
        """Pack Python training examples into device-resident tensors for optimization."""
        observations = self._tensor_from_array(np.asarray([example.observation for example in examples], dtype=np.float32), dtype=torch.float32)
        action_masks = self._tensor_from_array(np.asarray([example.action_mask for example in examples], dtype=np.bool_), dtype=torch.bool)
        heuristic_biases = self._tensor_from_array(np.asarray([example.heuristic_bias for example in examples], dtype=np.float32), dtype=torch.float32)
        actions = self._tensor_from_array(np.asarray([example.action_id for example in examples], dtype=np.int64), dtype=torch.long)
        returns = self._tensor_from_array(np.asarray([example.discounted_return for example in examples], dtype=np.float32), dtype=torch.float32)
        advantages = self._tensor_from_array(np.asarray([example.advantage for example in examples], dtype=np.float32), dtype=torch.float32)
        old_log_probabilities = self._tensor_from_array(np.asarray([example.old_log_probability for example in examples], dtype=np.float32), dtype=torch.float32)
        threat_targets = self._tensor_from_array(np.asarray([example.threat_target for example in examples], dtype=np.float32), dtype=torch.float32)
        return {
            "observations": observations,
            "action_masks": action_masks,
            "heuristic_biases": heuristic_biases,
            "actions": actions,
            "returns": returns,
            "advantages": advantages,
            "old_log_probabilities": old_log_probabilities,
            "threat_targets": threat_targets,
        }

    def predict_values(self, observations: np.ndarray) -> np.ndarray:
        """Predict state values for one observation or a batch of observations."""
        observation_array = np.asarray(observations, dtype=np.float32)
        if observation_array.ndim == 1:
            observation_array = np.expand_dims(observation_array, axis=0)
        observation_tensor = torch.as_tensor(observation_array, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            _, values, _ = self.network(observation_tensor)
        return values.detach().cpu().numpy()

    def evaluate_batch(
        self,
        batch: dict[str, torch.Tensor],
        *,
        use_heuristic_bias: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate a training batch and return log-probs, entropy, values, and threats."""
        logits, values, threat_predictions = self.network(batch["observations"])
        masked_logits = self._apply_mask(
            logits,
            batch["action_masks"],
            batch["heuristic_biases"],
            use_heuristic_bias=use_heuristic_bias,
        )
        distribution = Categorical(logits=masked_logits)
        log_probabilities = distribution.log_prob(batch["actions"])
        entropy = distribution.entropy()
        return log_probabilities, entropy, values, threat_predictions

    def to_state(self) -> dict[str, Any]:
        """Serialize architecture, weights, and RNG state for checkpointing."""
        return {
            "observation_size": self.observation_size,
            "action_count": self.action_count,
            "hidden_sizes": list(self.hidden_sizes),
            "model_type": self.model_type,
            "transformer_embedding_size": self.transformer_embedding_size,
            "transformer_heads": self.transformer_heads,
            "transformer_layers": self.transformer_layers,
            "input_layout": dict(self.input_layout),
            "device": str(self.device),
            "model_state": {name: tensor.detach().cpu() for name, tensor in self.network.state_dict().items()},
            "rng_state": self._rng.bit_generator.state,
            "torch_rng_state": self._torch_generator.get_state(),
        }

    @classmethod
    def from_state(cls, state: dict[str, Any], device_override: str | None = None) -> TorchPolicyModel:
        """Reconstruct a policy model from `to_state()` output."""
        model = cls(
            int(state["observation_size"]),
            int(state["action_count"]),
            hidden_sizes=tuple(int(value) for value in state.get("hidden_sizes", (512, 512, 256, 256))),
            device=device_override or str(state.get("device", "cpu")),
            model_type=str(state.get("model_type", "mlp")),
            transformer_embedding_size=int(state.get("transformer_embedding_size", 128)),
            transformer_heads=int(state.get("transformer_heads", 8)),
            transformer_layers=int(state.get("transformer_layers", 2)),
            input_layout=state.get("input_layout"),
        )
        model.load_state(state)
        return model

    def load_state(self, state: dict[str, Any], *, load_rng_state: bool = True) -> None:
        """Load network parameters and optionally restore RNG state from serialized data."""
        self.network.load_state_dict(state["model_state"])
        if load_rng_state and "rng_state" in state:
            self._rng.bit_generator.state = state["rng_state"]
        if load_rng_state and "torch_rng_state" in state:
            try:
                self._torch_generator.set_state(state["torch_rng_state"])
            except RuntimeError:
                pass

    def set_seed(self, seed: int) -> None:
        """Reset NumPy and Torch RNG state for reproducible exploration."""
        self._rng = np.random.default_rng(seed)
        self._torch_generator.manual_seed(seed)

    def iter_minibatches(
        self,
        batch: dict[str, torch.Tensor],
        *,
        minibatch_size: int,
        shuffle: bool = True,
    ) -> tuple[dict[str, torch.Tensor], ...]:
        """Split a packed training batch into minibatches on the model device."""
        batch_size = batch["observations"].shape[0]
        if batch_size == 0:
            return ()
        if shuffle:
            indices = torch.randperm(batch_size, device=self.device)
        else:
            indices = torch.arange(batch_size, device=self.device)
        minibatches: list[dict[str, torch.Tensor]] = []
        for start in range(0, batch_size, minibatch_size):
            batch_indices = indices[start:start + minibatch_size]
            minibatches.append({key: value[batch_indices] for key, value in batch.items()})
        return tuple(minibatches)

    @staticmethod
    def _apply_mask(
        logits: torch.Tensor,
        action_mask: torch.Tensor,
        heuristic_bias: torch.Tensor | None,
        *,
        use_heuristic_bias: bool = True,
    ) -> torch.Tensor:
        """Apply optional heuristic bias and force illegal actions to large negative logits."""
        if use_heuristic_bias and heuristic_bias is not None:
            logits = logits + heuristic_bias
        return logits.masked_fill(~action_mask, -1e9)

    def zero_parameters(self) -> None:
        """Zero all network parameters for deterministic tests and controlled baselines."""
        with torch.no_grad():
            for parameter in self.network.parameters():
                parameter.zero_()

    def _tensor_from_array(self, values: np.ndarray, *, dtype: torch.dtype) -> torch.Tensor:
        """Convert a NumPy array into an appropriately typed tensor on the model device."""
        tensor = torch.from_numpy(values)
        if tensor.dtype != dtype:
            tensor = tensor.to(dtype=dtype)
        if self.device.type == "cpu":
            return tensor.to(device=self.device)
        if self._use_pinned_memory and not tensor.is_pinned():
            tensor = tensor.pin_memory()
        return tensor.to(device=self.device, non_blocking=True)


LightweightPolicyModel = TorchPolicyModel