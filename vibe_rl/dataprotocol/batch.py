from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(slots=True)
class TorchBatch:
    """GPU-ready tensors."""

    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_states: torch.Tensor
    terminated: torch.Tensor
    weights: torch.Tensor | None = None


@dataclass(slots=True)
class Batch:
    """
    A batch of transitions collated into contiguous numpy arrays.
    Call .to_torch() to convert to tensors before feeding to networks.
    """

    states: np.ndarray  # (batch_size, *obs_shape)
    actions: np.ndarray  # (batch_size,)
    rewards: np.ndarray  # (batch_size,)
    next_states: np.ndarray  # (batch_size, *obs_shape)
    terminated: np.ndarray  # (batch_size,) bool

    # Optional fields for prioritized replay
    indices: np.ndarray | None = None
    weights: np.ndarray | None = None

    def to_torch(self, device: torch.device | str = "cpu") -> TorchBatch:
        return TorchBatch(
            states=torch.as_tensor(self.states, dtype=torch.float32, device=device),
            actions=torch.as_tensor(self.actions, dtype=torch.long, device=device),
            rewards=torch.as_tensor(self.rewards, dtype=torch.float32, device=device),
            next_states=torch.as_tensor(self.next_states, dtype=torch.float32, device=device),
            terminated=torch.as_tensor(self.terminated, dtype=torch.float32, device=device),
            weights=(
                torch.as_tensor(self.weights, dtype=torch.float32, device=device)
                if self.weights is not None
                else None
            ),
        )
