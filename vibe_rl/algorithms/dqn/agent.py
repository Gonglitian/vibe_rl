from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from vibe_rl.agent.base import BaseAgent
from vibe_rl.algorithms.dqn.config import DQNConfig
from vibe_rl.algorithms.dqn.network import QNetwork
from vibe_rl.dataprotocol.replay_buffer import ReplayBuffer
from vibe_rl.dataprotocol.transition import Transition
from vibe_rl.utils.schedule import LinearSchedule


class DQNAgent(BaseAgent):
    """
    Deep Q-Network agent (Mnih et al., 2015).

    Features:
      - Experience replay
      - Target network with periodic hard updates
      - Linear epsilon decay
    """

    def __init__(
        self,
        obs_shape: tuple[int, ...],
        n_actions: int,
        config: DQNConfig | None = None,
    ) -> None:
        self.config = config or DQNConfig()
        self.n_actions = n_actions
        self.device = torch.device(self.config.device)

        obs_dim = int(np.prod(obs_shape))
        self.q_net = QNetwork(obs_dim, n_actions, self.config.hidden_sizes).to(self.device)
        self.target_net = QNetwork(obs_dim, n_actions, self.config.hidden_sizes).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=self.config.lr)
        self.buffer = ReplayBuffer(self.config.buffer_capacity, obs_shape)
        self.epsilon_schedule = LinearSchedule(
            start=self.config.epsilon_start,
            end=self.config.epsilon_end,
            steps=self.config.epsilon_decay_steps,
        )
        self._learn_step_count = 0

    def observe(self, transition: Transition) -> None:
        """Store a transition in the replay buffer."""
        self.buffer.push(transition)

    def act(self, state: np.ndarray, *, explore: bool = True) -> int:
        if explore and np.random.random() < self.epsilon_schedule.value:
            return int(np.random.randint(self.n_actions))
        with torch.no_grad():
            state_t = torch.as_tensor(
                state, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            q_values = self.q_net(state_t)
            return int(q_values.argmax(dim=1).item())

    def learn(self) -> dict[str, float]:
        if len(self.buffer) < self.config.min_buffer_size:
            return {}

        batch = self.buffer.sample(self.config.batch_size).to_torch(self.device)

        # Current Q values: Q(s, a)
        q_values = self.q_net(batch.states).gather(1, batch.actions.unsqueeze(1)).squeeze(1)

        # Target: r + gamma * max_a' Q_target(s', a') * (1 - terminated)
        with torch.no_grad():
            next_q = self.target_net(batch.next_states).max(dim=1).values
            target = batch.rewards + self.config.gamma * next_q * (1.0 - batch.terminated)

        loss = F.mse_loss(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self._learn_step_count += 1
        self.epsilon_schedule.step()

        if self._learn_step_count % self.config.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return {
            "loss": loss.item(),
            "q_mean": q_values.mean().item(),
            "epsilon": self.epsilon_schedule.value,
        }

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "q_net": self.q_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "learn_step_count": self._learn_step_count,
                "epsilon_step": self.epsilon_schedule._current_step,
            },
            path,
        )

    def load(self, path: Path | str) -> None:
        path = Path(path)
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.q_net.load_state_dict(checkpoint["q_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self._learn_step_count = checkpoint["learn_step_count"]
        self.epsilon_schedule._current_step = checkpoint["epsilon_step"]
