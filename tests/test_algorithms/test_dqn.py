import numpy as np

from vibe_rl.algorithms.dqn.agent import DQNAgent
from vibe_rl.algorithms.dqn.config import DQNConfig
from vibe_rl.algorithms.dqn.network import QNetwork
from vibe_rl.dataprotocol.transition import Transition


class TestQNetwork:
    def test_output_shape(self):
        import torch

        net = QNetwork(obs_dim=4, n_actions=2, hidden_sizes=(32, 32))
        x = torch.randn(8, 4)
        out = net(x)
        assert out.shape == (8, 2)

    def test_single_input(self):
        import torch

        net = QNetwork(obs_dim=2, n_actions=3)
        x = torch.randn(1, 2)
        out = net(x)
        assert out.shape == (1, 3)


class TestDQNAgent:
    def _make_agent(self) -> DQNAgent:
        config = DQNConfig(
            hidden_sizes=(32, 32),
            buffer_capacity=100,
            min_buffer_size=10,
            batch_size=8,
        )
        return DQNAgent(obs_shape=(2,), n_actions=4, config=config)

    def test_act_explore(self):
        agent = self._make_agent()
        state = np.array([0.5, 0.5], dtype=np.float32)
        action = agent.act(state, explore=True)
        assert 0 <= action < 4

    def test_act_greedy(self):
        agent = self._make_agent()
        state = np.array([0.5, 0.5], dtype=np.float32)
        action = agent.act(state, explore=False)
        assert 0 <= action < 4

    def test_observe_and_learn(self):
        agent = self._make_agent()
        # Fill buffer past min_buffer_size
        for i in range(20):
            t = Transition(
                state=np.random.randn(2).astype(np.float32),
                action=i % 4,
                reward=1.0,
                next_state=np.random.randn(2).astype(np.float32),
                terminated=False,
            )
            agent.observe(t)
        metrics = agent.learn()
        assert "loss" in metrics
        assert "q_mean" in metrics
        assert "epsilon" in metrics

    def test_learn_returns_empty_when_buffer_small(self):
        agent = self._make_agent()
        # Only add a few transitions (< min_buffer_size)
        for i in range(5):
            t = Transition(
                state=np.random.randn(2).astype(np.float32),
                action=0,
                reward=0.0,
                next_state=np.random.randn(2).astype(np.float32),
                terminated=False,
            )
            agent.observe(t)
        metrics = agent.learn()
        assert metrics == {}

    def test_save_and_load(self, tmp_path):
        agent = self._make_agent()
        for i in range(20):
            t = Transition(
                state=np.random.randn(2).astype(np.float32),
                action=i % 4,
                reward=1.0,
                next_state=np.random.randn(2).astype(np.float32),
                terminated=False,
            )
            agent.observe(t)
        agent.learn()

        save_path = tmp_path / "test_agent.pt"
        agent.save(save_path)
        assert save_path.exists()

        agent2 = self._make_agent()
        agent2.load(save_path)
        assert agent2._learn_step_count == agent._learn_step_count
