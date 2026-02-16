"""Tests for JAX DQN: networks, config, agent, and training smoke test."""

import jax
import jax.numpy as jnp
import numpy as np

from vibe_rl.algorithms.dqn import DQN, DQNConfig, DQNState, QNetwork
from vibe_rl.dataprotocol.replay_buffer import ReplayBuffer
from vibe_rl.dataprotocol.transition import Transition
from vibe_rl.env import make
from vibe_rl.env.wrappers import AutoResetWrapper


class TestQNetwork:
    def test_output_shape(self):
        key = jax.random.key(0)
        net = QNetwork(obs_dim=4, n_actions=2, hidden_sizes=(32, 32), key=key)
        x = jax.random.normal(key, (4,))
        out = net(x)
        assert out.shape == (2,)

    def test_batched_via_vmap(self):
        key = jax.random.key(0)
        net = QNetwork(obs_dim=4, n_actions=2, hidden_sizes=(32, 32), key=key)
        batch = jax.random.normal(key, (8, 4))
        out = jax.vmap(net)(batch)
        assert out.shape == (8, 2)

    def test_single_input(self):
        key = jax.random.key(1)
        net = QNetwork(obs_dim=2, n_actions=3, key=key)
        x = jax.random.normal(key, (2,))
        out = net(x)
        assert out.shape == (3,)


class TestDQNConfig:
    def test_defaults(self):
        config = DQNConfig()
        assert config.hidden_sizes == (128, 128)
        assert config.lr == 1e-3
        assert config.gamma == 0.99

    def test_frozen(self):
        config = DQNConfig()
        try:
            config.lr = 0.01  # type: ignore[misc]
            assert False, "Should have raised"
        except AttributeError:
            pass


class TestDQNInit:
    def test_returns_dqn_state(self):
        config = DQNConfig(hidden_sizes=(32, 32))
        rng = jax.random.PRNGKey(0)
        state = DQN.init(rng, obs_shape=(4,), n_actions=2, config=config)
        assert isinstance(state, DQNState)

    def test_step_starts_at_zero(self):
        config = DQNConfig(hidden_sizes=(32, 32))
        rng = jax.random.PRNGKey(0)
        state = DQN.init(rng, obs_shape=(4,), n_actions=2, config=config)
        assert int(state.step) == 0

    def test_deterministic_init(self):
        config = DQNConfig(hidden_sizes=(32, 32))
        s1 = DQN.init(jax.random.PRNGKey(0), (4,), 2, config)
        s2 = DQN.init(jax.random.PRNGKey(0), (4,), 2, config)
        leaves1 = jax.tree.leaves(s1.params)
        leaves2 = jax.tree.leaves(s2.params)
        for a, b in zip(leaves1, leaves2, strict=False):
            assert jnp.array_equal(a, b)


class TestDQNAct:
    def _make(self):
        config = DQNConfig(hidden_sizes=(32, 32))
        rng = jax.random.PRNGKey(0)
        state = DQN.init(rng, obs_shape=(4,), n_actions=2, config=config)
        return state, config

    def test_returns_action_and_state(self):
        state, config = self._make()
        obs = jnp.ones(4)
        action, new_state = DQN.act(state, obs, config=config, explore=True)
        assert action.shape == ()
        assert isinstance(new_state, DQNState)

    def test_greedy_action_deterministic(self):
        state, config = self._make()
        obs = jnp.ones(4)
        a1, _ = DQN.act(state, obs, config=config, explore=False)
        a2, _ = DQN.act(state, obs, config=config, explore=False)
        assert jnp.array_equal(a1, a2)

    def test_rng_advances(self):
        state, config = self._make()
        obs = jnp.zeros(4)
        _, new_state = DQN.act(state, obs, config=config, explore=True)
        assert not jnp.array_equal(state.rng, new_state.rng)


class TestDQNUpdate:
    def _make(self):
        config = DQNConfig(hidden_sizes=(32, 32), batch_size=16)
        rng = jax.random.PRNGKey(0)
        state = DQN.init(rng, obs_shape=(4,), n_actions=2, config=config)
        k1, k2 = jax.random.split(rng)
        batch = Transition(
            obs=jax.random.normal(k1, (16, 4)),
            action=jax.random.randint(k2, (16,), 0, 2),
            reward=jax.random.normal(k1, (16,)),
            next_obs=jax.random.normal(k2, (16, 4)),
            done=jnp.zeros(16),
        )
        return state, batch, config

    def test_returns_state_and_metrics(self):
        state, batch, config = self._make()
        new_state, metrics = DQN.update(state, batch, config=config)
        assert isinstance(new_state, DQNState)
        assert hasattr(metrics, "loss")
        assert hasattr(metrics, "q_mean")
        assert hasattr(metrics, "epsilon")

    def test_step_increments(self):
        state, batch, config = self._make()
        new_state, _ = DQN.update(state, batch, config=config)
        assert int(new_state.step) == 1

    def test_metrics_finite(self):
        state, batch, config = self._make()
        _, metrics = DQN.update(state, batch, config=config)
        assert jnp.isfinite(metrics.loss)
        assert jnp.isfinite(metrics.q_mean)

    def test_params_change(self):
        state, batch, config = self._make()
        new_state, _ = DQN.update(state, batch, config=config)
        old_leaves = jax.tree.leaves(state.params)
        new_leaves = jax.tree.leaves(new_state.params)
        any_changed = any(
            not jnp.array_equal(a, b) for a, b in zip(old_leaves, new_leaves, strict=False)
        )
        assert any_changed

    def test_multiple_updates(self):
        state, batch, config = self._make()
        losses = []
        for _ in range(20):
            state, metrics = DQN.update(state, batch, config=config)
            losses.append(float(metrics.loss))
        # Loss should decrease on a fixed batch
        assert losses[-1] < losses[0]


class TestDQNSmokeTest:
    def test_training_loop(self):
        """Full training loop smoke test: init -> act -> buffer -> update."""
        config = DQNConfig(
            hidden_sizes=(32, 32),
            lr=1e-3,
            batch_size=32,
            target_update_freq=100,
            epsilon_decay_steps=500,
        )
        env, env_params = make("CartPole-v1")
        env = AutoResetWrapper(env)
        env_params = env.default_params()

        rng = jax.random.PRNGKey(42)
        rng, env_key, agent_key = jax.random.split(rng, 3)

        obs, env_state = env.reset(env_key, env_params)
        state = DQN.init(agent_key, obs_shape=(4,), n_actions=2, config=config)
        buffer = ReplayBuffer(capacity=5_000, obs_shape=(4,))

        metrics = None
        for step in range(300):
            action, state = DQN.act(state, obs, config=config, explore=True)
            rng, step_key = jax.random.split(rng)
            next_obs, env_state, reward, done, _ = env.step(
                step_key, env_state, action, env_params,
            )
            buffer.push(
                np.asarray(obs), int(action), float(reward),
                np.asarray(next_obs), float(done),
            )
            obs = next_obs

            if len(buffer) >= 64:
                batch = buffer.sample(config.batch_size)
                state, metrics = DQN.update(state, batch, config=config)

        assert int(state.step) > 0
        assert metrics is not None
        assert jnp.isfinite(metrics.loss)
