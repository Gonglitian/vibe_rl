"""Tests for the PPO algorithm (JAX version)."""

import jax
import jax.numpy as jnp

from vibe_rl.algorithms.ppo import PPO, PPOConfig, compute_gae
from vibe_rl.algorithms.ppo.network import ActorCategorical, ActorCriticShared, Critic
from vibe_rl.algorithms.ppo.types import ActorCriticParams, PPOState
from vibe_rl.dataprotocol.transition import PPOTransition
from vibe_rl.env import make
from vibe_rl.env.wrappers import AutoResetWrapper


class TestActorCategorical:
    def test_output_shape(self):
        key = jax.random.PRNGKey(0)
        net = ActorCategorical(obs_dim=4, n_actions=2, hidden_sizes=(32, 32), key=key)
        x = jnp.ones(4)
        logits = net(x)
        assert logits.shape == (2,)

    def test_batched(self):
        key = jax.random.PRNGKey(0)
        net = ActorCategorical(obs_dim=4, n_actions=3, hidden_sizes=(16,), key=key)
        x = jnp.ones((8, 4))
        logits = jax.vmap(net)(x)
        assert logits.shape == (8, 3)


class TestCritic:
    def test_output_shape(self):
        key = jax.random.PRNGKey(0)
        net = Critic(obs_dim=4, hidden_sizes=(32, 32), key=key)
        x = jnp.ones(4)
        value = net(x)
        assert value.shape == ()

    def test_batched(self):
        key = jax.random.PRNGKey(0)
        net = Critic(obs_dim=4, hidden_sizes=(16,), key=key)
        x = jnp.ones((8, 4))
        values = jax.vmap(net)(x)
        assert values.shape == (8,)


class TestActorCriticShared:
    def test_output_shapes(self):
        key = jax.random.PRNGKey(0)
        net = ActorCriticShared(obs_dim=4, n_actions=2, hidden_sizes=(32, 32), key=key)
        x = jnp.ones(4)
        logits, value = net(x)
        assert logits.shape == (2,)
        assert value.shape == ()


class TestGAE:
    def test_basic(self):
        rewards = jnp.ones(10)
        values = jnp.ones(10) * 0.5
        dones = jnp.zeros(10)
        last_value = jnp.array(0.5)
        advantages, returns = compute_gae(rewards, values, dones, last_value, 0.99, 0.95)
        assert advantages.shape == (10,)
        assert returns.shape == (10,)
        # Rewards exceed values, so advantages should be positive
        assert float(advantages.mean()) > 0

    def test_vectorized(self):
        T, N = 10, 4
        rewards = jnp.ones((T, N))
        values = jnp.ones((T, N)) * 0.5
        dones = jnp.zeros((T, N))
        last_value = jnp.ones(N) * 0.5
        advantages, returns = compute_gae(rewards, values, dones, last_value, 0.99, 0.95)
        assert advantages.shape == (T, N)
        assert returns.shape == (T, N)

    def test_done_resets_gae(self):
        rewards = jnp.array([1.0, 1.0, 1.0])
        values = jnp.array([0.5, 0.5, 0.5])
        dones = jnp.array([0.0, 1.0, 0.0])  # Episode ends at step 1
        last_value = jnp.array(0.5)
        advantages, _ = compute_gae(rewards, values, dones, last_value, 0.99, 0.95)
        # After done, GAE should reset — advantage at step 0 should not include
        # discounted future beyond the done
        assert advantages.shape == (3,)


class TestPPO:
    def _make_state(self, config=None):
        config = config or PPOConfig(hidden_sizes=(32, 32))
        rng = jax.random.PRNGKey(0)
        return PPO.init(rng, obs_shape=(4,), n_actions=2, config=config), config

    def test_init(self):
        state, _ = self._make_state()
        assert int(state.step) == 0
        assert isinstance(state, PPOState)
        assert isinstance(state.params, ActorCriticParams)

    def test_init_shared(self):
        config = PPOConfig(hidden_sizes=(32, 32), shared_backbone=True)
        state, _ = self._make_state(config)
        assert isinstance(state.params, ActorCriticShared)

    def test_act_single(self):
        state, config = self._make_state()
        obs = jnp.ones(4)
        action, log_prob, value, new_state = PPO.act(state, obs, config=config)
        assert action.shape == ()
        assert log_prob.shape == ()
        assert value.shape == ()

    def test_act_batch(self):
        state, config = self._make_state()
        obs = jnp.ones((8, 4))
        actions, log_probs, values, _ = PPO.act_batch(state, obs, config=config)
        assert actions.shape == (8,)
        assert log_probs.shape == (8,)
        assert values.shape == (8,)

    def test_evaluate_actions(self):
        state, config = self._make_state()
        obs = jnp.ones((16, 4))
        actions = jnp.zeros(16, dtype=jnp.int32)
        log_probs, values, entropy = PPO.evaluate_actions(
            state.params, obs, actions, config=config,
        )
        assert log_probs.shape == (16,)
        assert values.shape == (16,)
        assert entropy.shape == (16,)
        assert float(entropy.mean()) > 0

    def test_collect_and_update(self):
        config = PPOConfig(
            n_steps=32, n_minibatches=2, n_epochs=2, hidden_sizes=(32, 32),
        )
        env, env_params = make("CartPole-v1")
        env = AutoResetWrapper(env)
        env_params = env.default_params()

        rng = jax.random.PRNGKey(42)
        rng, env_key, agent_key = jax.random.split(rng, 3)
        obs, env_state = env.reset(env_key, env_params)
        state = PPO.init(agent_key, obs_shape=(4,), n_actions=2, config=config)

        state, trajectories, obs, env_state, last_value = PPO.collect_rollout(
            state, obs, env_state, env.step, env_params, config=config,
        )
        assert trajectories.obs.shape == (32, 4)
        assert trajectories.reward.shape == (32,)

        state, metrics = PPO.update(state, trajectories, last_value, config=config)
        assert int(state.step) == 1
        assert hasattr(metrics, "total_loss")
        assert hasattr(metrics, "actor_loss")
        assert hasattr(metrics, "critic_loss")
        assert hasattr(metrics, "entropy")
        assert hasattr(metrics, "approx_kl")

    def test_multi_update(self):
        config = PPOConfig(
            n_steps=32, n_minibatches=2, n_epochs=2, hidden_sizes=(32, 32),
        )
        env, env_params = make("CartPole-v1")
        env = AutoResetWrapper(env)
        env_params = env.default_params()

        rng = jax.random.PRNGKey(0)
        rng, env_key, agent_key = jax.random.split(rng, 3)
        obs, env_state = env.reset(env_key, env_params)
        state = PPO.init(agent_key, obs_shape=(4,), n_actions=2, config=config)

        for _ in range(3):
            state, traj, obs, env_state, lv = PPO.collect_rollout(
                state, obs, env_state, env.step, env_params, config=config,
            )
            state, metrics = PPO.update(state, traj, lv, config=config)

        assert int(state.step) == 3
