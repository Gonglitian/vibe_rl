"""Tests that all algorithms' core functions compile correctly under jax.jit.

Verifies that act() and update() can be JIT-compiled and produce
correct results, including repeated calls with different inputs.
"""

import jax
import jax.numpy as jnp

from vibe_rl.algorithms.dqn import DQN, DQNConfig
from vibe_rl.algorithms.ppo import PPO, PPOConfig
from vibe_rl.algorithms.sac import SAC, SACConfig
from vibe_rl.dataprotocol.transition import Transition
from vibe_rl.env import make
from vibe_rl.env.wrappers import AutoResetWrapper


class TestDQNJIT:
    def test_act_jit(self):
        config = DQNConfig(hidden_sizes=(32, 32))
        state = DQN.init(jax.random.PRNGKey(0), (4,), 2, config)
        obs = jnp.ones(4)
        # First call triggers compilation
        action, s1 = DQN.act(state, obs, config=config, explore=True)
        assert action.shape == ()
        # Second call uses cached compiled function
        action2, s2 = DQN.act(s1, obs * 2, config=config, explore=True)
        assert action2.shape == ()

    def test_update_jit(self):
        config = DQNConfig(hidden_sizes=(32, 32))
        state = DQN.init(jax.random.PRNGKey(0), (4,), 2, config)
        k = jax.random.PRNGKey(1)
        batch = Transition(
            obs=jax.random.normal(k, (16, 4)),
            action=jax.random.randint(k, (16,), 0, 2),
            reward=jnp.ones(16),
            next_obs=jax.random.normal(k, (16, 4)),
            done=jnp.zeros(16),
        )
        # Two consecutive updates to verify re-use of compiled fn
        s1, m1 = DQN.update(state, batch, config=config)
        s2, m2 = DQN.update(s1, batch, config=config)
        assert int(s2.step) == 2
        assert jnp.isfinite(m2.loss)


class TestPPOJIT:
    def test_act_jit(self):
        config = PPOConfig(hidden_sizes=(32, 32))
        state = PPO.init(jax.random.PRNGKey(0), (4,), 2, config)
        obs = jnp.ones(4)
        action, lp, value, s1 = PPO.act(state, obs, config=config)
        assert action.shape == ()
        assert lp.shape == ()
        assert value.shape == ()

    def test_collect_rollout_jit(self):
        config = PPOConfig(hidden_sizes=(32, 32), n_steps=16, n_minibatches=2, n_epochs=2)
        env, env_params = make("CartPole-v1")
        env = AutoResetWrapper(env)
        env_params = env.default_params()
        rng = jax.random.PRNGKey(0)
        rng, ek, ak = jax.random.split(rng, 3)
        obs, env_state = env.reset(ek, env_params)
        state = PPO.init(ak, (4,), 2, config)

        # collect_rollout is JIT-compiled via decorator
        state, traj, obs, env_state, lv = PPO.collect_rollout(
            state, obs, env_state, env.step, env_params, config=config,
        )
        assert traj.obs.shape == (16, 4)

    def test_update_jit(self):
        config = PPOConfig(hidden_sizes=(32, 32), n_steps=16, n_minibatches=2, n_epochs=2)
        env, env_params = make("CartPole-v1")
        env = AutoResetWrapper(env)
        env_params = env.default_params()
        rng = jax.random.PRNGKey(0)
        rng, ek, ak = jax.random.split(rng, 3)
        obs, env_state = env.reset(ek, env_params)
        state = PPO.init(ak, (4,), 2, config)

        state, traj, obs, env_state, lv = PPO.collect_rollout(
            state, obs, env_state, env.step, env_params, config=config,
        )
        state, metrics = PPO.update(state, traj, lv, config=config)
        assert int(state.step) == 1
        assert jnp.isfinite(metrics.total_loss)


class TestSACJIT:
    def test_act_jit(self):
        config = SACConfig(hidden_sizes=(32, 32))
        state = SAC.init(jax.random.PRNGKey(0), (4,), 2, config)
        obs = jnp.ones(4)
        action, s1 = SAC.act(state, obs, config=config, explore=True)
        assert action.shape == (2,)
        action2, s2 = SAC.act(s1, obs, config=config, explore=False)
        assert action2.shape == (2,)

    def test_update_jit(self):
        config = SACConfig(hidden_sizes=(32, 32), batch_size=16)
        state = SAC.init(jax.random.PRNGKey(0), (4,), 2, config)
        k = jax.random.PRNGKey(1)
        batch = Transition(
            obs=jax.random.normal(k, (16, 4)),
            action=jax.random.normal(k, (16, 2)),
            reward=jnp.ones(16),
            next_obs=jax.random.normal(k, (16, 4)),
            done=jnp.zeros(16),
        )
        s1, m1 = SAC.update(state, batch, config=config)
        s2, m2 = SAC.update(s1, batch, config=config)
        assert int(s2.step) == 2
        assert jnp.isfinite(m2.critic_loss)
        assert jnp.isfinite(m2.actor_loss)
