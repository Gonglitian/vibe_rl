"""Tests for the JAX training runner module.

Tests the evaluator, PPO (PureJaxRL-style), DQN (hybrid), and SAC (hybrid)
training loops with small configs to verify correctness.
"""

from __future__ import annotations

from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp

from vibe_rl.algorithms.dqn import DQN, DQNConfig
from vibe_rl.algorithms.ppo import PPO, PPOConfig
from vibe_rl.algorithms.sac import SAC, SACConfig
from vibe_rl.env import make
from vibe_rl.env.base import EnvParams, EnvState, Environment
from vibe_rl.env.spaces import Box
from vibe_rl.env.wrappers import AutoResetWrapper
from vibe_rl.runner import (
    DQNTrainResult,
    EvalMetrics,
    PPOMetricsHistory,
    PPOTrainState,
    RunnerConfig,
    SACTrainResult,
    evaluate,
    jit_evaluate,
    train_dqn,
    train_ppo,
    train_sac,
)


# ---- Tiny continuous-action environment for SAC tests ----

class _TinyContState(EnvState):
    """State for a trivial 1-D continuous env."""

    x: jax.Array


class _TinyContParams(EnvParams):
    """Params for the tiny continuous env."""

    max_steps: int = eqx.field(static=True, default=50)


class TinyContinuousEnv(Environment):
    """Trivial continuous-action environment: move x toward 0.

    obs = [x], action in [-1, 1], reward = -|x|, done after max_steps.
    """

    def reset(self, key, params):
        x = jax.random.uniform(key, (), minval=-1.0, maxval=1.0)
        state = _TinyContState(x=x, time=jnp.int32(0))
        return jnp.array([x]), state

    def step(self, key, state, action, params):
        new_x = jnp.clip(state.x + action[0] * 0.1, -1.0, 1.0)
        reward = -jnp.abs(new_x)
        new_time = state.time + 1
        done = new_time >= params.max_steps
        new_state = _TinyContState(x=new_x, time=new_time)
        return jnp.array([new_x]), new_state, reward, done, {}

    def default_params(self):
        return _TinyContParams()

    def observation_space(self, params):
        return Box(low=-1.0, high=1.0, shape=(1,))

    def action_space(self, params):
        return Box(low=-1.0, high=1.0, shape=(1,))


# ---- Evaluator tests ----

class TestEvaluator:
    def test_evaluate_cartpole(self):
        env, env_params = make("CartPole-v1")
        config = DQNConfig(hidden_sizes=(16, 16))
        rng = jax.random.PRNGKey(0)
        agent_state = DQN.init(rng, obs_shape=(4,), n_actions=2, config=config)

        def greedy_act(agent_state, obs):
            q_values = agent_state.params(obs)
            return jnp.argmax(q_values)

        result = evaluate(
            greedy_act, agent_state, env, env_params,
            n_episodes=4, max_steps=100, rng=jax.random.PRNGKey(42),
        )
        assert isinstance(result, EvalMetrics)
        assert result.mean_return.shape == ()
        assert result.std_return.shape == ()
        assert result.mean_length.shape == ()
        assert float(result.mean_length) > 0

    def test_jit_evaluate(self):
        env, env_params = make("CartPole-v1")
        config = DQNConfig(hidden_sizes=(16, 16))
        rng = jax.random.PRNGKey(1)
        agent_state = DQN.init(rng, obs_shape=(4,), n_actions=2, config=config)

        def greedy_act(agent_state, obs):
            q_values = agent_state.params(obs)
            return jnp.argmax(q_values)

        result = jit_evaluate(
            greedy_act, agent_state, env, env_params,
            n_episodes=4, max_steps=100, rng=jax.random.PRNGKey(42),
        )
        assert isinstance(result, EvalMetrics)
        assert float(result.mean_length) > 0

    def test_evaluate_deterministic(self):
        """Same seed should give same results."""
        env, env_params = make("CartPole-v1")
        config = DQNConfig(hidden_sizes=(16, 16))
        agent_state = DQN.init(jax.random.PRNGKey(0), (4,), 2, config)

        def greedy_act(agent_state, obs):
            return jnp.argmax(agent_state.params(obs))

        r1 = evaluate(
            greedy_act, agent_state, env, env_params,
            n_episodes=4, max_steps=100, rng=jax.random.PRNGKey(99),
        )
        r2 = evaluate(
            greedy_act, agent_state, env, env_params,
            n_episodes=4, max_steps=100, rng=jax.random.PRNGKey(99),
        )
        assert float(r1.mean_return) == float(r2.mean_return)
        assert float(r1.mean_length) == float(r2.mean_length)


# ---- PPO PureJaxRL runner tests ----

class TestTrainPPO:
    def test_basic_training(self):
        env, env_params = make("CartPole-v1")
        env = AutoResetWrapper(env)
        env_params = env.default_params()

        ppo_config = PPOConfig(
            n_steps=32, n_minibatches=2, n_epochs=2,
            hidden_sizes=(16, 16),
        )
        runner_config = RunnerConfig(total_timesteps=128, seed=0)

        train_state, history = train_ppo(
            env, env_params,
            ppo_config=ppo_config,
            runner_config=runner_config,
        )

        # 128 total steps / 32 per rollout = 4 updates
        n_updates = 128 // 32
        assert isinstance(train_state, PPOTrainState)
        assert isinstance(history, PPOMetricsHistory)
        assert history.total_loss.shape == (n_updates,)
        assert history.actor_loss.shape == (n_updates,)
        assert history.critic_loss.shape == (n_updates,)
        assert history.entropy.shape == (n_updates,)
        assert history.approx_kl.shape == (n_updates,)
        # Step counter should advance
        assert int(train_state.agent_state.step) == n_updates

    def test_explicit_shapes(self):
        env, env_params = make("CartPole-v1")
        env = AutoResetWrapper(env)
        env_params = env.default_params()

        ppo_config = PPOConfig(
            n_steps=16, n_minibatches=2, n_epochs=1,
            hidden_sizes=(8, 8),
        )
        runner_config = RunnerConfig(total_timesteps=64, seed=42)

        train_state, history = train_ppo(
            env, env_params,
            ppo_config=ppo_config,
            runner_config=runner_config,
            obs_shape=(4,),
            n_actions=2,
        )

        assert int(train_state.agent_state.step) == 4  # 64 / 16

    def test_deterministic(self):
        """Same seed produces same results."""
        env, env_params = make("CartPole-v1")
        env = AutoResetWrapper(env)
        env_params = env.default_params()

        ppo_config = PPOConfig(
            n_steps=16, n_minibatches=2, n_epochs=1,
            hidden_sizes=(8, 8),
        )
        runner_config = RunnerConfig(total_timesteps=32, seed=7)

        _, h1 = train_ppo(env, env_params, ppo_config=ppo_config, runner_config=runner_config)
        _, h2 = train_ppo(env, env_params, ppo_config=ppo_config, runner_config=runner_config)

        assert jnp.allclose(h1.total_loss, h2.total_loss)


# ---- DQN hybrid runner tests ----

class TestTrainDQN:
    def test_basic_training(self):
        env, env_params = make("CartPole-v1")
        env = AutoResetWrapper(env)
        env_params = env.default_params()

        dqn_config = DQNConfig(
            hidden_sizes=(16, 16), batch_size=8,
            target_update_freq=50, epsilon_decay_steps=200,
        )
        runner_config = RunnerConfig(
            total_timesteps=300, warmup_steps=32,
            buffer_size=500, log_interval=100, seed=0,
        )

        result = train_dqn(
            env, env_params,
            dqn_config=dqn_config,
            runner_config=runner_config,
        )

        assert isinstance(result, DQNTrainResult)
        # Should have completed some episodes
        assert len(result.episode_returns) > 0
        # Should have logged some metrics
        assert len(result.metrics_log) > 0
        assert "loss" in result.metrics_log[0]
        assert "q_mean" in result.metrics_log[0]
        assert "epsilon" in result.metrics_log[0]

    def test_callback(self):
        env, env_params = make("CartPole-v1")
        env = AutoResetWrapper(env)
        env_params = env.default_params()

        dqn_config = DQNConfig(hidden_sizes=(16, 16), batch_size=8)
        runner_config = RunnerConfig(
            total_timesteps=200, warmup_steps=32,
            buffer_size=300, log_interval=50, seed=1,
        )

        called = []

        def my_callback(step, agent_state, metrics):
            called.append(step)

        train_dqn(
            env, env_params,
            dqn_config=dqn_config,
            runner_config=runner_config,
            callback=my_callback,
        )
        assert len(called) > 0

    def test_agent_state_progresses(self):
        env, env_params = make("CartPole-v1")
        env = AutoResetWrapper(env)
        env_params = env.default_params()

        dqn_config = DQNConfig(hidden_sizes=(16, 16), batch_size=8)
        runner_config = RunnerConfig(
            total_timesteps=100, warmup_steps=16,
            buffer_size=200, seed=2,
        )

        result = train_dqn(
            env, env_params,
            dqn_config=dqn_config,
            runner_config=runner_config,
        )
        # Step counter should have advanced
        assert int(result.agent_state.step) > 0


# ---- SAC hybrid runner tests ----

class TestTrainSAC:
    def test_basic_training(self):
        env = TinyContinuousEnv()
        env = AutoResetWrapper(env)
        env_params = env.default_params()

        sac_config = SACConfig(
            hidden_sizes=(16, 16), batch_size=8,
            actor_lr=1e-3, critic_lr=1e-3, alpha_lr=1e-3,
        )
        runner_config = RunnerConfig(
            total_timesteps=200, warmup_steps=32,
            buffer_size=500, log_interval=100, seed=0,
        )

        result = train_sac(
            env, env_params,
            sac_config=sac_config,
            runner_config=runner_config,
            obs_shape=(1,),
            action_dim=1,
        )

        assert isinstance(result, SACTrainResult)
        assert len(result.episode_returns) > 0
        assert len(result.metrics_log) > 0
        assert "actor_loss" in result.metrics_log[0]
        assert "critic_loss" in result.metrics_log[0]
        assert "alpha" in result.metrics_log[0]

    def test_agent_state_progresses(self):
        env = TinyContinuousEnv()
        env = AutoResetWrapper(env)
        env_params = env.default_params()

        sac_config = SACConfig(
            hidden_sizes=(16, 16), batch_size=8,
        )
        runner_config = RunnerConfig(
            total_timesteps=100, warmup_steps=16,
            buffer_size=200, seed=3,
        )

        result = train_sac(
            env, env_params,
            sac_config=sac_config,
            runner_config=runner_config,
            obs_shape=(1,),
            action_dim=1,
        )
        assert int(result.agent_state.step) > 0


# ---- RunnerConfig tests ----

class TestRunnerConfig:
    def test_defaults(self):
        cfg = RunnerConfig()
        assert cfg.total_timesteps == 100_000
        assert cfg.eval_every == 5_000
        assert cfg.eval_episodes == 10
        assert cfg.buffer_size == 100_000
        assert cfg.warmup_steps == 1_000

    def test_frozen(self):
        cfg = RunnerConfig()
        try:
            cfg.seed = 42  # type: ignore
            assert False, "Should be frozen"
        except AttributeError:
            pass
