"""Integration tests: training runners produce JSONL metrics via MetricsLogger.

Trains each algorithm for a small number of steps and verifies that
a ``metrics.jsonl`` file is created, parseable, and contains the
expected fields.
"""

from __future__ import annotations

from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp

from vibe_rl.algorithms.dqn import DQNConfig
from vibe_rl.algorithms.ppo import PPOConfig
from vibe_rl.algorithms.sac import SACConfig
from vibe_rl.env import make
from vibe_rl.env.base import EnvParams, EnvState, Environment
from vibe_rl.env.spaces import Box
from vibe_rl.env.wrappers import AutoResetWrapper
from vibe_rl.metrics import read_metrics
from vibe_rl.run_dir import RunDir
from vibe_rl.runner import RunnerConfig, train_dqn, train_ppo, train_sac


# ---- Tiny continuous-action environment for SAC tests ----

class _TinyContState(EnvState):
    x: jax.Array


class _TinyContParams(EnvParams):
    max_steps: int = eqx.field(static=True, default=50)


class _TinyContinuousEnv(Environment):
    """Trivial continuous-action env for SAC integration tests."""

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


# ---- PPO ----

class TestPPOMetricsIntegration:
    def test_ppo_writes_jsonl(self, tmp_path: Path) -> None:
        env, env_params = make("CartPole-v1")
        env = AutoResetWrapper(env)
        env_params = env.default_params()

        ppo_config = PPOConfig(
            n_steps=32, n_minibatches=2, n_epochs=2,
            hidden_sizes=(16, 16),
        )
        runner_config = RunnerConfig(total_timesteps=128, seed=0)
        run_dir = RunDir("test_ppo", base_dir=tmp_path, run_id="ppo_run")

        train_ppo(
            env, env_params,
            ppo_config=ppo_config,
            runner_config=runner_config,
            run_dir=run_dir,
        )

        metrics_path = run_dir.log_path()
        assert metrics_path.exists()

        records = read_metrics(metrics_path)
        n_updates = 128 // 32  # 4
        assert len(records) == n_updates

        for rec in records:
            assert "step" in rec
            assert "wall_time" in rec
            assert "total_loss" in rec
            assert "actor_loss" in rec
            assert "critic_loss" in rec
            assert "entropy" in rec
            assert "approx_kl" in rec

        # Steps should be monotonically increasing
        steps = [r["step"] for r in records]
        assert steps == sorted(steps)
        assert steps[-1] == 128

    def test_ppo_no_run_dir_still_works(self) -> None:
        """Training without run_dir should work as before."""
        env, env_params = make("CartPole-v1")
        env = AutoResetWrapper(env)
        env_params = env.default_params()

        ppo_config = PPOConfig(
            n_steps=16, n_minibatches=2, n_epochs=1,
            hidden_sizes=(8, 8),
        )
        runner_config = RunnerConfig(total_timesteps=32, seed=0)

        train_state, history = train_ppo(
            env, env_params,
            ppo_config=ppo_config,
            runner_config=runner_config,
        )
        assert history.total_loss.shape == (2,)


# ---- DQN ----

class TestDQNMetricsIntegration:
    def test_dqn_writes_jsonl(self, tmp_path: Path) -> None:
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
        run_dir = RunDir("test_dqn", base_dir=tmp_path, run_id="dqn_run")

        result = train_dqn(
            env, env_params,
            dqn_config=dqn_config,
            runner_config=runner_config,
            run_dir=run_dir,
        )

        metrics_path = run_dir.log_path()
        assert metrics_path.exists()

        records = read_metrics(metrics_path)
        assert len(records) > 0

        # Should have both training-metric records and episode records
        training_records = [r for r in records if "loss" in r]
        episode_records = [r for r in records if "episode_return" in r]

        assert len(training_records) > 0
        assert len(episode_records) > 0

        for rec in training_records:
            assert "step" in rec
            assert "wall_time" in rec
            assert "loss" in rec
            assert "q_mean" in rec
            assert "epsilon" in rec

        for rec in episode_records:
            assert "step" in rec
            assert "wall_time" in rec
            assert "episode_return" in rec
            assert "episode_length" in rec

    def test_dqn_no_run_dir_still_works(self) -> None:
        env, env_params = make("CartPole-v1")
        env = AutoResetWrapper(env)
        env_params = env.default_params()

        dqn_config = DQNConfig(hidden_sizes=(16, 16), batch_size=8)
        runner_config = RunnerConfig(
            total_timesteps=100, warmup_steps=16,
            buffer_size=200, seed=0,
        )

        result = train_dqn(
            env, env_params,
            dqn_config=dqn_config,
            runner_config=runner_config,
        )
        assert isinstance(result.metrics_log, list)


# ---- SAC ----

class TestSACMetricsIntegration:
    def test_sac_writes_jsonl(self, tmp_path: Path) -> None:
        env = _TinyContinuousEnv()
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
        run_dir = RunDir("test_sac", base_dir=tmp_path, run_id="sac_run")

        result = train_sac(
            env, env_params,
            sac_config=sac_config,
            runner_config=runner_config,
            obs_shape=(1,),
            action_dim=1,
            run_dir=run_dir,
        )

        metrics_path = run_dir.log_path()
        assert metrics_path.exists()

        records = read_metrics(metrics_path)
        assert len(records) > 0

        training_records = [r for r in records if "actor_loss" in r]
        episode_records = [r for r in records if "episode_return" in r]

        assert len(training_records) > 0
        assert len(episode_records) > 0

        for rec in training_records:
            assert "step" in rec
            assert "wall_time" in rec
            assert "actor_loss" in rec
            assert "critic_loss" in rec
            assert "alpha" in rec
            assert "entropy" in rec
            assert "q_mean" in rec

        for rec in episode_records:
            assert "step" in rec
            assert "wall_time" in rec
            assert "episode_return" in rec
            assert "episode_length" in rec

    def test_sac_no_run_dir_still_works(self) -> None:
        env = _TinyContinuousEnv()
        env = AutoResetWrapper(env)
        env_params = env.default_params()

        sac_config = SACConfig(hidden_sizes=(16, 16), batch_size=8)
        runner_config = RunnerConfig(
            total_timesteps=100, warmup_steps=16,
            buffer_size=200, seed=0,
        )

        result = train_sac(
            env, env_params,
            sac_config=sac_config,
            runner_config=runner_config,
            obs_shape=(1,),
            action_dim=1,
        )
        assert isinstance(result.metrics_log, list)


# ---- RunDir structure ----

class TestRunDirStructure:
    def test_run_dir_has_correct_layout(self, tmp_path: Path) -> None:
        """Verify RunDir creates the expected directory structure."""
        run_dir = RunDir("test_exp", base_dir=tmp_path, run_id="my_run")

        assert (run_dir.root / "logs").is_dir()
        assert (run_dir.root / "checkpoints").is_dir()
        assert (run_dir.root / "videos").is_dir()
        assert (run_dir.root / "artifacts").is_dir()

        # log_path() returns the metrics.jsonl path
        assert run_dir.log_path() == run_dir.root / "logs" / "metrics.jsonl"
