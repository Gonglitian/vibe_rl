#!/usr/bin/env python3
"""Unified training script with preset selection.

Select a preset configuration and optionally override any field::

    python scripts/train.py cartpole_ppo
    python scripts/train.py cartpole_ppo --algo.lr 1e-3
    python scripts/train.py pendulum_sac --runner.total_timesteps 500000
    python scripts/train.py cartpole_ppo --help
"""

from __future__ import annotations

from vibe_rl.algorithms.dqn.config import DQNConfig
from vibe_rl.algorithms.ppo.config import PPOConfig
from vibe_rl.algorithms.sac.config import SACConfig
from vibe_rl.configs import TrainConfig, cli
from vibe_rl.env import make
from vibe_rl.env.wrappers import AutoResetWrapper
from vibe_rl.run_dir import RunDir
from vibe_rl.runner import train_dqn, train_ppo, train_sac


def _algo_name(algo: object) -> str:
    """Return a short name for the algorithm config."""
    if isinstance(algo, PPOConfig):
        return "ppo"
    if isinstance(algo, DQNConfig):
        return "dqn"
    if isinstance(algo, SACConfig):
        return "sac"
    return type(algo).__name__.lower()


def main(config: TrainConfig) -> None:
    env, env_params = make(config.env_id)
    env = AutoResetWrapper(env)
    env_params = env.default_params()

    algo = config.algo

    # Create a RunDir for this experiment
    experiment_name = f"{config.env_id}_{_algo_name(algo)}"
    run_dir = RunDir(experiment_name)
    run_dir.save_config(config)
    print(f"Run directory: {run_dir.root}")

    if isinstance(algo, PPOConfig):
        train_state, metrics = train_ppo(
            env,
            env_params,
            ppo_config=algo,
            runner_config=config.runner,
            run_dir=run_dir,
        )
        final_loss = float(metrics.total_loss[-1])
        final_entropy = float(metrics.entropy[-1])
        print(
            f"Training complete | "
            f"final_loss={final_loss:.4f} | "
            f"final_entropy={final_entropy:.4f}"
        )

    elif isinstance(algo, DQNConfig):
        result = train_dqn(
            env,
            env_params,
            dqn_config=algo,
            runner_config=config.runner,
            run_dir=run_dir,
        )
        n_episodes = len(result.episode_returns)
        last_returns = (
            result.episode_returns[-10:]
            if n_episodes >= 10
            else result.episode_returns
        )
        mean_return = sum(last_returns) / len(last_returns) if last_returns else 0.0
        print(
            f"Training complete | "
            f"episodes={n_episodes} | "
            f"mean_return(last 10)={mean_return:.1f}"
        )

    elif isinstance(algo, SACConfig):
        result = train_sac(
            env,
            env_params,
            sac_config=algo,
            runner_config=config.runner,
            run_dir=run_dir,
        )
        n_episodes = len(result.episode_returns)
        last_returns = (
            result.episode_returns[-10:]
            if n_episodes >= 10
            else result.episode_returns
        )
        mean_return = sum(last_returns) / len(last_returns) if last_returns else 0.0
        print(
            f"Training complete | "
            f"episodes={n_episodes} | "
            f"mean_return(last 10)={mean_return:.1f}"
        )

    else:
        raise TypeError(f"Unknown algorithm config type: {type(algo)}")

    print(f"Metrics: {run_dir.log_path()}")


if __name__ == "__main__":
    main(cli())
