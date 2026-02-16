#!/usr/bin/env python3
"""Train DQN via CLI.

Usage::

    python scripts/train_dqn.py --help
    python scripts/train_dqn.py --env-id CartPole-v1
    python scripts/train_dqn.py --env-id CartPole-v1 --dqn.lr 5e-4 --dqn.batch-size 128
    python scripts/train_dqn.py --runner.total-timesteps 200000 --runner.seed 123
"""

from __future__ import annotations

from dataclasses import dataclass

import tyro

from vibe_rl.algorithms.dqn.config import DQNConfig
from vibe_rl.env import make
from vibe_rl.env.wrappers import AutoResetWrapper
from vibe_rl.runner import RunnerConfig, train_dqn


@dataclass(frozen=True)
class TrainDQNArgs:
    """DQN training configuration."""

    # Environment
    env_id: str = "CartPole-v1"

    # Algorithm hyperparameters
    dqn: DQNConfig = DQNConfig()

    # Runner / outer-loop settings
    runner: RunnerConfig = RunnerConfig()


def main(args: TrainDQNArgs) -> None:
    env, env_params = make(args.env_id)
    env = AutoResetWrapper(env)
    env_params = env.default_params()

    result = train_dqn(
        env,
        env_params,
        dqn_config=args.dqn,
        runner_config=args.runner,
    )

    n_episodes = len(result.episode_returns)
    last_returns = result.episode_returns[-10:] if n_episodes >= 10 else result.episode_returns
    mean_return = sum(last_returns) / len(last_returns) if last_returns else 0.0
    print(
        f"Training complete | "
        f"episodes={n_episodes} | "
        f"mean_return(last 10)={mean_return:.1f}"
    )


if __name__ == "__main__":
    main(tyro.cli(TrainDQNArgs))
