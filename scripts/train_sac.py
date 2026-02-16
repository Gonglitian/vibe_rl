#!/usr/bin/env python3
"""Train SAC via CLI.

Usage::

    python scripts/train_sac.py --help
    python scripts/train_sac.py --env-id Pendulum-v1
    python scripts/train_sac.py --env-id Pendulum-v1 --sac.actor-lr 1e-3 --sac.batch-size 512
    python scripts/train_sac.py --runner.total-timesteps 200000 --runner.seed 123
"""

from __future__ import annotations

from dataclasses import dataclass

import tyro

from vibe_rl.algorithms.sac.config import SACConfig
from vibe_rl.env import make
from vibe_rl.env.wrappers import AutoResetWrapper
from vibe_rl.runner import RunnerConfig, train_sac


@dataclass(frozen=True)
class TrainSACArgs:
    """SAC training configuration."""

    # Environment
    env_id: str = "Pendulum-v1"

    # Algorithm hyperparameters
    sac: SACConfig = SACConfig()

    # Runner / outer-loop settings
    runner: RunnerConfig = RunnerConfig()


def main(args: TrainSACArgs) -> None:
    env, env_params = make(args.env_id)
    env = AutoResetWrapper(env)
    env_params = env.default_params()

    result = train_sac(
        env,
        env_params,
        sac_config=args.sac,
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
    main(tyro.cli(TrainSACArgs))
