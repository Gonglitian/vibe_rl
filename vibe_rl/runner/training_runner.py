from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from vibe_rl.algorithms.dqn.agent import DQNAgent
from vibe_rl.dataprotocol.transition import Transition
from vibe_rl.env.base import BaseEnv
from vibe_rl.runner.evaluator import evaluate
from vibe_rl.utils.logging import Logger


class TrainingRunner:
    """
    Orchestrates the training loop: env interaction, agent updates, logging.

    This is intentionally a simple, linear runner. No vectorized envs,
    no multi-processing -- clarity over speed.
    """

    def __init__(
        self,
        env: BaseEnv,
        agent: DQNAgent,
        total_steps: int = 100_000,
        eval_every: int = 5_000,
        eval_episodes: int = 10,
        log_dir: str | Path = "runs/default",
        save_every: int = 25_000,
    ) -> None:
        self.env = env
        self.agent = agent
        self.total_steps = total_steps
        self.eval_every = eval_every
        self.eval_episodes = eval_episodes
        self.save_every = save_every
        self.logger = Logger(log_dir)

    def run(self) -> None:
        state = self.env.reset()
        episode_reward = 0.0
        episode_steps = 0
        metrics: dict[str, Any] = {}

        for global_step in range(1, self.total_steps + 1):
            action = self.agent.act(state, explore=True)
            next_state, reward, terminated, truncated, info = self.env.step(action)

            self.agent.observe(
                Transition(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    terminated=terminated,
                )
            )

            learn_metrics = self.agent.learn()
            metrics.update(learn_metrics)

            episode_reward += reward
            episode_steps += 1
            state = next_state

            if terminated or truncated:
                self.logger.log_episode(
                    {
                        "episode_reward": episode_reward,
                        "episode_steps": episode_steps,
                        **{k: v for k, v in metrics.items() if isinstance(v, (int, float))},
                    }
                )
                state = self.env.reset()
                episode_reward = 0.0
                episode_steps = 0
                metrics = {}

            if global_step % self.eval_every == 0:
                eval_stats = evaluate(self.env, self.agent, self.eval_episodes)
                print(
                    f"[Eval @ step {global_step}] "
                    f"mean_reward={eval_stats['mean_reward']:.2f} "
                    f"std={eval_stats['std_reward']:.2f}"
                )

            if global_step % self.save_every == 0:
                save_path = Path(self.logger.log_dir) / f"checkpoint_{global_step}.pt"
                self.agent.save(save_path)

        self.logger.close()
