from __future__ import annotations

import numpy as np

from vibe_rl.agent.base import BaseAgent
from vibe_rl.env.base import BaseEnv


def evaluate(env: BaseEnv, agent: BaseAgent, n_episodes: int = 10) -> dict[str, float]:
    """
    Run the agent in greedy mode (no exploration) and return aggregate stats.

    Returns:
        {"mean_reward": ..., "std_reward": ..., "mean_steps": ...}
    """
    rewards = []
    steps = []

    for _ in range(n_episodes):
        state = env.reset()
        total_reward = 0.0
        episode_steps = 0
        done = False

        while not done:
            action = agent.act(state, explore=False)
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            episode_steps += 1
            done = terminated or truncated

        rewards.append(total_reward)
        steps.append(episode_steps)

    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_steps": float(np.mean(steps)),
    }
