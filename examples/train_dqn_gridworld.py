"""Train DQN on a 5x5 GridWorld."""

from vibe_rl.algorithms.dqn.agent import DQNAgent
from vibe_rl.algorithms.dqn.config import DQNConfig
from vibe_rl.env.grid_world import GridWorldEnv
from vibe_rl.runner.training_runner import TrainingRunner
from vibe_rl.utils.seeding import seed_everything


def main() -> None:
    seed_everything(42)

    env = GridWorldEnv(size=5, max_steps=100)
    config = DQNConfig(
        hidden_sizes=(64, 64),
        lr=5e-4,
        buffer_capacity=10_000,
        min_buffer_size=500,
        epsilon_decay_steps=10_000,
        target_update_freq=500,
    )
    agent = DQNAgent(
        obs_shape=env.observation_space.shape,
        n_actions=env.action_space.n,
        config=config,
    )
    runner = TrainingRunner(
        env=env,
        agent=agent,
        total_steps=30_000,
        eval_every=5_000,
        log_dir="runs/dqn_gridworld",
    )
    runner.run()
    print("Training complete.")


if __name__ == "__main__":
    main()
