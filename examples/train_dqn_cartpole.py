"""Train DQN on CartPole."""

from vibe_rl.algorithms.dqn.agent import DQNAgent
from vibe_rl.algorithms.dqn.config import DQNConfig
from vibe_rl.env.cart_pole import CartPoleEnv
from vibe_rl.runner.training_runner import TrainingRunner
from vibe_rl.utils.seeding import seed_everything


def main() -> None:
    seed_everything(42)

    env = CartPoleEnv(max_steps=500)
    config = DQNConfig(
        hidden_sizes=(128, 128),
        lr=1e-3,
        gamma=0.99,
        batch_size=64,
        buffer_capacity=50_000,
        min_buffer_size=1_000,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay_steps=20_000,
        target_update_freq=1_000,
    )
    agent = DQNAgent(
        obs_shape=env.observation_space.shape,
        n_actions=env.action_space.n,
        config=config,
    )
    runner = TrainingRunner(
        env=env,
        agent=agent,
        total_steps=100_000,
        eval_every=10_000,
        log_dir="runs/dqn_cartpole",
    )
    runner.run()
    print("Training complete.")


if __name__ == "__main__":
    main()
