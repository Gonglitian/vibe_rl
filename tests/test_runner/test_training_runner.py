from vibe_rl.algorithms.dqn.agent import DQNAgent
from vibe_rl.algorithms.dqn.config import DQNConfig
from vibe_rl.env.grid_world import GridWorldEnv
from vibe_rl.runner.evaluator import evaluate
from vibe_rl.runner.training_runner import TrainingRunner


class TestEvaluator:
    def test_evaluate_returns_stats(self):
        env = GridWorldEnv(size=3, max_steps=20)
        config = DQNConfig(hidden_sizes=(16, 16), buffer_capacity=50, min_buffer_size=5)
        agent = DQNAgent(obs_shape=(2,), n_actions=4, config=config)
        stats = evaluate(env, agent, n_episodes=3)
        assert "mean_reward" in stats
        assert "std_reward" in stats
        assert "mean_steps" in stats


class TestTrainingRunner:
    def test_short_run(self, tmp_path):
        env = GridWorldEnv(size=3, max_steps=20)
        config = DQNConfig(
            hidden_sizes=(16, 16),
            buffer_capacity=200,
            min_buffer_size=50,
            batch_size=16,
        )
        agent = DQNAgent(obs_shape=(2,), n_actions=4, config=config)
        runner = TrainingRunner(
            env=env,
            agent=agent,
            total_steps=200,
            eval_every=100,
            eval_episodes=2,
            log_dir=str(tmp_path / "test_run"),
            save_every=100,
        )
        runner.run()  # should not crash
        assert (tmp_path / "test_run" / "progress.csv").exists()
        assert (tmp_path / "test_run" / "checkpoint_100.pt").exists()
