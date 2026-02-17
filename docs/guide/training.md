# Training

vibe-rl provides two training-loop paradigms — **PureJaxRL** for on-policy algorithms and **Hybrid** for off-policy algorithms — unified behind a single CLI entry point and a shared `RunnerConfig`.

## Two paradigms

| Paradigm | Algorithms | Loop style | Replay buffer |
|----------|-----------|------------|---------------|
| **PureJaxRL** | PPO | `jax.lax.scan` (single JIT) | None |
| **Hybrid** | DQN, SAC | Python `for` + jitted inner steps | NumPy ring buffer |

**PureJaxRL** compiles the entire collect → update pipeline into one `jax.lax.scan`. After a one-time JIT compilation, the hardware runs at full speed with zero Python overhead. The trade-off: no I/O is possible mid-training — metrics are batch-written after the scan completes.

**Hybrid** loops use a Python `for` over timesteps for replay-buffer management, logging callbacks, and periodic checkpointing. The hot path (action selection + env step + gradient update) is `@jax.jit`-compiled, so Python overhead is minimal.

## Unified entry point — `scripts/train.py`

The simplest way to train is through the unified CLI:

```bash
python scripts/train.py cartpole_ppo
python scripts/train.py cartpole_dqn
python scripts/train.py pendulum_sac
```

Each command selects a **preset** — a pre-tuned `TrainConfig` bundling the environment, algorithm hyperparameters, and runner settings. Override any field from the command line:

```bash
python scripts/train.py cartpole_ppo --algo.lr 1e-3
python scripts/train.py pendulum_sac --runner.total_timesteps 500000
python scripts/train.py cartpole_ppo --algo.hidden_sizes '(128, 128)'
```

The script:
1. Creates a `RunDir` for the experiment (e.g. `runs/CartPole-v1_ppo_20260216_143022/`).
2. Saves a frozen `config.json` snapshot.
3. Dispatches to `train_ppo`, `train_dqn`, or `train_sac`.
4. Auto-plots a reward curve to `artifacts/reward_curve.png` (when `runner.plot=True` and matplotlib is installed).

See [Configuration](./configuration.md) for the full preset list and CLI override syntax.

## PPO training (PureJaxRL)

**File:** `src/vibe_rl/runner/train_ppo.py`

The PPO runner compiles the entire collect-rollout → compute-GAE → mini-batch-SGD pipeline into a single `jax.lax.scan`:

```python
from vibe_rl.algorithms.ppo import PPOConfig
from vibe_rl.env import make
from vibe_rl.env.wrappers import AutoResetWrapper
from vibe_rl.runner import RunnerConfig, train_ppo

env, env_params = make("CartPole-v1")
env = AutoResetWrapper(env)  # required for lax.scan loop

ppo_config = PPOConfig(n_steps=128, hidden_sizes=(64, 64))
runner_config = RunnerConfig(total_timesteps=100_000)

train_state, metrics = train_ppo(
    env, env_params,
    ppo_config=ppo_config,
    runner_config=runner_config,
)
```

::: warning
The environment **must** be wrapped with `AutoResetWrapper` (or equivalent) so that episodes auto-reset inside the scan body.
:::

### Vectorized environments

When `ppo_config.num_envs > 1`, the runner uses `jax.vmap` to run N environments in parallel. Total timesteps per update becomes `n_steps * num_envs`:

```python
ppo_config = PPOConfig(n_steps=128, num_envs=8, hidden_sizes=(64, 64))
train_state, metrics = train_ppo(
    env, env_params,
    ppo_config=ppo_config,
    runner_config=runner_config,
)
```

### Scan body (one iteration = one update)

Each scan iteration performs:

1. **`PPO.collect_rollout()`** — step the env `n_steps` times, collecting `(obs, action, reward, done, log_prob, value)` transitions.
2. **`PPO.update()`** — compute GAE advantages, then run `n_epochs` of mini-batch SGD over `n_minibatches` random slices.

### Return types

- **`PPOTrainState`** — final agent state, env state, observations, and RNG key.
- **`PPOMetricsHistory`** — per-update arrays of shape `(n_updates,)`: `total_loss`, `actor_loss`, `critic_loss`, `entropy`, `approx_kl`.

### Metrics I/O

Since `lax.scan` forbids side effects, PPO batch-writes all metrics to JSONL **after** the scan completes. Pass a `run_dir` to enable:

```python
from vibe_rl.run_dir import RunDir

run_dir = RunDir("ppo_cartpole")
train_state, metrics = train_ppo(
    env, env_params,
    ppo_config=ppo_config,
    runner_config=runner_config,
    run_dir=run_dir,
)
# Metrics written to runs/.../logs/metrics.jsonl
```

## DQN training (Hybrid)

**File:** `src/vibe_rl/runner/train_dqn.py`

DQN uses a mutable replay buffer that cannot live inside `lax.scan`. The loop splits into:

- **Outer loop**: Python `for step in range(...)` — pushes transitions into the buffer, handles callbacks, manages checkpoints.
- **Inner step**: `@jax.jit`-compiled `_act_and_step()` — epsilon-greedy action selection + env step with no Python overhead.

```python
from vibe_rl.algorithms.dqn import DQNConfig
from vibe_rl.runner import RunnerConfig, train_dqn

result = train_dqn(
    env, env_params,
    dqn_config=DQNConfig(),
    runner_config=RunnerConfig(total_timesteps=50_000),
)
# result.agent_state, result.episode_returns, result.metrics_log
```

### Loop structure

```
for step in 1..total_timesteps:
    1. _act_and_step()          (jitted: epsilon-greedy + env.step)
    2. buffer.push(transition)  (numpy)
    3. Track episode returns    (on done)
    4. if buffer >= warmup_steps:
       a. batch = buffer.sample(batch_size)
       b. DQN.update(batch)    (jitted: TD loss + target network)
    5. Periodic checkpoint      (ckpt_mgr.save)
    6. Periodic metrics log     (MetricsLogger.write)
```

**Metrics logged**: `loss`, `q_mean`, `epsilon` (every `log_interval` steps), plus `episode_return` and `episode_length` on each episode completion.

## SAC training (Hybrid)

**File:** `src/vibe_rl/runner/train_sac.py`

SAC follows the same hybrid structure as DQN, but uses continuous actions and a `_ContinuousReplayBuffer` with float32 action storage:

```python
from vibe_rl.algorithms.sac import SACConfig
from vibe_rl.runner import RunnerConfig, train_sac

result = train_sac(
    env, env_params,
    sac_config=SACConfig(),
    runner_config=RunnerConfig(total_timesteps=100_000),
    action_dim=1,
)
```

**Metrics logged**: `actor_loss`, `critic_loss`, `alpha`, `entropy`, `q_mean` (every `log_interval` steps), plus `episode_return` and `episode_length` on each episode completion.

## RunnerConfig

`RunnerConfig` is a frozen dataclass that controls the outer training loop. Algorithm-specific settings live in `PPOConfig`, `DQNConfig`, or `SACConfig`.

```python
from vibe_rl.runner import RunnerConfig

runner_config = RunnerConfig(
    total_timesteps=100_000,
    eval_every=5_000,
    eval_episodes=10,
    seed=42,
)
```

Key fields:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `total_timesteps` | `int` | `100_000` | Training budget in env steps |
| `eval_every` | `int` | `5_000` | Evaluate every N steps |
| `eval_episodes` | `int` | `10` | Episodes per evaluation |
| `log_interval` | `int` | `1_000` | Log metrics every N steps |
| `buffer_size` | `int` | `100_000` | Replay buffer capacity (off-policy only) |
| `warmup_steps` | `int` | `1_000` | Random steps before training (off-policy only) |
| `seed` | `int` | `0` | PRNG seed |
| `plot` | `bool` | `True` | Auto-generate reward curve after training |

For checkpointing fields (`checkpoint_dir`, `resume`, etc.), see [Checkpointing](./checkpointing.md). For multi-device fields (`num_devices`, `fsdp_devices`), see [Multi-GPU](./multi-gpu.md). The full field reference is in [Configuration](./configuration.md).

## Evaluation

**File:** `src/vibe_rl/runner/evaluator.py`

The evaluator runs a greedy policy across multiple episodes in parallel using `jax.vmap` + `jax.lax.while_loop` — fully JIT-compiled with no Python loops:

```python
from vibe_rl.runner import evaluate, jit_evaluate

eval_metrics = evaluate(
    act_fn,         # (agent_state, obs) -> action
    agent_state,
    env, env_params,
    n_episodes=10,
    max_steps=500,
    rng=jax.random.PRNGKey(99),
)

print(eval_metrics.mean_return)   # scalar
print(eval_metrics.std_return)    # scalar
print(eval_metrics.mean_length)   # scalar
```

`act_fn` should be a pure greedy action selector: `(agent_state, obs) -> action`. It must be JIT-compatible and should **not** update agent state.

A pre-jitted convenience `jit_evaluate` is also exported for cases where you want to skip the explicit `jax.jit` wrapper.

### EvalMetrics

| Field | Description |
|-------|-------------|
| `mean_return` | Mean episode return across all episodes |
| `std_return` | Standard deviation of episode returns |
| `mean_length` | Mean episode length |

## Callbacks (DQN / SAC)

The DQN and SAC runners accept an optional `callback` parameter that is called every `log_interval` steps:

```python
def my_callback(step: int, agent_state, metrics: dict) -> None:
    if step % 10_000 == 0:
        print(f"Step {step}: {metrics}")

result = train_dqn(
    env, env_params,
    dqn_config=DQNConfig(),
    runner_config=RunnerConfig(total_timesteps=50_000),
    callback=my_callback,
)
```

The callback signature is `callback(step, agent_state, metrics_dict)`. The `metrics_dict` contains the same fields that are written to JSONL (e.g. `loss`, `q_mean`, `epsilon` for DQN).

::: tip
PPO does not support callbacks because the entire loop runs inside `lax.scan`. Use the returned `PPOMetricsHistory` for post-training analysis instead.
:::

## Algorithm-specific scripts

For direct access without the preset system, each algorithm has a dedicated training function:

```python
from vibe_rl.runner import train_ppo, train_dqn, train_sac
```

| Function | Algorithm | Paradigm | Return type |
|----------|-----------|----------|-------------|
| `train_ppo` | PPO | PureJaxRL | `(PPOTrainState, PPOMetricsHistory)` |
| `train_dqn` | DQN | Hybrid | `DQNTrainResult` |
| `train_sac` | SAC | Hybrid | `SACTrainResult` |
| `train_ppo_multigpu` | PPO (multi-GPU) | GSPMD | `(PPOTrainState, PPOMetricsHistory)` |

`DQNTrainResult` and `SACTrainResult` are named tuples with fields:
- `agent_state` — final trained agent state
- `episode_returns` — list of per-episode returns
- `metrics_log` — list of training metrics dicts

## Next steps

- [Checkpointing](./checkpointing.md) — save, restore, and resume training runs
- [Multi-GPU](./multi-gpu.md) — data-parallel and FSDP training across devices
- [Metrics & Logging](./metrics.md) — JSONL, WandB, and TensorBoard backends
- [Plotting](./plotting.md) — visualize reward curves
