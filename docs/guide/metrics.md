# Metrics & Logging

vibe-rl provides structured metrics logging via JSONL files with optional fan-out to WandB and TensorBoard, plus compact console logging for training progress.

**Source:** `src/vibe_rl/metrics.py`

## MetricsLogger

`MetricsLogger` writes one JSON object per line to a JSONL file. Each line is self-describing — fields can vary between entries, so you can log different metrics at different stages of training.

```python
from vibe_rl.run_dir import RunDir
from vibe_rl.metrics import MetricsLogger

run = RunDir("dqn_cartpole")
logger = MetricsLogger(run.log_path())  # logs/metrics.jsonl

logger.write({"step": 1000, "loss": 0.42, "reward": 195.0})
logger.write({"step": 2000, "loss": 0.31, "episode_return": 200.0})
logger.close()
```

### Automatic fields

- **`wall_time`** — seconds since logger creation, added automatically if not present in the record. Useful for correlating training speed with metrics.

### JAX/numpy conversion

JAX arrays and numpy scalars are auto-converted to Python floats before JSON serialization, so you can pass `jnp.ndarray` or `np.floating` values directly:

```python
import jax.numpy as jnp

loss = jnp.array(0.42)
logger.write({"step": 1000, "loss": loss})  # loss stored as 0.42
```

### Context manager

`MetricsLogger` supports `with` statements for automatic cleanup:

```python
with MetricsLogger(run.log_path()) as logger:
    for step in range(1000):
        logger.write({"step": step, "loss": compute_loss()})
# file handle and backends auto-closed
```

### Where metrics live

Inside a `RunDir`, metrics are stored at `logs/metrics.jsonl`:

```
runs/dqn_cartpole_20260216_143022/
├── logs/
│   └── metrics.jsonl    ← MetricsLogger writes here
├── checkpoints/
├── videos/
└── artifacts/
```

The path is created via `run.log_path()`, which returns `<run_root>/logs/metrics.jsonl` by default. Parent directories are created automatically.

## Reading metrics back

Use `read_metrics()` to load all records from a JSONL file:

```python
from vibe_rl.metrics import read_metrics

records = read_metrics("runs/dqn_cartpole_20260216_143022/logs/metrics.jsonl")
# [{"step": 1000, "loss": 0.42, "reward": 195.0, "wall_time": 12.345}, ...]
```

Returns an empty list if the file doesn't exist. Each record is a plain Python dict.

## Backends

Backends fan out every `write()` call to an external service. Pass them as a list when creating the logger:

```python
logger = MetricsLogger(run.log_path(), backends=[backend1, backend2])
```

Any object implementing the `LogBackend` protocol works:

```python
class LogBackend(Protocol):
    def log(self, record: dict[str, Any], step: int | None = None) -> None: ...
    def close(self) -> None: ...
```

### WandbBackend

Fan out metrics to [Weights & Biases](https://wandb.ai). Requires `pip install wandb`.

```python
from vibe_rl.metrics import MetricsLogger, WandbBackend

backend = WandbBackend(
    project="rl",
    entity="my-team",
    config={"algo": "dqn", "lr": 1e-3},
    run_dir=run.root,   # persists run ID for resumption
)
logger = MetricsLogger(run.log_path(), backends=[backend])
```

**Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `project` | `"vibe_rl"` | W&B project name |
| `entity` | `None` | W&B entity (user or team); `None` uses default |
| `config` | `None` | Experiment config dict logged at init |
| `run_dir` | `None` | Directory for persisting run ID (`wandb_id.txt`) |
| `**init_kwargs` | — | Extra kwargs passed to `wandb.init()` |

**Auto-resume:** When `run_dir` is provided, the W&B run ID is saved to `<run_dir>/wandb_id.txt`. On the next launch, if that file exists, the previous run is automatically resumed with `resume="must"`.

### TensorBoardBackend

Fan out scalar metrics to TensorBoard. Requires `pip install tensorboardX` (or `torch.utils.tensorboard`).

```python
from vibe_rl.metrics import TensorBoardBackend

backend = TensorBoardBackend(log_dir=run.logs)
logger = MetricsLogger(run.log_path(), backends=[backend])
```

All numeric values (`int` and `float`) in each `write()` call are recorded as TensorBoard scalars. Non-numeric fields are silently skipped.

View with:

```bash
tensorboard --logdir runs/dqn_cartpole_20260216_143022/logs
```

### Combining backends

You can attach multiple backends simultaneously:

```python
logger = MetricsLogger(
    run.log_path(),
    backends=[
        WandbBackend(project="rl", run_dir=run.root),
        TensorBoardBackend(log_dir=run.logs),
    ],
)
```

Every `write()` call writes to the JSONL file **and** fans out to all backends.

## Resuming a WandB run

The `resume_wandb()` convenience function creates a `WandbBackend` that resumes a previous run:

```python
from vibe_rl.metrics import resume_wandb, MetricsLogger

backend = resume_wandb(run.root, project="rl")
logger = MetricsLogger(run.log_path(), backends=[backend])
```

It reads the run ID from `<run_dir>/wandb_id.txt` (written automatically when `run_dir` is passed to `WandbBackend`). If no ID file exists, a fresh run is started.

**Typical resume workflow:**

```python
from vibe_rl.run_dir import RunDir
from vibe_rl.metrics import MetricsLogger, resume_wandb

# Re-open an existing run directory
run = RunDir(run_id="dqn_cartpole_20260215_143022")

# Resume WandB and continue logging
backend = resume_wandb(run.root, project="rl")
logger = MetricsLogger(run.log_path(), backends=[backend])

# New metrics append to the same JSONL and WandB run
logger.write({"step": 50000, "loss": 0.12, "reward": 200.0})
```

## Console logging

### setup_logging

`setup_logging()` configures the `vibe_rl` logger with a compact formatter:

```python
from vibe_rl.metrics import setup_logging

setup_logging()  # installs compact formatter on "vibe_rl" logger
```

**Output format:** `{level} {timestamp}.{ms} [{logger_name}] {message}`

```
I 2026-02-15 14:30:22.123 [vibe_rl] Starting training
W 2026-02-15 14:30:25.456 [vibe_rl] Gradient norm exceeded threshold
```

**Level abbreviations:**

| Abbreviation | Level |
|:---:|-------|
| `D` | DEBUG |
| `I` | INFO |
| `W` | WARNING |
| `E` | ERROR |
| `C` | CRITICAL |

Safe to call multiple times — existing handlers are replaced, not duplicated. Pass `level=logging.DEBUG` to see debug output.

### log_step_progress

`log_step_progress()` emits a single-line training progress message:

```python
from vibe_rl.metrics import setup_logging, log_step_progress

setup_logging()

log_step_progress(5000, 100_000, metrics={"loss": 0.42, "reward": 195.0})
# I 2026-02-15 14:30:22.123 [vibe_rl] step 5000/100000 (5.0%) | loss=0.42 reward=195
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `step` | `int` | Current training step |
| `total_steps` | `int` | Total planned training steps |
| `metrics` | `dict` (optional) | Scalar metrics to append; `step` and `wall_time` keys are excluded |
| `logger_name` | `str` | Logger name (default `"vibe_rl"`) |

## Integration with training runners

The training runners (`train_ppo`, `train_sac`, `train_dqn`) accept a `run_dir` parameter. When provided, the runner automatically creates a `MetricsLogger` and writes metrics at the configured `log_interval`:

```python
from vibe_rl.run_dir import RunDir
from vibe_rl.metrics import setup_logging

setup_logging()

run = RunDir("ppo_cartpole")
run.save_config(config)

# Runner handles MetricsLogger internally
train_state, metrics = train_ppo(
    config=config,
    run_dir=run,
)
```

**Algorithm-specific metrics logged:**

| Algorithm | Metrics |
|-----------|---------|
| PPO | `actor_loss`, `critic_loss`, `entropy`, `approx_kl`, `clip_fraction` |
| SAC | `actor_loss`, `critic_loss`, `alpha`, `entropy`, `q_mean` |
| DQN | `loss`, `q_mean`, `epsilon` |

## Complete example

A full training setup with console logging, JSONL metrics, and WandB:

```python
from vibe_rl.run_dir import RunDir
from vibe_rl.metrics import (
    MetricsLogger,
    WandbBackend,
    setup_logging,
    log_step_progress,
    read_metrics,
)

# 1. Set up console logging
setup_logging()

# 2. Create run directory
run = RunDir("dqn_cartpole")
run.save_config({"algo": "dqn", "lr": 1e-3})

# 3. Create logger with WandB backend
backend = WandbBackend(
    project="rl",
    config={"algo": "dqn", "lr": 1e-3},
    run_dir=run.root,
)
logger = MetricsLogger(run.log_path(), backends=[backend])

# 4. Training loop
for step in range(0, 100_000, 1000):
    loss = train_step()
    reward = evaluate()

    logger.write({"step": step, "loss": loss, "reward": reward})
    log_step_progress(step, 100_000, metrics={"loss": loss, "reward": reward})

logger.close()

# 5. Read back for analysis
records = read_metrics(run.log_path())
final_reward = records[-1]["reward"]
```
