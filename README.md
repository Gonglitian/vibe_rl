<p align="center">
  <h1 align="center">vibe-rl</h1>
  <p align="center">
    <strong>Pure JAX reinforcement learning — fast, functional, and hackable.</strong>
  </p>
  <p align="center">
    <a href="#quick-start">Quick Start</a> &bull;
    <a href="#algorithms">Algorithms</a> &bull;
    <a href="#environments">Environments</a> &bull;
    <a href="#architecture">Architecture</a> &bull;
    <a href="#multi-gpu">Multi-GPU</a>
  </p>
</p>

---

**vibe-rl** is a lightweight reinforcement learning library built entirely on JAX. Every training loop compiles end-to-end with `jax.jit`, scales across devices with `jax.pmap`, and vectorizes environments with `jax.vmap` — no Python loop overhead, no framework magic, just pure functions and XLA.

## Highlights

- **End-to-end JIT** — Rollout collection, GAE, and mini-batch SGD all run inside `jax.lax.scan`. One compilation, zero Python overhead.
- **Multi-GPU / TPU** — Single-line `jax.pmap` scaling with automatic gradient synchronization via `lax.pmean`.
- **Vectorized environments** — Run thousands of parallel environments on a single GPU with `jax.vmap`.
- **Purely functional** — No mutable state. Every function is a pure transform: `(state, input) → (state, output)`.
- **Equinox neural networks** — PyTree-native models that compose naturally with JAX transforms.
- **~2,700 lines of code** — Small enough to read in an afternoon, complete enough to train real agents.

## Algorithms

| Algorithm | Type | Action Space | Key Features |
|-----------|------|-------------|--------------|
| **PPO** | On-policy | Discrete | Clipped surrogate, GAE, mini-batch SGD, multi-GPU |
| **DQN** | Off-policy | Discrete | Target network, epsilon-greedy, replay buffer |
| **SAC** | Off-policy | Continuous | Twin Q-networks, auto-tuned temperature, tanh squashing |

All algorithms expose a uniform functional API:

```python
state = Algorithm.init(rng, obs_shape, n_actions, config=config)
action, state = Algorithm.act(state, obs, config=config)
state, metrics = Algorithm.update(state, batch, config=config)
```

## Environments

### Built-in (Pure JAX)

| Environment | Obs Dim | Action Space | Description |
|-------------|---------|-------------|-------------|
| `CartPole-v1` | 4 | Discrete(2) | Classic balance task |
| `Pendulum-v1` | 3 | Continuous(1) | Swing-up control |
| `GridWorld-v0` | 2 | Discrete(4) | Grid navigation |

All environments follow the functional Gymnax-style API:

```python
from vibe_rl.env import make

env, params = make("CartPole-v1")
obs, state = env.reset(key, params)
obs, state, reward, done, info = env.step(key, state, action, params)
```

### Wrappers

| Wrapper | Purpose |
|---------|---------|
| `AutoResetWrapper` | Auto-resets on done (essential for `lax.scan` loops) |
| `ObsNormWrapper` | Running mean/variance normalization (Welford's algorithm) |
| `RewardScaleWrapper` | Constant reward scaling |
| `GymnasiumWrapper` | Adapter for standard Gymnasium envs (not jit-compatible) |

## Quick Start

### Installation

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
# CPU (development)
uv sync

# GPU (CUDA 12)
uv sync && uv pip install --upgrade "jax[cuda12]"

# TPU
uv sync && uv pip install --upgrade "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

# With all optional deps (environments, checkpointing, dev tools)
uv sync --all-extras
```

### Train PPO on CartPole

```python
import jax
from vibe_rl.algorithms.ppo import PPO, PPOConfig
from vibe_rl.env import make
from vibe_rl.env.wrappers import AutoResetWrapper

config = PPOConfig(hidden_sizes=(64, 64), lr=2.5e-4, n_steps=128)

env, env_params = make("CartPole-v1")
env = AutoResetWrapper(env)
env_params = env.default_params()

rng = jax.random.PRNGKey(0)
rng, env_key, agent_key = jax.random.split(rng, 3)
obs, env_state = env.reset(env_key, env_params)
state = PPO.init(agent_key, obs_shape=(4,), n_actions=2, config=config)

for _ in range(200):
    state, trajectories, obs, env_state, last_value = PPO.collect_rollout(
        state, obs, env_state, env.step, env_params, config=config,
    )
    state, metrics = PPO.update(state, trajectories, last_value, config=config)
```

### Train SAC on Pendulum (Continuous Control)

```python
import jax
from vibe_rl.algorithms.sac import SAC, SACConfig
from vibe_rl.env import make
from vibe_rl.env.wrappers import AutoResetWrapper
from vibe_rl.dataprotocol.replay_buffer import ReplayBuffer

config = SACConfig(hidden_sizes=(256, 256), actor_lr=3e-4, critic_lr=3e-4,
                   action_low=-2.0, action_high=2.0, autotune_alpha=True)

env, env_params = make("Pendulum-v1")
env = AutoResetWrapper(env)
state = SAC.init(jax.random.PRNGKey(0), obs_shape=(3,), action_dim=1, config=config)
buffer = ReplayBuffer(capacity=50_000, obs_shape=(3,), action_shape=(1,))

# Standard off-policy loop: act → store → sample → update
```

See [`examples/`](examples/) for complete, runnable training scripts.

## Multi-GPU

Scale PPO across all available devices with zero code changes to the algorithm:

```python
from vibe_rl.runner import train_ppo_multigpu, RunnerConfig

runner_config = RunnerConfig(total_timesteps=1_000_000, num_envs=8)

# Automatically distributes across all detected devices
final_state, metrics = train_ppo_multigpu(
    env, env_params, ppo_config=config, runner_config=runner_config,
)
```

Under the hood:
- Each device runs its own vectorized environments (`jax.vmap`)
- Gradients are synchronized across devices via `jax.lax.pmean`
- Checkpoints support `unreplicate` / `replicate` for device-agnostic save/load

## Architecture

```
src/vibe_rl/
├── algorithms/
│   ├── ppo/            # PPO: agent, config, network, types
│   ├── dqn/            # DQN: agent, config, network, types
│   └── sac/            # SAC: agent, config, network, types
├── env/                # Pure-JAX environments & wrappers
├── runner/             # Training loops (single-GPU, multi-GPU)
├── dataprotocol/       # Transitions, replay buffer
├── agent/              # Agent protocol (typing interface)
├── checkpoint.py       # Orbax + Equinox checkpointing
├── metrics.py          # Training metrics
├── schedule.py         # Learning rate schedules
├── seeding.py          # PRNG utilities
└── types.py            # Core type aliases
```

### Design Principles

1. **Pure functions everywhere** — Algorithms are stateless namespaces (`PPO.init`, `PPO.act`, `PPO.update`). All state is explicit in `NamedTuple` containers.

2. **`lax.scan` over Python loops** — Rollout collection, epoch iteration, and mini-batch SGD all use `jax.lax.scan` for XLA fusion.

3. **Frozen dataclass configs** — Immutable configs (`PPOConfig`, `DQNConfig`, `SACConfig`) are safe to pass as `jit` static arguments.

4. **Equinox models** — Neural networks are `eqx.Module` instances, automatically registered as PyTrees. No special parameter handling needed.

## Core Stack

| Package | Role |
|---------|------|
| [JAX](https://github.com/jax-ml/jax) | XLA-accelerated numerical computing |
| [Equinox](https://github.com/patrick-kidger/equinox) | Neural network library (PyTree-native) |
| [Optax](https://github.com/google-deepmind/optax) | Gradient-based optimizers |
| [Distrax](https://github.com/google-deepmind/distrax) | Probability distributions |
| [Chex](https://github.com/google-deepmind/chex) | Testing utilities & shape assertions |

**Optional:**

| Package | Role |
|---------|------|
| [Gymnax](https://github.com/RobertTLange/gymnax) | Additional JAX environments |
| [Brax](https://github.com/google/brax) | Continuous control environments |
| [Orbax](https://github.com/google/orbax) | Production checkpointing |

## Development

```bash
# Install all dependencies
uv sync --all-extras

# Run tests
pytest tests/

# Run tests in parallel
pytest tests/ -n auto

# Lint
ruff check src/ tests/
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Ensure tests pass (`pytest tests/`)
4. Ensure code passes lint (`ruff check`)
5. Submit a pull request

## License

MIT
