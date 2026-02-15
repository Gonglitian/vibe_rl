# vibe-rl

Pure JAX reinforcement learning experiments using Equinox, Optax, and Gymnax.

## Installation

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

### CPU (development)

```bash
uv sync
```

### GPU (CUDA 12)

```bash
uv sync
uv pip install --upgrade "jax[cuda12]"
```

### TPU

```bash
uv sync
uv pip install --upgrade "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

### With environments and checkpointing

```bash
uv sync --extra envs --extra checkpoint
```

### All dependencies (including dev)

```bash
uv sync --all-extras
```

## Core Stack

| Package | Role |
|---------|------|
| **JAX** | XLA-accelerated numerical computing |
| **Equinox** | Neural network library (PyTree-native) |
| **Optax** | Gradient-based optimizers |
| **Chex** | Testing utilities and shape assertions |
| **Distrax** | Probability distributions for JAX |
| **Gymnax** | Pure JAX RL environments |
| **Brax** | Continuous control environments |
| **Orbax** | Production checkpointing |
