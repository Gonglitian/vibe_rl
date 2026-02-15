# JAX RL Environment Ecosystem Research

> Phase 1 Research for vibe_rl — February 2026

---

## Executive Summary

The JAX RL environment ecosystem has matured significantly. There are **pure JAX** implementations covering classic control (Gymnax), physics/continuous control (Brax), combinatorial optimization (Jumanji), board games (Pgx), and more. All follow a convergent **functional, stateless API** pattern that enables `jax.jit` + `jax.vmap` for massive parallelism.

**Key recommendation**: vibe_rl should adopt a **pure JAX-first** strategy with a Gymnax-compatible functional API. Gymnasium wrapper support should be secondary/optional. Our toy environments should be written in pure JAX.

---

## 1. Gymnax

**Repository**: [RobertTLange/gymnax](https://github.com/RobertTLange/gymnax) | ~860 stars | Active (v0.0.9, May 2025)

### What It Is
Pure JAX reimplementation of classic RL environments. The primary library for simple/classic environments in the JAX ecosystem.

### Environments (28+)
| Category | Environments |
|----------|-------------|
| Classic Control | CartPole-v1, Pendulum-v1, Acrobot-v1, MountainCar-v0, MountainCarContinuous-v0 |
| Bsuite | Catch, DeepSea, MemoryChain, UmbrellaChain, DiscountingChain, MNISTBandit |
| MinAtar | Breakout, Asterix, Freeway, SpaceInvaders, Seaquest |
| Misc | FourRooms, MetaMaze, PointRobot, Reacher, Swimmer, Pong, Bandits |

### API Interface

**Core difference from Gymnasium**: functional, stateless design with explicit RNG key threading.

```python
import jax
import gymnax

# Create environment + default parameters
env, env_params = gymnax.make("CartPole-v1")

# Reset (requires RNG key, returns state explicitly)
rng = jax.random.PRNGKey(0)
rng, key_reset = jax.random.split(rng)
obs, state = env.reset(key_reset, env_params)

# Step (requires RNG key + explicit state, no hidden mutation)
rng, key_step = jax.random.split(rng)
obs, state, reward, done, info = env.step(key_step, state, action, env_params)
```

**Key API differences vs Gymnasium**:
| Aspect | Gymnasium | Gymnax |
|--------|-----------|--------|
| State | Hidden in `env` object | Explicit `EnvState` passed around |
| RNG | Internal `np_random` seed | Explicit `jax.random.PRNGKey` per call |
| Params | Set at construction | Separate `EnvParams` dataclass, can be vmapped |
| Return | `(obs, reward, terminated, truncated, info)` | `(obs, state, reward, done, info)` |
| Mutation | `env.step()` mutates internal state | Pure function, no side effects |

### JIT + vmap

```python
# JIT the entire training loop
@jax.jit
def train_step(runner_state):
    obs, state, rng = runner_state
    rng, key = jax.random.split(rng)
    action = policy(obs)
    obs, state, reward, done, info = env.step(key, state, action, env_params)
    return (obs, state, rng)

# vmap across N parallel environments
batch_reset = jax.vmap(env.reset, in_axes=(0, None))
batch_step = jax.vmap(env.step, in_axes=(0, 0, 0, None))

keys = jax.random.split(rng, num_envs)
obs_batch, state_batch = batch_reset(keys, env_params)

# Nested vmap for meta-learning (vmap over env_params too)
meta_reset = jax.vmap(jax.vmap(env.reset, in_axes=(0, None)), in_axes=(0, 0))
```

### Performance
- CartPole: **920x speedup** vs Gymnasium (46s → 0.05s for 1M steps on A100)
- MinAtar-Breakout: **250x speedup**
- With 2000 parallel envs on A100: **1M+ steps/second**

### Limitations
- No continuous control locomotion (HalfCheetah, Walker2d, Ant → use Brax)
- No full Atari (only MinAtar simplified versions)
- Limited rendering support

---

## 2. Brax

**Repository**: [google/brax](https://github.com/google/brax) | ~3.1k stars | v0.14.0 (Dec 2025)

### What It Is
Google's differentiable physics engine in JAX for continuous control RL. Reimplements MuJoCo-style locomotion tasks.

### Environments
Ant, HalfCheetah, Humanoid, Hopper, Walker2d, Reacher, Fetch, Grasp. Also supports loading custom MJCF/URDF models.

### Physics Backends
| Backend | Description | Recommendation |
|---------|-------------|----------------|
| **MJX** (MuJoCo XLA) | JAX port of MuJoCo physics | **Recommended** for new code |
| Generalized | Featherstone's algorithm | Most accurate native backend |
| Positional | Position-based dynamics | Fast but less accurate |
| Spring | Spring-damper constraints | Fastest, least stable |

> **Important**: As of v0.13.0, only `brax.training` is actively maintained. For environments, Google recommends MuJoCo Playground. For physics, use MJX directly.

### API

```python
from brax import envs
import jax

env = envs.create(env_name="ant", backend="mjx")
reset_fn = jax.jit(env.reset)
step_fn = jax.jit(env.step)

state = reset_fn(jax.random.PRNGKey(0))
# state.obs, state.reward, state.done, state.metrics

state = step_fn(state, action)

# Vectorized
batch_reset = jax.jit(jax.vmap(env.reset))
states = batch_reset(jax.random.split(rng, 1000))
```

**API difference from Gymnax**: Brax returns a single `State` object (not a tuple). Step takes `(state, action)` rather than `(key, state, action, params)`. Brax doesn't separate `EnvParams` — parameters are baked into the environment at creation.

### Performance
- 100-1000x faster than CPU MuJoCo
- ~200M steps/sec on TPUv3
- Fully JIT/vmap/pmap compatible

### For vibe_rl
Brax is relevant if we want continuous control tasks. Its API is similar but not identical to Gymnax. A thin adapter layer would be needed.

---

## 3. Gymnasium Wrapper Strategy

### The Problem
Gymnasium environments are fundamentally incompatible with JAX:
- **Stateful OOP** (hidden internal state) vs JAX's **pure functional** requirement
- **NumPy arrays** vs `jax.numpy` arrays
- **Not JIT-compilable** (Python control flow, side effects)

### numpy ↔ jax.numpy Conversion Overhead
- The conversion itself is cheap (`jax.numpy.array(np_array)` is near-zero cost)
- The **real bottleneck is CPU→GPU transfer**: PCIe bandwidth (~16 GB/s) is orders of magnitude slower than GPU compute
- This transfer happens **every step** when using a Gymnasium wrapper
- **Measured impact**: 100-1000x slower than pure JAX end-to-end

### Existing Solutions

| Solution | Approach | Performance |
|----------|----------|-------------|
| Gymnasium `JaxToNumpy` wrapper | Official wrapper, converts at boundary | Still CPU-bound |
| EnvPool | C++ parallel environment vectorization | ~1M Atari frames/sec, ~3M MuJoCo steps/sec |
| Async workers | CPU threads step envs, GPU trains | Hides latency, doesn't eliminate it |

### Recommendation for vibe_rl
**Do NOT invest heavily in Gymnasium wrapping.** The performance gap is 100-1000x. Provide at most a thin convenience wrapper for users who need it, but make the pure JAX path the primary API.

```
Pure JAX path:  env.step() → jit/vmap → 1M+ steps/sec
Gymnasium path: env.step() → numpy → jax.numpy → GPU transfer → ~1K steps/sec
```

---

## 4. Other JAX Environment Libraries

### Jumanji (InstaDeep)
**Repository**: [instadeepai/jumanji](https://github.com/instadeepai/jumanji) | ~806 stars | ICLR 2024

Combinatorial optimization and routing problems: BinPack, TSP, CVRP, Knapsack, Maze, RubiksCube, Game2048, Tetris, Snake, etc. (22 environments)

- Hybrid API inspired by both Gym and DeepMind dm_env
- Uses `TimeStep` structure with `step_type`, `reward`, `discount`, `observation`
- Full JIT/vmap support
- Provides wrappers for Gymnasium, dm_env, Acme, Stable Baselines3

### Pgx
**Repository**: [sotetsuk/pgx](https://github.com/sotetsuk/pgx) | ~586 stars | NeurIPS 2023

Board/strategy games: Chess, Go (9x9, 19x19), Shogi, Othello, Hex, Connect Four, Backgammon, poker variants, MinAtar. Functional JAX API, fully JIT/vmap compatible.

### Other Notable Libraries

| Library | Domain | Speedup | Notes |
|---------|--------|---------|-------|
| **Craftax** | Open-ended (Crafter+NetHack) | 250x | ICML 2024 Spotlight |
| **NAVIX** | MiniGrid in JAX | 200,000x (batched) | Drop-in MiniGrid replacement |
| **XLand-MiniGrid** | Meta-RL benchmarks | 10x vs async Gymnasium | NeurIPS 2023 |
| **JaxMARL** | Multi-agent RL | 12,500x | Hanabi, Overcooked, SMAX |
| **Mctx** | MCTS algorithms (not envs) | N/A | AlphaZero/MuZero in JAX |

---

## 5. API Convergence: The De Facto Standard

There is **no formal standard**, but a clear **convergent pattern** across all JAX env libraries:

```python
# Universal pattern:
env = make_env(name)                         # Create environment
obs, state = env.reset(rng_key, params?)     # Functional reset
obs, state, reward, done, info = env.step(   # Functional step
    rng_key, state, action, params?
)
```

**Common principles**:
1. **Explicit state**: No hidden mutation. State is a PyTree passed in/out.
2. **RNG key threading**: Every stochastic operation takes a `jax.random.PRNGKey`.
3. **JIT/vmap composable**: All functions are pure, enabling JAX transformations.
4. **Params separate from state** (Gymnax pattern): Enables vmapping over different environment configs.

**Minor divergences**:
- Gymnax: `(key, state, action, params)` → `(obs, state, reward, done, info)`
- Brax: `(state, action)` → `State` object (key/params baked in)
- Jumanji: Returns `TimeStep` (dm_env style) instead of tuple

---

## 6. Recommendations for vibe_rl

### Recommendation 1: Pure JAX-First Strategy

**Primary**: Support pure JAX environments natively (Gymnax, Brax, Jumanji, Pgx, etc.)
**Secondary**: Optional thin Gymnasium wrapper for convenience (not performance-critical path)

**Rationale**: The 100-1000x performance gap makes Gymnasium wrapping a second-class citizen. The entire JAX RL ecosystem is converging on pure JAX environments.

### Recommendation 2: Adopt Gymnax-Style API

Our environment interface should follow the Gymnax pattern:

```python
class Environment:
    def reset(self, key: PRNGKey, params: EnvParams) -> Tuple[Obs, EnvState]:
        ...

    def step(self, key: PRNGKey, state: EnvState, action: Action, params: EnvParams
    ) -> Tuple[Obs, EnvState, float, bool, dict]:
        ...
```

**Why Gymnax over Brax API**:
- More explicit (key + params visible)
- Separate `EnvParams` enables meta-learning (vmap over params)
- Tuple returns are simpler than Brax's `State` object
- Gymnax is the most widely adopted API for non-physics environments

**Brax adapter**: Provide a thin wrapper to adapt Brax environments to our API.

### Recommendation 3: Rewrite Toy Environments in Pure JAX

**Yes, rewrite.** Our built-in toy environments (CartPole, etc.) should be pure JAX:

- Enables the full `jit` + `vmap` + `lax.scan` pipeline
- No numpy↔jax conversion overhead
- Users get the full speedup out of the box
- Reference implementation for custom environments

A CartPole in pure JAX is ~50 lines. The investment is minimal and the payoff is enormous.

### Recommendation 4: Environment Registry

```python
import vibe_rl

# Built-in pure JAX environments
env, params = vibe_rl.make("CartPole-v1")

# Gymnax environments (if installed)
env, params = vibe_rl.make("gymnax/Pendulum-v1")

# Brax environments (if installed, adapted to our API)
env, params = vibe_rl.make("brax/ant")

# Gymnasium fallback (slow, for compatibility)
env, params = vibe_rl.make("gymnasium/LunarLander-v3")
```

### Environment Priority for vibe_rl

| Priority | Source | Environments | Notes |
|----------|--------|-------------|-------|
| P0 | Built-in (pure JAX) | CartPole, Pendulum, MountainCar | Ship with library |
| P1 | Gymnax | All 28 classic/bsuite/MinAtar | First-class integration |
| P2 | Brax | Ant, HalfCheetah, Humanoid | Adapter wrapper needed |
| P3 | Jumanji/Pgx | Domain-specific | Optional extensions |
| P4 | Gymnasium | Everything else | Slow path, convenience only |

---

## Summary Table

| Library | Domain | API Style | JIT/vmap | Speedup | Stars | Status |
|---------|--------|-----------|----------|---------|-------|--------|
| **Gymnax** | Classic control, bsuite, MinAtar | Functional (key, state, action, params) | Full | 250-920x | 858 | Active |
| **Brax** | Continuous control (MuJoCo-like) | Functional (state, action) | Full | 100-1000x | 3.1k | Training only* |
| **Jumanji** | Combinatorial optimization | Hybrid (Gym + dm_env) | Full | N/A | 806 | Active |
| **Pgx** | Board games | Functional | Full | 10-100x | 586 | Active |
| **Craftax** | Open-ended RL | Gymnax-compatible | Full | 250x | N/A | Active |
| **JaxMARL** | Multi-agent RL | Functional | Full | 12,500x | N/A | Active |
| **NAVIX** | MiniGrid | Functional | Full | 200,000x | 157 | Active |

*Brax physics simulation deprecated in favor of MJX; only `brax.training` actively maintained.
