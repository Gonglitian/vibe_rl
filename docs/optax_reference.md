# Optax Comprehensive Reference for Reinforcement Learning

> **Optax version**: 0.2.6 (stable, Sep 2025) / 0.2.7.dev (HEAD)
> **Sources**: optax.readthedocs.io, github.com/google-deepmind/optax, community implementations
> **Researched**: 2026-02-15

---

## Table of Contents

1. [Standard Optimizers](#1-standard-optimizers)
2. [Gradient Clipping Combinations](#2-gradient-clipping-combinations)
3. [Learning Rate Schedules](#3-learning-rate-schedules)
4. [RL-Specific Patterns](#4-rl-specific-patterns)
5. [Advanced Features](#5-advanced-features)
6. [Best Practices by Algorithm](#6-best-practices-by-algorithm)

---

## 1. Standard Optimizers

### 1.1 Adam

```python
optax.adam(
    learning_rate: float | Schedule,
    b1: float = 0.9,       # first moment decay
    b2: float = 0.999,     # second moment decay
    eps: float = 1e-8,      # numerical stability
    eps_root: float = 0.0,  # stability inside sqrt (must be >0 for meta-learning)
    mu_dtype: Optional[dtype] = None,  # accumulator precision
    nesterov: bool = False  # Nesterov momentum variant
)
```

**When to use**: Default choice for most RL tasks. Works well for PPO, A2C, and
general policy gradient methods. Robust to hyperparameter choices.

**RL-typical values**: `learning_rate=3e-4`, `eps=1e-5` (some PPO implementations
use 1e-5 instead of the default 1e-8 for stability).

### 1.2 AdamW (Adam with Decoupled Weight Decay)

```python
optax.adamw(
    learning_rate: float | Schedule,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    mu_dtype: Optional[dtype] = None,
    weight_decay: float = 1e-4,  # decoupled weight decay coefficient
    mask: Optional[PyTree | Callable] = None  # which params get weight decay
)
```

**When to use**: When you need L2 regularization that is decoupled from the
adaptive learning rate. Important for larger networks or when overfitting
is a concern. Preferred over Adam when using weight decay because vanilla
Adam couples weight decay with the learning rate adaptation.

**RL-typical values**: `learning_rate=3e-4`, `weight_decay=1e-4` to `1e-2`.
Often used in SAC and TD3 with larger networks.

### 1.3 SGD

```python
optax.sgd(
    learning_rate: float | Schedule,
    momentum: float | None = None,  # None = no momentum
    nesterov: bool = False
)
```

**When to use**: Rarely used as primary RL optimizer. Sometimes used for
the critic in actor-critic methods where you want simple, predictable
updates. Can outperform Adam with very careful tuning. Useful as a
baseline for optimizer ablation studies.

**RL-typical values**: `learning_rate=1e-3`, `momentum=0.9` if using momentum.

### 1.4 RMSProp

```python
optax.rmsprop(
    learning_rate: float | Schedule,
    decay: float = 0.9,       # EMA decay for squared gradients
    eps: float = 1e-8,
    initial_scale: float = 0.0,
    centered: bool = False,   # True = subtract mean of squared gradients
    momentum: float | None = None,
    nesterov: bool = False
)
```

**When to use**: Traditional choice for Atari/DQN-style algorithms
(the original DQN paper used RMSProp). Still relevant for A3C/A2C
implementations. The `centered=True` variant can help with non-stationary
objectives common in RL.

**RL-typical values**: `learning_rate=7e-4`, `decay=0.99`, `eps=1e-5`,
`centered=True` (for A3C-style).

### 1.5 Quick Comparison Table

| Optimizer | LR (RL) | Memory/param | Best For |
|-----------|---------|-------------|----------|
| Adam      | 3e-4    | 2 buffers   | General RL, PPO, policy gradients |
| AdamW     | 3e-4    | 2 buffers   | Large networks, SAC with regularization |
| SGD+Mom   | 1e-3    | 1 buffer    | Simple critics, fine-tuning |
| RMSProp   | 7e-4    | 1-2 buffers | DQN, A3C, Atari |

---

## 2. Gradient Clipping Combinations

### 2.1 Available Clipping Transforms

```python
# Clip by global L2 norm (most common in RL)
optax.clip_by_global_norm(max_norm: float)

# Clip each element to [-max_delta, +max_delta]
optax.clip(max_delta: float)

# Clip by per-parameter block-wise norm
optax.clip_by_block_rms(threshold: float = 1.0)
```

### 2.2 Chaining Clipping with Optimizers

The canonical pattern uses `optax.chain()` to compose transforms sequentially.
**Order matters**: clipping should come BEFORE the optimizer scaling.

```python
import optax

# Pattern 1: Global norm clipping + Adam (most common for RL)
optimizer = optax.chain(
    optax.clip_by_global_norm(max_norm=0.5),
    optax.adam(learning_rate=3e-4),
)

# Pattern 2: Element-wise clipping + Adam
optimizer = optax.chain(
    optax.clip(max_delta=1.0),
    optax.adam(learning_rate=3e-4),
)

# Pattern 3: Clipping + manual Adam decomposition for full control
optimizer = optax.chain(
    optax.clip_by_global_norm(max_norm=1.0),
    optax.scale_by_adam(b1=0.9, b2=0.999, eps=1e-5),
    optax.scale(-3e-4),  # negative because we minimize
)

# Pattern 4: Clipping + AdamW with schedule
schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0, peak_value=3e-4,
    warmup_steps=1000, decay_steps=100_000, end_value=0.0,
)
optimizer = optax.chain(
    optax.clip_by_global_norm(max_norm=0.5),
    optax.adamw(learning_rate=schedule, weight_decay=1e-4),
)
```

### 2.3 Understanding the Decomposition

The alias `optax.adam(lr)` is equivalent to:
```python
optax.chain(
    optax.scale_by_adam(b1=0.9, b2=0.999, eps=1e-8),
    optax.scale(-lr),
)
```

The alias `optax.adamw(lr, weight_decay=wd)` is equivalent to:
```python
optax.chain(
    optax.scale_by_adam(b1=0.9, b2=0.999, eps=1e-8),
    optax.add_decayed_weights(weight_decay=wd),
    optax.scale(-lr),
)
```

This decomposition is important: when you use `optax.chain()` with the
low-level `scale_by_adam`, you must include `optax.scale(-lr)` yourself
(negative sign because optax applies updates additively: `params + updates`).

### 2.4 Named Chains for Debugging

```python
optimizer = optax.named_chain(
    ("clip", optax.clip_by_global_norm(0.5)),
    ("adam", optax.scale_by_adam(eps=1e-5)),
    ("lr", optax.scale(-3e-4)),
)
# Access sub-states: opt_state["clip"], opt_state["adam"], etc.
```

---

## 3. Learning Rate Schedules

### 3.1 Warmup + Cosine Decay (Most Common for RL)

```python
schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0,         # start from zero
    peak_value=3e-4,        # ramp up to this
    warmup_steps=1000,      # linear warmup phase
    decay_steps=100_000,    # total steps INCLUDING warmup
    end_value=0.0,          # final LR
    exponent=1.0,           # cosine exponent (1.0 = standard cosine)
)

optimizer = optax.adam(learning_rate=schedule)
# or
optimizer = optax.chain(
    optax.clip_by_global_norm(0.5),
    optax.adamw(learning_rate=schedule),
)
```

### 3.2 Linear Decay (Common in PPO)

```python
# Simple linear decay from initial to zero
schedule = optax.linear_schedule(
    init_value=2.5e-4,
    end_value=0.0,
    transition_steps=total_timesteps // (num_envs * num_steps),
)

# Equivalent using polynomial_schedule with power=1
schedule = optax.polynomial_schedule(
    init_value=2.5e-4,
    end_value=0.0,
    power=1,                 # power=1 means linear
    transition_steps=total_opt_steps,
)

optimizer = optax.adam(learning_rate=schedule)
```

### 3.3 Warmup + Linear Decay (Composed)

```python
warmup_steps = 1000
total_steps = 100_000

warmup_fn = optax.linear_schedule(
    init_value=0.0,
    end_value=3e-4,
    transition_steps=warmup_steps,
)

decay_fn = optax.linear_schedule(
    init_value=3e-4,
    end_value=0.0,
    transition_steps=total_steps - warmup_steps,
)

schedule = optax.join_schedules(
    schedules=[warmup_fn, decay_fn],
    boundaries=[warmup_steps],
)

optimizer = optax.adam(learning_rate=schedule)
```

### 3.4 Warmup + Cosine (Composed Manually)

```python
warmup_fn = optax.linear_schedule(
    init_value=0.0,
    end_value=3e-4,
    transition_steps=warmup_steps,
)

cosine_fn = optax.cosine_decay_schedule(
    init_value=3e-4,
    decay_steps=total_steps - warmup_steps,
    alpha=0.0,  # minimum multiplier floor (end_value = init_value * alpha)
)

schedule = optax.join_schedules(
    schedules=[warmup_fn, cosine_fn],
    boundaries=[warmup_steps],
)
```

### 3.5 Exponential Decay

```python
schedule = optax.exponential_decay(
    init_value=1e-3,
    transition_steps=10_000,
    decay_rate=0.99,
    transition_begin=0,   # delay before decay starts
    staircase=False,      # True = step decay, False = smooth
    end_value=1e-5,       # optional floor
)
```

### 3.6 Piecewise Constant (Step Decay)

```python
schedule = optax.piecewise_constant_schedule(
    init_value=3e-4,
    boundaries_and_scales={
        50_000: 0.1,   # multiply by 0.1 at step 50k -> 3e-5
        100_000: 0.1,  # multiply by 0.1 at step 100k -> 3e-6
    },
)
```

### 3.7 Using Schedules with scale_by_schedule

When using `optax.chain()` with `scale_by_adam`, apply the schedule via
`scale_by_schedule` instead of passing it to the optimizer:

```python
schedule_fn = optax.warmup_cosine_decay_schedule(
    init_value=0.0, peak_value=1.0,
    warmup_steps=1000, decay_steps=100_000,
)

optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.scale_by_adam(eps=1e-5),
    optax.scale_by_schedule(schedule_fn),
    optax.scale(-3e-4),  # base learning rate
)
```

### 3.8 Warmup Constant (No Decay)

```python
schedule = optax.warmup_constant_schedule(
    init_value=0.0,
    peak_value=3e-4,
    warmup_steps=1000,
)
# Ramps from 0 to 3e-4 over 1000 steps, then stays constant.
```

---

## 4. RL-Specific Patterns

### 4.1 Separate Optimizers for Actor and Critic

**Approach A: Fully separate optimizer states** (recommended for SAC, TD3)

```python
import optax
import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState

# Define separate optimizers
actor_tx = optax.chain(
    optax.clip_by_global_norm(0.5),
    optax.adam(learning_rate=3e-4),
)

critic_tx = optax.chain(
    optax.clip_by_global_norm(0.5),
    optax.adam(learning_rate=3e-4),
)

# Create separate TrainStates
actor_state = TrainState.create(
    apply_fn=actor_network.apply,
    params=actor_params,
    tx=actor_tx,
)

critic_state = TrainState.create(
    apply_fn=critic_network.apply,
    params=critic_params,
    tx=critic_tx,
)

# Update them independently
@jax.jit
def update_critic(critic_state, batch):
    def critic_loss_fn(params):
        # ... compute TD loss ...
        return loss
    grads = jax.grad(critic_loss_fn)(critic_state.params)
    return critic_state.apply_gradients(grads=grads)

@jax.jit
def update_actor(actor_state, critic_state, batch):
    def actor_loss_fn(params):
        # ... compute policy gradient loss using critic_state.params ...
        return loss
    grads = jax.grad(actor_loss_fn)(actor_state.params)
    return actor_state.apply_gradients(grads=grads)
```

**Approach B: Single model with multi_transform** (useful for shared-backbone
actor-critic like PPO)

```python
import optax

def label_fn(params):
    """Label parameters as 'actor' or 'critic' based on pytree path."""
    flat_params = {}
    for path, _ in jax.tree_util.tree_leaves_with_path(params):
        key = '/'.join(str(k) for k in path)
        if 'actor' in key:
            flat_params[path] = 'actor'
        elif 'critic' in key:
            flat_params[path] = 'critic'
        else:
            flat_params[path] = 'shared'
    return jax.tree_util.tree_unflatten(
        jax.tree_util.tree_structure(params),
        [flat_params[p] for p, _ in jax.tree_util.tree_leaves_with_path(params)]
    )

# Different LRs for actor vs critic
optimizer = optax.multi_transform(
    transforms={
        'actor': optax.chain(
            optax.clip_by_global_norm(0.5),
            optax.adam(learning_rate=3e-4),
        ),
        'critic': optax.chain(
            optax.clip_by_global_norm(0.5),
            optax.adam(learning_rate=1e-3),
        ),
        'shared': optax.chain(
            optax.clip_by_global_norm(0.5),
            optax.adam(learning_rate=5e-4),
        ),
    },
    param_labels=label_fn,
)
```

**Approach C: Using optax.partition** (newer API, cleaner)

```python
optimizer = optax.partition(
    transforms={
        'actor': optax.adam(3e-4),
        'critic': optax.adam(1e-3),
    },
    param_labels=label_fn,
)
```

### 4.2 Gradient Accumulation

**Basic gradient accumulation with MultiSteps:**

```python
base_optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(learning_rate=3e-4),
)

# Accumulate gradients over 4 mini-steps before applying update
optimizer = optax.MultiSteps(
    opt=base_optimizer,
    every_k_schedule=4,      # accumulate 4 steps
    use_grad_mean=True,      # average (not sum) the accumulated gradients
)

# Usage is identical to a normal optimizer:
opt_state = optimizer.init(params)

@jax.jit
def train_step(params, opt_state, batch):
    grads = jax.grad(loss_fn)(params, batch)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state

# During intermediate steps, `updates` will be zeros.
# On every 4th step, actual updates are applied.
```

**Dynamic accumulation schedule:**

```python
# Increase accumulation steps over training
def accum_schedule(step):
    return jnp.where(step < 10_000, 2, 4)

optimizer = optax.MultiSteps(
    opt=optax.adam(3e-4),
    every_k_schedule=accum_schedule,
)
```

**Important note**: When combining `MultiSteps` with learning rate schedules,
the schedule step counter only advances on actual optimizer updates (not on
accumulation steps). This ensures mathematical equivalence with larger batches.

### 4.3 Multi-Loss Optimization (Policy + Value + Entropy)

**Pattern A: Single combined loss (standard PPO approach)**

```python
import jax
import jax.numpy as jnp
import optax

def ppo_loss_fn(params, batch, clip_eps=0.2, vf_coeff=0.5, ent_coeff=0.01):
    """Combined PPO loss: policy + value + entropy."""
    logits, values = model.apply(params, batch['obs'])

    # Policy loss (clipped surrogate)
    log_probs = jax.nn.log_softmax(logits)
    action_log_probs = jnp.take_along_axis(
        log_probs, batch['actions'][:, None], axis=-1
    ).squeeze(-1)
    ratio = jnp.exp(action_log_probs - batch['old_log_probs'])
    advantages = batch['advantages']

    surr1 = ratio * advantages
    surr2 = jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
    policy_loss = -jnp.mean(jnp.minimum(surr1, surr2))

    # Value loss
    value_loss = jnp.mean(jnp.square(values.squeeze() - batch['returns']))

    # Entropy bonus
    probs = jax.nn.softmax(logits)
    entropy = -jnp.sum(probs * log_probs, axis=-1).mean()

    total_loss = policy_loss + vf_coeff * value_loss - ent_coeff * entropy

    return total_loss, {
        'policy_loss': policy_loss,
        'value_loss': value_loss,
        'entropy': entropy,
    }

# Single optimizer for the combined loss
optimizer = optax.chain(
    optax.clip_by_global_norm(0.5),
    optax.adam(learning_rate=2.5e-4, eps=1e-5),
)

@jax.jit
def train_step(params, opt_state, batch):
    (loss, aux), grads = jax.value_and_grad(ppo_loss_fn, has_aux=True)(
        params, batch
    )
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, aux
```

**Pattern B: Separate gradient computation, combined update**

```python
def train_step_separate_grads(params, opt_state, batch):
    """Compute gradients for each loss separately, then combine."""

    def policy_loss_fn(params):
        logits, _ = model.apply(params, batch['obs'])
        # ... policy loss computation ...
        return policy_loss

    def value_loss_fn(params):
        _, values = model.apply(params, batch['obs'])
        return jnp.mean(jnp.square(values.squeeze() - batch['returns']))

    def entropy_fn(params):
        logits, _ = model.apply(params, batch['obs'])
        probs = jax.nn.softmax(logits)
        return -jnp.sum(probs * jnp.log(probs + 1e-8), axis=-1).mean()

    policy_grads = jax.grad(policy_loss_fn)(params)
    value_grads = jax.grad(value_loss_fn)(params)
    entropy_grads = jax.grad(entropy_fn)(params)

    # Weighted combination
    combined_grads = jax.tree.map(
        lambda pg, vg, eg: pg + 0.5 * vg - 0.01 * eg,
        policy_grads, value_grads, entropy_grads,
    )

    updates, opt_state = optimizer.update(combined_grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state
```

### 4.4 inject_hyperparams for Dynamic Hyperparameters

```python
# Useful for RL where you may want to adjust LR mid-training
@optax.inject_hyperparams
def create_optimizer(learning_rate, max_grad_norm=0.5):
    return optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.adam(learning_rate),
    )

optimizer = create_optimizer(learning_rate=3e-4)
opt_state = optimizer.init(params)

# Later, read current learning rate from state:
current_lr = opt_state.hyperparams['learning_rate']

# Or set it dynamically:
opt_state.hyperparams['learning_rate'] = 1e-4
```

---

## 5. Advanced Features

### 5.1 optax.MultiSteps (Gradient Accumulation)

```python
class optax.MultiSteps(
    opt: GradientTransformation,
    every_k_schedule: int | Callable[[Array], Array],
    use_grad_mean: bool = True,
    should_skip_update_fn: Optional[Callable] = None,
)
```

**State contains:**
- `mini_step`: current step within accumulation window
- `gradient_step`: number of actual optimizer updates performed
- `inner_opt_state`: wrapped optimizer state
- `acc_grads`: accumulated gradient buffer

**Full example with schedule awareness:**

```python
schedule = optax.linear_schedule(3e-4, 0.0, transition_steps=25_000)
# 25_000 refers to OUTER steps (actual updates), not mini-steps

base_opt = optax.chain(
    optax.clip_by_global_norm(0.5),
    optax.adam(learning_rate=schedule),
)

optimizer = optax.MultiSteps(base_opt, every_k_schedule=4)

# The schedule advances once every 4 mini-steps
# Total mini-steps needed: 25_000 * 4 = 100_000
```

### 5.2 optax.lookahead

```python
optax.lookahead(
    fast_optimizer: GradientTransformation,
    sync_period: int,        # steps between slow-fast synchronization
    slow_step_size: float,   # interpolation coefficient for slow params
    reset_state: bool = False,  # reset fast optimizer state after sync
)
```

**Usage:**

```python
fast_opt = optax.adam(learning_rate=1e-3)
optimizer = optax.lookahead(
    fast_optimizer=fast_opt,
    sync_period=6,          # sync every 6 steps
    slow_step_size=0.5,     # slow params = slow + 0.5 * (fast - slow)
)

opt_state = optimizer.init(params)

# The state contains LookaheadParams with .fast and .slow
# Use .fast for gradient computation, .slow for evaluation
for step in range(num_steps):
    grads = jax.grad(loss_fn)(opt_state.params.fast)  # conceptual
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
```

**RL relevance**: Lookahead can stabilize training in RL where the
loss landscape is highly non-stationary. The slow parameters act as
a smoothed checkpoint that prevents catastrophic policy degradation.

### 5.3 Masked Optimizers

```python
optax.masked(
    inner: GradientTransformation,
    mask: PyTree | Callable[[Params], PyTree],
)
```

**Use case: Freeze some parameters during RL fine-tuning**

```python
# Freeze the encoder, only train the policy head
def freeze_encoder_mask(params):
    return jax.tree.map(
        lambda path, _: not any('encoder' in str(p) for p in path),
        jax.tree_util.tree_leaves_with_path(params),
        jax.tree_util.tree_leaves(params),
    )
    # Simpler approach if you know the structure:
    return {
        'encoder': jax.tree.map(lambda _: False, params['encoder']),
        'actor_head': jax.tree.map(lambda _: True, params['actor_head']),
        'critic_head': jax.tree.map(lambda _: True, params['critic_head']),
    }

optimizer = optax.masked(
    optax.adam(3e-4),
    mask=freeze_encoder_mask,
)
```

**Use case: Skip weight decay for bias and normalization params**

```python
# Only apply weight decay to parameters with ndim >= 2 (kernels)
decay_mask_fn = lambda params: jax.tree.map(lambda x: x.ndim >= 2, params)

optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.scale_by_adam(),
    optax.add_decayed_weights(weight_decay=0.01, mask=decay_mask_fn),
    optax.scale(-3e-4),
)
```

### 5.4 optax.multi_transform / optax.partition

Apply entirely different optimizer configurations to different parameter groups.

```python
# multi_transform (established API)
optimizer = optax.multi_transform(
    transforms={
        'train': optax.adam(learning_rate=3e-4),
        'freeze': optax.set_to_zero(),  # zero updates = frozen
    },
    param_labels=label_fn,  # returns 'train' or 'freeze' per param leaf
)

# partition (newer, cleaner API - same functionality)
optimizer = optax.partition(
    transforms={
        'actor': optax.chain(
            optax.clip_by_global_norm(0.5),
            optax.adam(3e-4),
        ),
        'critic': optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adam(1e-3),
        ),
    },
    param_labels=label_fn,
)
```

### 5.5 Contrib: Schedule-Free Optimizers

```python
from optax import contrib

# Schedule-free Adam (no LR schedule needed)
optimizer = contrib.schedule_free_adamw(
    learning_rate=3e-4,
    warmup_steps=1000,
    weight_decay=0.01,
)

# Schedule-free SGD
optimizer = contrib.schedule_free_sgd(
    learning_rate=1e-2,
    warmup_steps=500,
)
```

### 5.6 Optimistic Gradient Descent (for GANs / Adversarial RL)

```python
# Useful for min-max problems in adversarial imitation learning
optimizer = optax.optimistic_gradient_descent(
    learning_rate=1e-3,
    alpha=1.0,
    beta=1.0,
)
```

---

## 6. Best Practices by Algorithm

### 6.1 PPO (Proximal Policy Optimization)

```python
import optax

def create_ppo_optimizer(
    learning_rate: float = 2.5e-4,
    max_grad_norm: float = 0.5,
    total_timesteps: int = 10_000_000,
    num_envs: int = 8,
    num_steps: int = 128,     # rollout length
    num_epochs: int = 4,
    num_minibatches: int = 4,
    anneal_lr: bool = True,
):
    """Standard PPO optimizer with linear LR annealing."""
    num_updates = total_timesteps // (num_envs * num_steps)
    total_opt_steps = num_updates * num_epochs * num_minibatches

    if anneal_lr:
        lr_schedule = optax.linear_schedule(
            init_value=learning_rate,
            end_value=0.0,
            transition_steps=total_opt_steps,
        )
    else:
        lr_schedule = learning_rate

    return optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.adam(learning_rate=lr_schedule, eps=1e-5),
    )

# Loss coefficients (not part of optax, but standard PPO):
# vf_coeff = 0.5      (value function loss coefficient)
# ent_coeff = 0.01    (entropy bonus coefficient)
# clip_eps = 0.2      (PPO clipping parameter)
```

**Key PPO optimizer notes:**
- `eps=1e-5` (not the default 1e-8) is standard in PPO implementations
  (from CleanRL, Stable-Baselines3 defaults).
- Linear LR annealing is standard; cosine also works.
- Global norm clipping at 0.5 is the most common value.
- Single optimizer for combined actor-critic loss.

### 6.2 SAC (Soft Actor-Critic)

```python
def create_sac_optimizers(
    actor_lr: float = 3e-4,
    critic_lr: float = 3e-4,
    alpha_lr: float = 3e-4,
    max_grad_norm: float = 1.0,
):
    """Separate optimizers for SAC's three components."""

    actor_tx = optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.adam(learning_rate=actor_lr),
    )

    critic_tx = optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.adam(learning_rate=critic_lr),
    )

    # Temperature parameter (log_alpha) optimizer
    alpha_tx = optax.adam(learning_rate=alpha_lr)

    return actor_tx, critic_tx, alpha_tx

# SAC training pattern with Flax TrainState:
actor_state = TrainState.create(
    apply_fn=actor.apply, params=actor_params, tx=actor_tx,
)
critic_state = TrainState.create(
    apply_fn=critic.apply, params=critic_params, tx=critic_tx,
)

# Temperature is typically a single parameter:
log_alpha = jnp.array(0.0)
alpha_opt_state = alpha_tx.init(log_alpha)

@jax.jit
def update_alpha(log_alpha, alpha_opt_state, log_probs, target_entropy):
    def alpha_loss_fn(log_alpha):
        alpha = jnp.exp(log_alpha)
        return -alpha * jnp.mean(log_probs + target_entropy)
    grads = jax.grad(alpha_loss_fn)(log_alpha)
    updates, alpha_opt_state = alpha_tx.update(grads, alpha_opt_state)
    log_alpha = optax.apply_updates(log_alpha, updates)
    return log_alpha, alpha_opt_state
```

**Key SAC optimizer notes:**
- Three separate optimizers (actor, critic, temperature).
- All typically use Adam with LR 3e-4 (from the original paper).
- No LR scheduling is standard (constant LR works well for SAC).
- Gradient clipping is optional but recommended for stability.

### 6.3 DQN / Rainbow

```python
def create_dqn_optimizer(
    learning_rate: float = 6.25e-5,  # Rainbow default
    eps: float = 1.5e-4,             # Rainbow default
    max_grad_norm: float = 10.0,
):
    """DQN/Rainbow optimizer (RMSProp or Adam)."""
    # Original DQN used RMSProp:
    # return optax.rmsprop(learning_rate=2.5e-4, decay=0.95, eps=0.01)

    # Rainbow / modern DQN uses Adam:
    return optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.adam(learning_rate=learning_rate, eps=eps),
    )
```

### 6.4 TD3 (Twin Delayed DDPG)

```python
def create_td3_optimizers(
    actor_lr: float = 3e-4,
    critic_lr: float = 3e-4,
):
    """TD3 uses separate optimizers, actor updated less frequently."""
    actor_tx = optax.adam(learning_rate=actor_lr)
    critic_tx = optax.adam(learning_rate=critic_lr)
    return actor_tx, critic_tx

# TD3 delayed update pattern:
@jax.jit
def td3_update(actor_state, critic_state, batch, step):
    # Always update critic
    critic_grads = jax.grad(critic_loss_fn)(critic_state.params, batch)
    critic_state = critic_state.apply_gradients(grads=critic_grads)

    # Update actor only every 2 steps
    def update_actor(actor_state):
        actor_grads = jax.grad(actor_loss_fn)(
            actor_state.params, critic_state.params, batch
        )
        return actor_state.apply_gradients(grads=actor_grads)

    actor_state = jax.lax.cond(
        step % 2 == 0,
        update_actor,
        lambda s: s,
        actor_state,
    )

    return actor_state, critic_state
```

### 6.5 General RL Optimizer Recipe

```python
def create_rl_optimizer(
    learning_rate: float = 3e-4,
    max_grad_norm: float = 0.5,
    weight_decay: float = 0.0,
    warmup_steps: int = 0,
    total_steps: int | None = None,
    schedule_type: str = 'constant',  # 'constant', 'linear', 'cosine'
    eps: float = 1e-5,
    gradient_accumulation_steps: int = 1,
):
    """Flexible optimizer factory for RL experiments."""

    # Build schedule
    if schedule_type == 'constant' or total_steps is None:
        lr_schedule = learning_rate
    elif schedule_type == 'linear':
        if warmup_steps > 0:
            warmup = optax.linear_schedule(0.0, learning_rate, warmup_steps)
            decay = optax.linear_schedule(
                learning_rate, 0.0, total_steps - warmup_steps
            )
            lr_schedule = optax.join_schedules(
                [warmup, decay], [warmup_steps]
            )
        else:
            lr_schedule = optax.linear_schedule(
                learning_rate, 0.0, total_steps
            )
    elif schedule_type == 'cosine':
        lr_schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=learning_rate,
            warmup_steps=warmup_steps,
            decay_steps=total_steps,
            end_value=learning_rate * 0.01,
        )

    # Build optimizer chain
    components = []
    components.append(optax.clip_by_global_norm(max_grad_norm))

    if weight_decay > 0:
        optimizer = optax.adamw(
            learning_rate=lr_schedule,
            eps=eps,
            weight_decay=weight_decay,
        )
    else:
        optimizer = optax.adam(learning_rate=lr_schedule, eps=eps)

    components.append(optimizer)
    tx = optax.chain(*components)

    # Wrap with gradient accumulation if needed
    if gradient_accumulation_steps > 1:
        tx = optax.MultiSteps(tx, every_k_schedule=gradient_accumulation_steps)

    return tx
```

---

## Appendix: Common Pitfalls

1. **Forgetting the negative sign**: When using `scale_by_adam()` + `scale()`,
   you must negate the learning rate: `optax.scale(-lr)`. The high-level aliases
   (`adam`, `adamw`) handle this automatically.

2. **Clipping order**: Always clip BEFORE the optimizer in `optax.chain()`.
   Clipping after `scale_by_adam` but before `scale(-lr)` clips the
   Adam-normalized gradients, which is what you typically want.

3. **Schedule step counting with MultiSteps**: The schedule counter only
   advances on actual updates, not on accumulation steps. Plan your
   `transition_steps` accordingly.

4. **eps in Adam for RL**: Many RL implementations use `eps=1e-5` instead
   of the default `1e-8`. This is because RL gradients can be very noisy
   and small eps values can lead to excessively large updates.

5. **Weight decay on biases**: Use the `mask` parameter in `adamw` or
   `add_decayed_weights` to exclude bias and normalization parameters
   from weight decay.

6. **Passing params to update()**: Some optax transforms (like
   `add_decayed_weights`) need the current parameters:
   `optimizer.update(grads, opt_state, params)`. If using `adamw`, this
   is handled internally by the chain.

7. **Schedule value ranges**: `warmup_cosine_decay_schedule` `decay_steps`
   parameter INCLUDES the warmup steps. So actual cosine decay duration
   is `decay_steps - warmup_steps`.
