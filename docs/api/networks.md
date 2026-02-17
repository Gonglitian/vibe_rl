# Networks Reference

Neural network building blocks for **vibe_rl**. All networks are [Equinox](https://docs.kidger.site/equinox/) modules — immutable pytrees that compose naturally with `jax.jit`, `jax.vmap`, and `eqx.filter_value_and_grad`.

---

## General-Purpose Encoders

```python
from vibe_rl.networks import Encoder, MLPEncoder, CNNEncoder, ViTEncoder, make_encoder
```

### Encoder Protocol

`vibe_rl.networks.encoder.Encoder`

```python
@runtime_checkable
class Encoder(Protocol):
    output_dim: int

    def __call__(self, x: jax.Array) -> jax.Array:
        """Encode a single observation into a feature vector."""
        ...
```

Any `eqx.Module` that maps an observation array to a flat feature vector and exposes its `output_dim` satisfies this protocol. All encoders below implement it.

---

### MLPEncoder

`vibe_rl.networks.encoder.MLPEncoder`

Simple MLP encoder for flat (vector) observations. Mirrors the hidden-layer pattern used in the algorithm networks.

```python
class MLPEncoder(eqx.Module):
    layers: list
    output_dim: int  # static field
```

#### Constructor

```python
MLPEncoder(
    input_dim: int,
    hidden_sizes: tuple[int, ...] = (64, 64),
    *,
    key: jax.Array,
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `input_dim` | `int` | — | Dimensionality of the flat observation. |
| `hidden_sizes` | `tuple[int, ...]` | `(64, 64)` | Widths of hidden layers. |
| `key` | `jax.Array` | — | PRNG key for initialization. |

#### \_\_call\_\_

```python
def __call__(self, x: jax.Array) -> jax.Array
```

| Parameter | Type | Description |
|---|---|---|
| `x` | `Array` | Flat observation, shape `(input_dim,)`. |

**Returns:** Feature vector of shape `(hidden_sizes[-1],)`. Activation: `tanh`.

---

### CNNEncoder

`vibe_rl.networks.cnn.CNNEncoder`

Convolutional encoder for image observations. Default architecture follows the Nature DQN paper (Mnih et al. 2015).

```
Conv2d(32, 8x8, stride 4) -> ReLU
Conv2d(64, 4x4, stride 2) -> ReLU
Conv2d(64, 3x3, stride 1) -> ReLU
Flatten -> Linear(512) -> ReLU
```

```python
class CNNEncoder(eqx.Module):
    conv_layers: list
    fc: eqx.nn.Linear
    output_dim: int  # static field
```

#### Constructor

```python
CNNEncoder(
    height: int,
    width: int,
    channels: int = 3,
    channel_sizes: tuple[int, ...] = (32, 64, 64),
    kernel_sizes: tuple[int, ...] = (8, 4, 3),
    strides: tuple[int, ...] = (4, 2, 1),
    mlp_hidden: int = 512,
    *,
    key: jax.Array,
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `height` | `int` | — | Image height. |
| `width` | `int` | — | Image width. |
| `channels` | `int` | `3` | Number of input channels. |
| `channel_sizes` | `tuple[int, ...]` | `(32, 64, 64)` | Output channels for each conv layer. |
| `kernel_sizes` | `tuple[int, ...]` | `(8, 4, 3)` | Kernel size for each conv layer. |
| `strides` | `tuple[int, ...]` | `(4, 2, 1)` | Stride for each conv layer. |
| `mlp_hidden` | `int` | `512` | Size of the fully-connected layer after flatten. |
| `key` | `jax.Array` | — | PRNG key for initialization. |

#### \_\_call\_\_

```python
def __call__(self, x: jax.Array) -> jax.Array
```

| Parameter | Type | Description |
|---|---|---|
| `x` | `Array` | Image array of shape `(H, W, C)` in `float32` `[0, 1]`. |

**Returns:** Feature vector of shape `(mlp_hidden,)`. Activation: `relu`.

> **Image format:** Input images use **channels-last** convention `(H, W, C)`. The encoder transposes to channels-first internally for Equinox `Conv2d`. Values should be `float32` in `[0, 1]` — divide `uint8` images by 255 before passing.

---

### ViTEncoder

`vibe_rl.networks.vit.ViTEncoder`

Lightweight Vision Transformer encoder following Dosovitskiy et al. (2020) with SigLIP-inspired design choices:

1. Patch embedding via non-overlapping `Conv2d`
2. Learnable positional embedding (additive)
3. Stacked pre-norm Transformer blocks (LayerNorm -> MHA -> residual -> LayerNorm -> MLP -> residual)
4. Mean-pool over patch tokens -> Linear projection

No CLS token, no dropout — intentionally lightweight for `jax.jit` RL workloads.

```python
class ViTEncoder(eqx.Module):
    patch_embed: eqx.nn.Conv2d
    pos_embed: jax.Array
    blocks: list
    norm: eqx.nn.LayerNorm
    head: eqx.nn.Linear
    output_dim: int  # static field
```

#### Constructor

```python
ViTEncoder(
    height: int,
    width: int,
    channels: int = 3,
    patch_size: int = 8,
    embed_dim: int = 128,
    depth: int = 4,
    num_heads: int = 4,
    mlp_ratio: float = 4.0,
    output_dim: int = 256,
    *,
    key: jax.Array,
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `height` | `int` | — | Image height (must be divisible by `patch_size`). |
| `width` | `int` | — | Image width (must be divisible by `patch_size`). |
| `channels` | `int` | `3` | Number of input channels. |
| `patch_size` | `int` | `8` | Side length of each square patch. |
| `embed_dim` | `int` | `128` | Embedding dimension for patch tokens. |
| `depth` | `int` | `4` | Number of Transformer blocks. |
| `num_heads` | `int` | `4` | Number of attention heads. |
| `mlp_ratio` | `float` | `4.0` | MLP hidden dim = `embed_dim * mlp_ratio`. |
| `output_dim` | `int` | `256` | Size of the final projection (encoder output). |
| `key` | `jax.Array` | — | PRNG key for initialization. |

#### \_\_call\_\_

```python
def __call__(self, x: jax.Array) -> jax.Array
```

| Parameter | Type | Description |
|---|---|---|
| `x` | `Array` | Image array of shape `(H, W, C)` in `float32` `[0, 1]`. |

**Returns:** Feature vector of shape `(output_dim,)`.

---

### make_encoder

`vibe_rl.networks.encoder.make_encoder`

Factory to create an encoder by name.

```python
def make_encoder(
    kind: str,
    *,
    key: jax.Array,
    # MLP args
    input_dim: int | None = None,
    hidden_sizes: tuple[int, ...] = (64, 64),
    # CNN / ViT shared args
    height: int | None = None,
    width: int | None = None,
    channels: int = 3,
    # CNN-specific
    channel_sizes: tuple[int, ...] = (32, 64, 64),
    kernel_sizes: tuple[int, ...] = (8, 4, 3),
    strides: tuple[int, ...] = (4, 2, 1),
    mlp_hidden: int = 512,
    # ViT-specific
    patch_size: int = 8,
    embed_dim: int = 128,
    depth: int = 4,
    num_heads: int = 4,
    mlp_ratio: float = 4.0,
    output_dim: int = 256,
) -> CNNEncoder | ViTEncoder | MLPEncoder
```

| Parameter | Type | Description |
|---|---|---|
| `kind` | `str` | One of `"mlp"`, `"cnn"`, or `"vit"`. |
| `key` | `jax.Array` | PRNG key for initialization. |
| (remaining) | | Forwarded to the chosen encoder constructor. |

**Returns:** An encoder instance satisfying the `Encoder` protocol.

**Raises:** `ValueError` if `kind` is not recognized, or if required arguments (`input_dim` for MLP, `height`/`width` for CNN/ViT) are missing.

**Example:**

```python
import jax

key = jax.random.PRNGKey(0)

# MLP for flat observations
encoder = make_encoder("mlp", key=key, input_dim=4)

# CNN for image observations
encoder = make_encoder("cnn", key=key, height=84, width=84, channels=3)

# ViT for image observations
encoder = make_encoder("vit", key=key, height=64, width=64, patch_size=8)
```

---

## Algorithm Networks

Networks used internally by the built-in algorithms. These can also be used directly for custom architectures.

### PPO Networks

`vibe_rl.algorithms.ppo.network`

```python
from vibe_rl.algorithms.ppo.network import ActorCategorical, Critic, ActorCriticShared
```

#### ActorCategorical

MLP actor for discrete action spaces: `obs -> action logits`.

```python
class ActorCategorical(eqx.Module):
    layers: list
```

**Constructor:**

```python
ActorCategorical(
    obs_dim: int,
    n_actions: int,
    hidden_sizes: tuple[int, ...] = (64, 64),
    *,
    key: jax.Array,
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `obs_dim` | `int` | — | Flat observation dimensionality. |
| `n_actions` | `int` | — | Number of discrete actions. |
| `hidden_sizes` | `tuple[int, ...]` | `(64, 64)` | Hidden layer widths. |
| `key` | `jax.Array` | — | PRNG key. |

**\_\_call\_\_(x: Array) -> Array:** Returns action logits (un-normalized log-probabilities), shape `(n_actions,)`. Activation: `tanh` (hidden layers), linear (output).

#### Critic

MLP critic: `obs -> scalar state-value V(s)`.

```python
class Critic(eqx.Module):
    layers: list
```

**Constructor:**

```python
Critic(
    obs_dim: int,
    hidden_sizes: tuple[int, ...] = (64, 64),
    *,
    key: jax.Array,
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `obs_dim` | `int` | — | Flat observation dimensionality. |
| `hidden_sizes` | `tuple[int, ...]` | `(64, 64)` | Hidden layer widths. |
| `key` | `jax.Array` | — | PRNG key. |

**\_\_call\_\_(x: Array) -> Array:** Returns scalar value estimate, shape `()`. Activation: `tanh` (hidden layers), linear (output).

#### ActorCriticShared

Shared-backbone actor-critic: `obs -> (logits, value)`. A single MLP backbone feeds into two separate linear heads.

```python
class ActorCriticShared(eqx.Module):
    backbone: list
    actor_head: eqx.nn.Linear
    critic_head: eqx.nn.Linear
```

**Constructor:**

```python
ActorCriticShared(
    obs_dim: int,
    n_actions: int,
    hidden_sizes: tuple[int, ...] = (64, 64),
    *,
    key: jax.Array,
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `obs_dim` | `int` | — | Flat observation dimensionality. |
| `n_actions` | `int` | — | Number of discrete actions. |
| `hidden_sizes` | `tuple[int, ...]` | `(64, 64)` | Backbone hidden layer widths. |
| `key` | `jax.Array` | — | PRNG key. |

**\_\_call\_\_(x: Array) -> tuple[Array, Array]:** Returns `(action_logits, value)`. Activation: `tanh` (backbone), linear (heads).

#### ActorCriticParams

Container for separate actor and critic parameters (used when `PPOConfig.shared_backbone=False`).

```python
class ActorCriticParams(NamedTuple):
    actor: Params      # ActorCategorical model
    critic: Params     # Critic model
```

---

### DQN Networks

`vibe_rl.algorithms.dqn.network`

```python
from vibe_rl.algorithms.dqn.network import QNetwork
```

#### QNetwork

Simple MLP Q-network: `obs -> Q(s, a)` for each discrete action.

```python
class QNetwork(eqx.Module):
    layers: list
```

**Constructor:**

```python
QNetwork(
    obs_dim: int,
    n_actions: int,
    hidden_sizes: tuple[int, ...] = (128, 128),
    *,
    key: jax.Array,
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `obs_dim` | `int` | — | Flat observation dimensionality. |
| `n_actions` | `int` | — | Number of discrete actions. |
| `hidden_sizes` | `tuple[int, ...]` | `(128, 128)` | Hidden layer widths. |
| `key` | `jax.Array` | — | PRNG key. |

**\_\_call\_\_(x: Array) -> Array:** Returns Q-values for all actions, shape `(n_actions,)`. Activation: `relu` (hidden layers), linear (output).

---

### SAC Networks

`vibe_rl.algorithms.sac.network`

```python
from vibe_rl.algorithms.sac.network import GaussianActor, TwinQNetwork
```

#### GaussianActor

Gaussian policy: `obs -> (mean, log_std)`. Outputs parameterize a diagonal Gaussian. The caller is responsible for reparameterized sampling and tanh squashing (handled by `SAC.act`).

```python
class GaussianActor(eqx.Module):
    layers: list
    mean_head: eqx.nn.Linear
    log_std_head: eqx.nn.Linear
```

**Constructor:**

```python
GaussianActor(
    obs_dim: int,
    action_dim: int,
    hidden_sizes: tuple[int, ...] = (256, 256),
    *,
    key: jax.Array,
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `obs_dim` | `int` | — | Flat observation dimensionality. |
| `action_dim` | `int` | — | Action vector dimensionality. |
| `hidden_sizes` | `tuple[int, ...]` | `(256, 256)` | Hidden layer widths. |
| `key` | `jax.Array` | — | PRNG key. |

**\_\_call\_\_(obs: Array) -> tuple[Array, Array]:** Returns `(mean, log_std)` each of shape `(action_dim,)`. Activation: `relu` (hidden layers), linear (heads).

#### QNetwork (SAC)

Single Q-network: `(obs, action) -> scalar Q-value`. Note: this is a separate class from the DQN `QNetwork` — it takes both observation and action as inputs.

```python
class QNetwork(eqx.Module):  # vibe_rl.algorithms.sac.network.QNetwork
    layers: list
```

**Constructor:**

```python
QNetwork(
    obs_dim: int,
    action_dim: int,
    hidden_sizes: tuple[int, ...] = (256, 256),
    *,
    key: jax.Array,
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `obs_dim` | `int` | — | Flat observation dimensionality. |
| `action_dim` | `int` | — | Action vector dimensionality. |
| `hidden_sizes` | `tuple[int, ...]` | `(256, 256)` | Hidden layer widths. |
| `key` | `jax.Array` | — | PRNG key. |

**\_\_call\_\_(obs: Array, action: Array) -> Array:** Concatenates `obs` and `action`, returns scalar Q-value, shape `()`. Activation: `relu` (hidden layers), linear (output).

#### TwinQNetwork

Twin Q-networks for clipped double-Q learning. Contains two independent `QNetwork` instances.

```python
class TwinQNetwork(eqx.Module):
    q1: QNetwork
    q2: QNetwork
```

**Constructor:**

```python
TwinQNetwork(
    obs_dim: int,
    action_dim: int,
    hidden_sizes: tuple[int, ...] = (256, 256),
    *,
    key: jax.Array,
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `obs_dim` | `int` | — | Flat observation dimensionality. |
| `action_dim` | `int` | — | Action vector dimensionality. |
| `hidden_sizes` | `tuple[int, ...]` | `(256, 256)` | Hidden layer widths (shared by both Q-networks). |
| `key` | `jax.Array` | — | PRNG key (split internally for the two networks). |

**\_\_call\_\_(obs: Array, action: Array) -> tuple[Array, Array]:** Returns `(q1_value, q2_value)`, each a scalar Q-value.
