# SAC (Soft Actor-Critic)

## 算法概述

SAC 是一种 off-policy 最大熵强化学习算法，专为 **连续动作空间** 设计。它在最大化累积回报的同时最大化策略熵，鼓励探索并提高鲁棒性。

核心思想：

- **最大熵目标**: `J(π) = Σ E[r(s,a) + α H(π(·|s))]` — 同时最大化回报和策略熵
- **Twin Q-networks**: 两个独立 Q 网络取最小值 (clipped double-Q)，缓解 Q 值过估计
- **重参数化采样 (reparameterization trick)**: 低方差策略梯度估计
- **Tanh squashing**: 将无界高斯动作映射到有界动作空间
- **自动温度调节**: 通过对偶梯度下降自动调整 α

**论文**: Haarnoja et al., *Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL with a Stochastic Actor*, ICML 2018; *Soft Actor-Critic Algorithms and Applications*, 2019 (arXiv:1812.05905)

---

## 连续动作空间支持

SAC 原生支持连续动作空间。动作生成流程：

```
obs → GaussianActor → (mean, log_std)
                         ↓
          重参数化: z = mean + std * ε,  ε ~ N(0, I)
                         ↓
          Tanh squashing: a_tanh = tanh(z)  ∈ [-1, 1]
                         ↓
          缩放至动作范围: a = low + 0.5 * (a_tanh + 1) * (high - low)
```

**Log-prob 修正** (change of variables):

```python
log π(a|s) = log N(z; mean, std) - Σ log(1 - tanh(z)²)
```

tanh 压缩改变了概率密度，必须用雅可比修正项补偿。

---

## SACConfig 字段说明

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `hidden_sizes` | `tuple[int, ...]` | `(256, 256)` | Actor/Critic MLP 隐藏层尺寸 |
| `actor_lr` | `float` | `3e-4` | Actor 学习率 |
| `critic_lr` | `float` | `3e-4` | Critic 学习率 |
| `alpha_lr` | `float` | `3e-4` | 温度参数 α 学习率 |
| `gamma` | `float` | `0.99` | 折扣因子 |
| `batch_size` | `int` | `256` | 每次更新的 mini-batch 大小 |
| `max_grad_norm` | `float` | `10.0` | 全局梯度范数裁剪阈值 |
| `tau` | `float` | `0.005` | 目标网络 Polyak 平均系数 |
| `init_alpha` | `float` | `1.0` | 温度参数 α 初始值 |
| `autotune_alpha` | `bool` | `True` | 是否自动调节温度 |
| `target_entropy_scale` | `float` | `1.0` | 目标熵 = -scale * action_dim |
| `action_low` | `float` | `-1.0` | 动作下界 (tanh 缩放后) |
| `action_high` | `float` | `1.0` | 动作上界 (tanh 缩放后) |
| `log_std_min` | `float` | `-20.0` | log_std 下限 (数值稳定性) |
| `log_std_max` | `float` | `2.0` | log_std 上限 |

三个独立优化器：

```python
# Actor
optax.chain(optax.clip_by_global_norm(max_grad_norm), optax.adam(actor_lr))
# Critic
optax.chain(optax.clip_by_global_norm(max_grad_norm), optax.adam(critic_lr))
# Alpha
optax.chain(optax.clip_by_global_norm(max_grad_norm), optax.adam(alpha_lr))
```

---

## 网络架构

### GaussianActor

高斯策略网络: `obs → (mean, log_std)`

```
输入 obs (obs_dim,)
  → Linear(obs_dim, 256) → ReLU
  → Linear(256, 256)     → ReLU
  ├→ mean_head:    Linear(256, action_dim) → mean
  └→ log_std_head: Linear(256, action_dim) → log_std
```

- 激活函数: **ReLU**
- `log_std` 被 clamp 在 `[log_std_min, log_std_max]` = `[-20, 2]`
- 输出的 `mean` 和 `log_std` 用于重参数化采样

### QNetwork (单个)

Q 网络将 (obs, action) 映射到标量 Q 值:

```
输入 concat([obs, action]) (obs_dim + action_dim,)
  → Linear(obs_dim + action_dim, 256) → ReLU
  → Linear(256, 256)                  → ReLU
  → Linear(256, 1)                    → squeeze
输出 q_value ()
```

### TwinQNetwork

包装两个独立的 `QNetwork`，用于 clipped double-Q learning:

```python
class TwinQNetwork(eqx.Module):
    q1: QNetwork
    q2: QNetwork

    def __call__(self, obs, action):
        return self.q1(obs, action), self.q2(obs, action)
```

训练时取 `min(Q1, Q2)` 作为 target，缓解 Q 值过估计。

---

## 自动温度调节 (Entropy Tuning)

SAC 的温度参数 α 控制探索与利用的平衡。设置 `autotune_alpha=True`（默认）启用自动调节。

**目标熵**: `H_target = -target_entropy_scale * action_dim`

**α 损失** (Eq. 18, arXiv:1812.05905):

```python
alpha_loss = -alpha * mean(stop_gradient(log_prob + target_entropy))
```

**直觉**: 当策略熵低于目标时，α 增大，鼓励更多探索；当策略熵高于目标时，α 减小。

**实现细节**: 优化对象为 `log_alpha`（而非 α 本身），确保 `alpha = exp(log_alpha) > 0`。

---

## 更新流程

每次 `SAC.update()` 按顺序执行三个梯度步：

### 1. Critic 更新 (Twin Q)

```python
# 从目标网络计算 soft Bellman target
next_action, next_log_prob = sample(actor, next_obs)
next_q1, next_q2 = Q_target(next_obs, next_action)
next_q_min = min(next_q1, next_q2)
target = reward + gamma * (next_q_min - alpha * next_log_prob) * (1 - done)

# 最小化 TD 误差
q1, q2 = Q_online(obs, action)
critic_loss = 0.5 * (mean((q1 - target)²) + mean((q2 - target)²))
```

### 2. Actor 更新

```python
# 最大化 Q 值，同时最大化熵
action_new, log_prob = sample(actor, obs)
q1, q2 = Q_online(obs, action_new)  # 使用更新后的 critic
q_min = min(q1, q2)
actor_loss = mean(alpha * log_prob - q_min)
```

### 3. Alpha 更新 (可选)

```python
# 自动调节温度
alpha_loss = -exp(log_alpha) * mean(stop_gradient(log_prob) + target_entropy)
```

### 4. 软目标更新 (Polyak 平均)

```python
target_params = (1 - tau) * target_params + tau * online_params
```

每步都做软更新 (τ = 0.005)，而非 DQN 的周期性硬更新。

---

## 状态容器

```python
class SACState(NamedTuple):
    actor_params: Params            # GaussianActor
    critic_params: Params           # TwinQNetwork (在线)
    target_critic_params: Params    # TwinQNetwork (目标)
    actor_opt_state: OptState       # Actor 优化器状态
    critic_opt_state: OptState      # Critic 优化器状态
    log_alpha: chex.Array           # log(α) 标量
    alpha_opt_state: OptState       # Alpha 优化器状态
    step: chex.Array                # 训练步计数器
    rng: chex.PRNGKey               # PRNG key
```

---

## 代码示例

### 基本训练 (Pendulum)

```python
import numpy as np
import jax
from vibe_rl.algorithms.sac import SAC, SACConfig
from vibe_rl.dataprotocol.replay_buffer import ReplayBuffer
from vibe_rl.env import make
from vibe_rl.env.wrappers import AutoResetWrapper

config = SACConfig(
    hidden_sizes=(256, 256),
    actor_lr=3e-4,
    critic_lr=3e-4,
    alpha_lr=3e-4,
    gamma=0.99,
    batch_size=256,
    tau=0.005,
    init_alpha=1.0,
    autotune_alpha=True,
    action_low=-2.0,
    action_high=2.0,
)

env, env_params = make("Pendulum-v1")
env = AutoResetWrapper(env)
env_params = env.default_params()

rng = jax.random.PRNGKey(42)
rng, env_key, agent_key = jax.random.split(rng, 3)

obs, env_state = env.reset(env_key, env_params)
state = SAC.init(agent_key, obs_shape=(3,), action_dim=1, config=config)

# 连续动作需要 float32 action_dtype
buffer = ReplayBuffer(
    capacity=50_000, obs_shape=(3,), action_shape=(1,), action_dtype=np.float32,
)

for step in range(50_000):
    # 随机探索采样
    action, state = SAC.act(state, obs, config=config, explore=True)

    rng, step_key = jax.random.split(rng)
    next_obs, env_state, reward, done, info = env.step(
        step_key, env_state, action, env_params,
    )

    buffer.push(
        obs=np.asarray(obs),
        action=np.asarray(action),
        reward=float(reward),
        next_obs=np.asarray(next_obs),
        done=float(done),
    )
    obs = next_obs

    if len(buffer) >= 1_000:
        batch = buffer.sample(config.batch_size)
        state, metrics = SAC.update(state, batch, config=config)

    if (step + 1) % 5_000 == 0:
        alpha = float(jax.numpy.exp(state.log_alpha))
        print(f"Step {step+1} | alpha={alpha:.3f}")
```

### 自定义超参

```python
config = SACConfig(
    hidden_sizes=(512, 512, 256),  # 更深的网络
    actor_lr=1e-4,                 # 较低的 actor 学习率
    critic_lr=3e-4,                # critic 可以快一些
    alpha_lr=1e-4,                 # 温度调节学习率
    tau=0.01,                      # 更快的目标网络更新
    init_alpha=0.2,                # 较低的初始温度
    autotune_alpha=False,          # 固定温度 (不自动调节)
    action_low=-1.0,               # 动作范围 [-1, 1]
    action_high=1.0,
    target_entropy_scale=0.5,      # 目标熵更低 (更少探索)
)
```

### 确定性评估

```python
# explore=False 使用策略均值 (tanh(mean))，无随机采样
action, state = SAC.act(state, obs, config=config, explore=False)
```

---

## API 参考

| 方法 | 签名 | 说明 |
|------|------|------|
| `SAC.init` | `(rng, obs_shape, action_dim, config) → SACState` | 初始化 agent |
| `SAC.act` | `(state, obs, config=, explore=True) → (action, state)` | 动作选择 |
| `SAC.update` | `(state, batch, config=) → (state, metrics)` | 单步梯度更新 (critic + actor + alpha) |

### SACMetrics

| 字段 | 说明 |
|------|------|
| `actor_loss` | Actor 损失 (α * log_prob - Q_min) |
| `critic_loss` | Twin Q 损失 (soft Bellman 残差) |
| `alpha_loss` | 温度 α 损失 |
| `alpha` | 当前温度值 exp(log_alpha) |
| `entropy` | 策略熵均值 (-log_prob) |
| `q_mean` | batch 中 Q 值的均值 |

---

## 源码路径

- Agent: `src/vibe_rl/algorithms/sac/agent.py`
- Config: `src/vibe_rl/algorithms/sac/config.py`
- Network: `src/vibe_rl/algorithms/sac/network.py`
- Types: `src/vibe_rl/algorithms/sac/types.py`
- Runner: `src/vibe_rl/runner/train_sac.py`
- Replay Buffer: `src/vibe_rl/dataprotocol/replay_buffer.py`
- Example: `examples/train_sac_pendulum.py`
