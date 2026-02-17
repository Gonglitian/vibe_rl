# PPO (Proximal Policy Optimization)

## 算法概述

PPO 是一种 on-policy 策略梯度算法，通过 **裁剪的替代目标函数 (clipped surrogate objective)** 限制每次更新的策略变化幅度，兼顾训练稳定性与样本效率。

核心思想：

- 每轮收集一段固定长度的轨迹 (rollout)，用 GAE 计算优势估计
- 对同一批数据做多轮 mini-batch SGD（重用数据提高样本效率）
- 用 clip 机制约束策略比率 `r(θ) = π_new / π_old`，防止过大的策略更新

**论文**: Schulman et al., *Proximal Policy Optimization Algorithms*, 2017

---

## PureJaxRL 设计

vibe-rl 的 PPO 实现采用 **PureJaxRL** 架构 — 整个训练循环（rollout 采集 + GAE 计算 + 多轮 mini-batch 更新）全部通过 `jax.lax.scan` 编译为单个 XLA 计算图。

```
┌────────────────────────────────────────────────────┐
│              jax.lax.scan (外层: updates)           │
│  ┌──────────────────────────────────────────────┐  │
│  │  collect_rollout: lax.scan × n_steps         │  │
│  │    act → env.step → PPOTransition            │  │
│  └──────────────────────────────────────────────┘  │
│                      ↓                              │
│            compute_gae (reverse scan)               │
│                      ↓                              │
│  ┌──────────────────────────────────────────────┐  │
│  │  update: lax.scan × n_epochs                 │  │
│  │    ┌──────────────────────────────────────┐  │  │
│  │    │  lax.scan × n_minibatches            │  │  │
│  │    │    loss_fn → grad → optimizer step   │  │  │
│  │    └──────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────┘
```

**关键设计点**:

- **纯函数**: `PPO` 类不可实例化，所有方法为 `@staticmethod`，状态通过 `PPOState` 显式传递
- **lax.scan 驱动**: rollout 采集、epoch 循环、minibatch 循环均使用 `lax.scan`，避免 Python 循环开销
- **vmap 并行环境**: 通过 `jax.vmap` 对多个环境进行向量化并行采集
- **Frozen config**: `PPOConfig` 为 `frozen=True` 的 dataclass，可安全作为 JIT 的 static argument

---

## PPOConfig 字段说明

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `hidden_sizes` | `tuple[int, ...]` | `(64, 64)` | Actor/Critic MLP 隐藏层尺寸 |
| `lr` | `float` | `2.5e-4` | Adam 学习率 |
| `max_grad_norm` | `float` | `0.5` | 全局梯度范数裁剪阈值 |
| `gamma` | `float` | `0.99` | 折扣因子 |
| `gae_lambda` | `float` | `0.95` | GAE lambda 参数 |
| `clip_eps` | `float` | `0.2` | PPO 裁剪范围 ε |
| `vf_coef` | `float` | `0.5` | 价值函数损失系数 |
| `ent_coef` | `float` | `0.01` | 熵正则化系数 |
| `n_steps` | `int` | `128` | 每轮 rollout 步数 |
| `n_minibatches` | `int` | `4` | Mini-batch 数量 |
| `n_epochs` | `int` | `4` | 每轮数据的 SGD epoch 数 |
| `num_envs` | `int` | `1` | 并行环境数量 |
| `shared_backbone` | `bool` | `False` | 是否使用共享骨干网络 |

优化器由 `make_optimizer()` 构建：

```python
optax.chain(
    optax.clip_by_global_norm(max_grad_norm),
    optax.adam(lr, eps=1e-5),
)
```

---

## 网络架构

### ActorCategorical

离散动作空间的策略网络: `obs → action logits`

```
输入 obs (obs_dim,)
  → Linear(obs_dim, 64) → tanh
  → Linear(64, 64)      → tanh
  → Linear(64, n_actions)
输出 logits (n_actions,)
```

激活函数使用 `tanh`。输出为未归一化的 log 概率 (logits)，通过 `jax.random.categorical` 采样动作。

### Critic

状态价值网络: `obs → V(s)`

```
输入 obs (obs_dim,)
  → Linear(obs_dim, 64) → tanh
  → Linear(64, 64)      → tanh
  → Linear(64, 1)       → squeeze
输出 value ()
```

结构与 Actor 相同，只是输出维度为 1（标量值函数）。

### ActorCriticShared

共享骨干变体: `obs → (logits, value)`

```
输入 obs (obs_dim,)
  → Linear(obs_dim, 64) → tanh    ┐
  → Linear(64, 64)      → tanh    ┤ 共享骨干
                                   ├→ actor_head:  Linear(64, n_actions) → logits
                                   └→ critic_head: Linear(64, 1)         → value
```

设置 `PPOConfig(shared_backbone=True)` 启用。可减少参数量，但 actor 和 critic 的特征耦合在一起。

---

## GAE 计算 (compute_gae)

GAE (Generalized Advantage Estimation) 通过反向时间扫描计算优势估计：

```python
def compute_gae(rewards, values, dones, last_value, gamma, gae_lambda):
    def _scan_fn(carry, transition):
        gae, next_value = carry
        reward, value, done = transition
        delta = reward + gamma * next_value * (1.0 - done) - value
        gae = delta + gamma * gae_lambda * (1.0 - done) * gae
        return (gae, value), gae

    init_carry = (jnp.zeros_like(last_value), last_value)
    _, advantages = jax.lax.scan(
        _scan_fn, init_carry, (rewards, values, dones), reverse=True,
    )
    returns = advantages + values
    return advantages, returns
```

**关键细节**:

- **反向扫描** (`reverse=True`): 从最后一步向前递推，`last_value` 为 bootstrap 值
- **done 掩码** (`1.0 - done`): episode 终止时截断 GAE 递推
- **返回值**: `advantages` 用于策略梯度，`returns = advantages + values` 用于价值函数目标

### PPO 损失函数

```python
# 裁剪替代目标
ratio = exp(log_prob_new - log_prob_old)
surr1 = ratio * advantages
surr2 = clip(ratio, 1 - ε, 1 + ε) * advantages
actor_loss = -min(surr1, surr2).mean()

# 裁剪价值损失
value_clipped = old_value + clip(value - old_value, -ε, ε)
critic_loss = 0.5 * max((value - returns)², (value_clipped - returns)²).mean()

# 总损失
total_loss = actor_loss + vf_coef * critic_loss - ent_coef * entropy
```

---

## 状态容器

```python
class PPOState(NamedTuple):
    params: Params          # ActorCriticParams 或 ActorCriticShared
    opt_state: OptState     # Optax 优化器状态
    step: chex.Array        # 训练步计数器 (标量)
    rng: chex.PRNGKey       # PRNG key

class ActorCriticParams(NamedTuple):
    actor: Params           # ActorCategorical
    critic: Params          # Critic
```

---

## 代码示例

### 基本训练 (CartPole)

```python
import jax
from vibe_rl.algorithms.ppo import PPO, PPOConfig
from vibe_rl.env import make
from vibe_rl.env.wrappers import AutoResetWrapper

config = PPOConfig()
env, env_params = make("CartPole-v1")
env = AutoResetWrapper(env)
env_params = env.default_params()

rng = jax.random.PRNGKey(42)
rng, env_key, agent_key = jax.random.split(rng, 3)

obs, env_state = env.reset(env_key, env_params)
state = PPO.init(agent_key, obs_shape=(4,), n_actions=2, config=config)

for update in range(200):
    # 采集 rollout (内部使用 lax.scan)
    state, trajectories, obs, env_state, last_value = PPO.collect_rollout(
        state, obs, env_state, env.step, env_params, config=config,
    )

    # PPO 更新 (多轮 mini-batch SGD)
    state, metrics = PPO.update(
        state, trajectories, last_value, config=config,
    )

    if (update + 1) % 10 == 0:
        print(f"Update {update+1} | loss={float(metrics.total_loss):.4f}")
```

### 自定义超参

```python
config = PPOConfig(
    hidden_sizes=(128, 128),   # 更大的网络
    lr=3e-4,                   # 调高学习率
    n_steps=256,               # 更长的 rollout
    n_minibatches=8,           # 更多 minibatch
    n_epochs=10,               # 更多 epoch
    clip_eps=0.1,              # 更紧的 clip 范围
    ent_coef=0.005,            # 减少探索
    gamma=0.999,               # 长视野折扣
    gae_lambda=0.98,           # 更高的 GAE lambda
    shared_backbone=True,      # 使用共享骨干网络
)
```

### 多环境并行训练

```python
config = PPOConfig(num_envs=8, n_steps=128, n_minibatches=4)

# 初始化 N 个并行环境
rng, *env_keys = jax.random.split(rng, config.num_envs + 1)
obs_batch, env_states = jax.vmap(env.reset, in_axes=(0, None))(
    jnp.stack(env_keys), env_params,
)

state = PPO.init(agent_key, obs_shape=(4,), n_actions=2, config=config)

# 使用 collect_rollout_batch 进行向量化采集
state, trajectories, obs_batch, env_states, last_values = PPO.collect_rollout_batch(
    state, obs_batch, env_states, env.step, env_params, config=config,
)
state, metrics = PPO.update(state, trajectories, last_values, config=config)
```

---

## API 参考

| 方法 | 签名 | 说明 |
|------|------|------|
| `PPO.init` | `(rng, obs_shape, n_actions, config) → PPOState` | 初始化 agent |
| `PPO.act` | `(state, obs, config=) → (action, log_prob, value, state)` | 单个观测的动作采样 |
| `PPO.act_batch` | `(state, obs, config=) → (actions, log_probs, values, state)` | 批量观测的动作采样 |
| `PPO.evaluate_actions` | `(params, obs, actions, config=) → (log_probs, values, entropy)` | 重新评估动作 (用于 update) |
| `PPO.get_value` | `(state, obs, config=) → value` | 单个观测的价值估计 |
| `PPO.get_value_batch` | `(state, obs, config=) → values` | 批量观测的价值估计 |
| `PPO.collect_rollout` | `(state, obs, env_state, env_step_fn, env_params, config=) → (state, traj, obs, env_state, last_value)` | 单环境 rollout 采集 |
| `PPO.collect_rollout_batch` | `(state, obs, env_states, env_step_fn, env_params, config=) → (state, traj, obs, env_states, last_values)` | 多环境并行 rollout 采集 |
| `PPO.update` | `(state, trajectories, last_value, config=) → (state, metrics)` | PPO 参数更新 |

### PPOMetrics

| 字段 | 说明 |
|------|------|
| `total_loss` | 总损失 (actor + vf_coef * critic - ent_coef * entropy) |
| `actor_loss` | 裁剪替代目标损失 |
| `critic_loss` | 裁剪价值函数损失 |
| `entropy` | 策略熵均值 |
| `approx_kl` | 近似 KL 散度 (old_log_prob - new_log_prob 的均值) |

---

## 源码路径

- Agent: `src/vibe_rl/algorithms/ppo/agent.py`
- Config: `src/vibe_rl/algorithms/ppo/config.py`
- Network: `src/vibe_rl/algorithms/ppo/network.py`
- Types: `src/vibe_rl/algorithms/ppo/types.py`
- Runner: `src/vibe_rl/runner/train_ppo.py`
- Example: `examples/train_ppo_cartpole.py`
