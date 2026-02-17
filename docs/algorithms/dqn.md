# DQN (Deep Q-Network)

## 算法概述

DQN 是一种 off-policy 值函数方法，通过神经网络逼近最优动作价值函数 Q*(s, a)，结合 **经验回放 (experience replay)** 和 **目标网络 (target network)** 实现稳定训练。

核心思想：

- 用 Q 网络逼近 Q(s, a)，选择 argmax Q(s, a) 作为策略
- 用目标网络提供稳定的 TD target: `r + γ * max_a' Q_target(s', a')`
- 从经验回放缓冲区中均匀采样 mini-batch 进行梯度更新
- 用 ε-greedy 策略在探索与利用之间平衡

**论文**: Mnih et al., *Human-level control through deep reinforcement learning*, Nature 2015

---

## 混合循环设计

与 PPO 的 PureJaxRL 架构不同，DQN 采用 **混合循环 (hybrid loop)** 设计：

```
┌─────────────────────────────────────────────────────────┐
│  Python outer loop (不可 JIT)                           │
│                                                         │
│  for step in range(total_steps):                        │
│      ┌───────────────────────────────────────────────┐  │
│      │  JIT: DQN.act(state, obs) → action            │  │
│      └───────────────────────────────────────────────┘  │
│      ┌───────────────────────────────────────────────┐  │
│      │  JIT: env.step(key, state, action) → next_obs │  │
│      └───────────────────────────────────────────────┘  │
│                                                         │
│      buffer.push(obs, action, reward, ...)  ← numpy    │
│      batch = buffer.sample(batch_size)      ← numpy→jax│
│                                                         │
│      ┌───────────────────────────────────────────────┐  │
│      │  JIT: DQN.update(state, batch) → state, loss  │  │
│      └───────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

**为什么不用 lax.scan?**

经验回放缓冲区 (`ReplayBuffer`) 使用 numpy 数组存储，支持 O(1) 插入和随机采样。这个可变数据结构无法在 JIT 编译图中使用，因此训练外层循环保留在 Python 中，而 `act()`、`update()` 等热路径函数通过 `@jax.jit` 编译。

---

## DQNConfig 字段说明

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `hidden_sizes` | `tuple[int, ...]` | `(128, 128)` | Q 网络 MLP 隐藏层尺寸 |
| `lr` | `float` | `1e-3` | Adam 学习率 |
| `gamma` | `float` | `0.99` | 折扣因子 |
| `batch_size` | `int` | `64` | 每次更新的 mini-batch 大小 |
| `max_grad_norm` | `float` | `10.0` | 全局梯度范数裁剪阈值 |
| `target_update_freq` | `int` | `1000` | 目标网络硬更新频率 (步数) |
| `epsilon_start` | `float` | `1.0` | ε-greedy 起始值 |
| `epsilon_end` | `float` | `0.01` | ε-greedy 终止值 |
| `epsilon_decay_steps` | `int` | `50000` | ε 线性衰减步数 |

优化器由 `make_optimizer()` 构建：

```python
optax.chain(
    optax.clip_by_global_norm(max_grad_norm),
    optax.adam(lr),
)
```

---

## QNetwork 架构

Q 网络将观测映射到每个离散动作的 Q 值: `obs → Q(s, a) for all a`

```
输入 obs (obs_dim,)
  → Linear(obs_dim, 128) → ReLU
  → Linear(128, 128)     → ReLU
  → Linear(128, n_actions)
输出 q_values (n_actions,)
```

注意激活函数为 **ReLU**（PPO 的 Actor/Critic 使用 tanh）。

---

## Epsilon-Greedy 探索 + Linear Schedule

DQN 使用 ε-greedy 策略进行探索，ε 按线性计划从 `epsilon_start` 衰减到 `epsilon_end`：

```python
frac = clip(step / epsilon_decay_steps, 0.0, 1.0)
epsilon = epsilon_start + frac * (epsilon_end - epsilon_start)

# 以概率 ε 选择随机动作，否则选择 greedy 动作
if random() < epsilon:
    action = random_action
else:
    action = argmax(Q(obs))
```

**时间线** (默认参数):
- Step 0: ε = 1.0 (完全随机)
- Step 25,000: ε ≈ 0.505 (一半随机)
- Step 50,000+: ε = 0.01 (几乎 greedy)

评估时设置 `explore=False` 使用纯 greedy 策略。

---

## 目标网络更新

DQN 使用 **硬更新 (hard update)** 策略：每隔 `target_update_freq` 步将在线网络参数完整复制到目标网络。

```python
# 在 update() 内部
new_step = state.step + 1
new_target_params = jax.lax.cond(
    new_step % config.target_update_freq == 0,
    lambda: new_params,        # 复制在线网络
    lambda: state.target_params,  # 保持不变
)
```

TD target 计算：

```python
# Q(s, a) — 在线网络对当前动作的估值
q_values = Q_online(obs)[action]

# target — 目标网络提供稳定目标
next_q_max = max(Q_target(next_obs))
target = reward + gamma * next_q_max * (1.0 - done)

# MSE 损失
loss = mean((q_values - stop_gradient(target))²)
```

---

## ReplayBuffer 使用

```python
from vibe_rl.dataprotocol.replay_buffer import ReplayBuffer

# 初始化 — 指定容量和观测形状
buffer = ReplayBuffer(capacity=50_000, obs_shape=(4,))

# 存储 transition
buffer.push(obs=obs, action=action, reward=reward, next_obs=next_obs, done=done)

# 采样 mini-batch (返回 Transition of jax arrays)
batch = buffer.sample(batch_size=64)
```

**设计细节**:

- **存储**: 预分配 numpy 数组，O(1) 循环插入
- **采样**: 均匀随机采样，返回 `Transition` (内含 jax arrays)
- **不可 JIT**: 缓冲区在 Python 作用域中管理，`sample()` 输出传入 JIT 编译的 `update()`
- **连续动作**: 支持 `action_shape` 和 `action_dtype` 参数

---

## 状态容器

```python
class DQNState(NamedTuple):
    params: Params          # 在线 Q 网络 (Equinox model)
    target_params: Params   # 目标 Q 网络
    opt_state: OptState     # Optax 优化器状态
    step: chex.Array        # 训练步计数器 (标量)
    rng: chex.PRNGKey       # PRNG key
```

---

## 代码示例

### 基本训练 (CartPole)

```python
import numpy as np
import jax
from vibe_rl.algorithms.dqn import DQN, DQNConfig
from vibe_rl.dataprotocol.replay_buffer import ReplayBuffer
from vibe_rl.env import make
from vibe_rl.env.wrappers import AutoResetWrapper

config = DQNConfig(
    hidden_sizes=(128, 128),
    lr=1e-3,
    gamma=0.99,
    batch_size=64,
    target_update_freq=1_000,
    epsilon_start=1.0,
    epsilon_end=0.01,
    epsilon_decay_steps=20_000,
)

env, env_params = make("CartPole-v1")
env = AutoResetWrapper(env)
env_params = env.default_params()

rng = jax.random.PRNGKey(42)
rng, env_key, agent_key = jax.random.split(rng, 3)

obs, env_state = env.reset(env_key, env_params)
state = DQN.init(agent_key, obs_shape=(4,), n_actions=2, config=config)
buffer = ReplayBuffer(capacity=50_000, obs_shape=(4,))

for step in range(100_000):
    # ε-greedy 动作选择
    action, state = DQN.act(state, obs, config=config, explore=True)

    rng, step_key = jax.random.split(rng)
    next_obs, env_state, reward, done, info = env.step(
        step_key, env_state, action, env_params,
    )

    # 存入回放缓冲区
    buffer.push(
        obs=np.asarray(obs),
        action=int(action),
        reward=float(reward),
        next_obs=np.asarray(next_obs),
        done=float(done),
    )
    obs = next_obs

    # 缓冲区足够大时开始训练
    if len(buffer) >= 1_000:
        batch = buffer.sample(config.batch_size)
        state, metrics = DQN.update(state, batch, config=config)

    if (step + 1) % 10_000 == 0:
        print(f"Step {step+1} | loss={float(metrics.loss):.4f}")
```

### 自定义超参

```python
config = DQNConfig(
    hidden_sizes=(256, 256),       # 更大的网络
    lr=5e-4,                       # 较低学习率
    batch_size=128,                # 更大的 batch
    target_update_freq=500,        # 更频繁的目标更新
    epsilon_start=0.5,             # 较低的初始探索
    epsilon_end=0.05,              # 较高的最终探索
    epsilon_decay_steps=100_000,   # 更慢的衰减
)
```

### 评估 (Greedy)

```python
# explore=False 禁用 ε-greedy，使用纯 greedy 策略
action, state = DQN.act(state, obs, config=config, explore=False)
```

---

## API 参考

| 方法 | 签名 | 说明 |
|------|------|------|
| `DQN.init` | `(rng, obs_shape, n_actions, config) → DQNState` | 初始化 agent |
| `DQN.act` | `(state, obs, config=, explore=True) → (action, state)` | ε-greedy 动作选择 |
| `DQN.update` | `(state, batch, config=) → (state, metrics)` | 单步梯度更新 |

### DQNMetrics

| 字段 | 说明 |
|------|------|
| `loss` | TD 均方误差损失 |
| `q_mean` | batch 中 Q 值的均值 |
| `epsilon` | 当前 ε 值 |

---

## 源码路径

- Agent: `src/vibe_rl/algorithms/dqn/agent.py`
- Config: `src/vibe_rl/algorithms/dqn/config.py`
- Network: `src/vibe_rl/algorithms/dqn/network.py`
- Types: `src/vibe_rl/algorithms/dqn/types.py`
- Runner: `src/vibe_rl/runner/train_dqn.py`
- Replay Buffer: `src/vibe_rl/dataprotocol/replay_buffer.py`
- Example: `examples/train_dqn_cartpole.py`
