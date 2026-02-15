# Stoix JAX RL 架构设计笔记

> **来源**: [EdanToledo/Stoix](https://github.com/EdanToledo/Stoix) — 基于 JAX 的单智能体强化学习研究框架
> **本地路径**: `stoix_reference/`
> **许可证**: Apache 2.0

---

## 1. 项目总览

Stoix 是一个**研究导向**的端到端 JAX 强化学习框架，实现了 **23+ 种 RL 算法**，覆盖 on-policy、off-policy、model-based 和分布式训练。核心设计哲学：

- **端到端 JAX 编译**: 训练循环（含环境）全部可 JIT 编译
- **双架构范式**: Anakin（同步单进程）和 Sebulba（异步分布式）
- **Hydra 配置驱动**: 通过 YAML 组合实验配置，零代码修改切换算法/环境/网络
- **CleanRL 风格**: 每个算法一个文件，允许适度代码重复以保持可读性

---

## 2. 核心架构：Anakin vs Sebulba

这是整个框架最重要的设计决策。

### 2.1 Anakin（同步架构）

```
┌─────────────────────────────────────────────────────────┐
│  jax.pmap (across devices, axis_name="device")          │
│  ┌───────────────────────────────────────────────────┐  │
│  │  jax.vmap (across batch, axis_name="batch")       │  │
│  │  ┌─────────────────────────────────────────────┐  │  │
│  │  │  jax.lax.scan (across num_updates_per_eval) │  │  │
│  │  │    → _env_step (scan over rollout_length)   │  │  │
│  │  │    → compute advantages (GAE)               │  │  │
│  │  │    → _update_epoch (scan over epochs)       │  │  │
│  │  │        → _update_minibatch (scan)           │  │  │
│  │  └─────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────┘  │
│  Gradient sync: pmean("batch") → pmean("device")        │
└─────────────────────────────────────────────────────────┘
```

**关键特征:**
- 所有组件（环境、网络、优化器）在**同一进程**内运行
- 通过 `pmap` 实现多设备数据并行，`vmap` 实现批次并行
- 状态形状: `(n_devices, update_batch_size, num_envs, ...)`
- 整个 `learn` 函数被 `jax.pmap` 包裹（参考 `stoix/systems/ppo/anakin/ff_ppo.py:489`）
- **适用场景**: 单机多 GPU/TPU，环境本身是 JAX 实现的

### 2.2 Sebulba（异步分布式架构）

```
┌──────────────┐     OnPolicyPipeline     ┌──────────────┐
│  Actor线程1   │ ─────────────────────▶  │              │
│  Actor线程2   │ ─────────────────────▶  │  Learner线程  │
│  Actor线程N   │ ─────────────────────▶  │              │
└──────────────┘                          └──────┬───────┘
       ▲                                         │
       │         ParameterServer                 │
       └─────────────────────────────────────────┘

独立线程: Async Evaluator (单独设备)
```

**关键特征:**
- Actor 线程独立收集数据，通过 `OnPolicyPipeline` 发送到 Learner
- Learner 线程异步处理 rollout 数据，通过 `ParameterServer` 分发新参数
- LearnerState 更轻量：只包含 `params, opt_states, key`（无 env_state）
- 支持**非 JAX 环境**（Gymnasium、Envpool 等）
- **适用场景**: 大规模分布式训练，需要高吞吐量

### 2.3 选择建议

| 场景 | 推荐架构 |
|------|---------|
| JAX 环境 + 单机多卡 | Anakin |
| 非 JAX 环境 | Sebulba |
| 追求最高 SPS | Anakin |
| 需要异步评估 | Sebulba |
| 快速原型验证 | Anakin |

---

## 3. 训练循环深度解析（以 PPO Anakin 为例）

文件: `stoix/systems/ppo/anakin/ff_ppo.py`

### 3.1 数据收集（Rollout）

```python
# Line 81-135: _env_step
def _env_step(runner_state, _):
    # 1. 获取观测 → 可选标准化
    obs = runner_state.timestep.observation
    if running_statistics:
        obs = normalize(obs, running_statistics)

    # 2. 从策略采样动作 + 获取 Value
    action_dist = actor_apply_fn(params.actor_params, obs)
    action = action_dist.sample(seed=key)
    value = critic_apply_fn(params.critic_params, obs)

    # 3. 环境 step
    env_state, timestep = env.step(env_state, action)

    # 4. 处理 truncation（关键！）
    #    被截断的 episode 需要 bootstrap value
    bootstrap_value = critic_apply_fn(params.critic_params, next_obs)

    # 5. 打包 PPOTransition
    transition = PPOTransition(
        done, action, value, reward, log_prob, obs, bootstrap_value
    )
    return runner_state, transition
```

### 3.2 优势估计 (GAE)

```python
# Line 170: 计算 GAE
advantages, targets = _calculate_gae(traj_batch, last_val, gamma, gae_lambda)
```

### 3.3 参数更新

```python
# Line 184-284: _update_minibatch
def _update_minibatch(train_state, batch):
    # Actor Loss: PPO clip loss
    ratio = exp(new_log_prob - old_log_prob)
    loss = -min(ratio * advantages, clip(ratio, 1±ε) * advantages)
    total_loss = loss - ent_coef * entropy

    # Critic Loss: clipped value loss
    value_loss = clipped_value_fn(value, old_value, targets, clip_eps)

    # 梯度同步（双层 pmean）
    grads = jax.lax.pmean(grads, axis_name="batch")   # 设备内同步
    grads = jax.lax.pmean(grads, axis_name="device")   # 跨设备同步

    # 应用更新
    updates, new_opt_state = optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
```

**核心洞察: 双层 `pmean`**

这是 Stoix 中最重要的分布式训练模式之一：
1. `pmean("batch")` — 在同一设备上的多个 batch 间同步梯度
2. `pmean("device")` — 跨设备同步梯度

这对应 `vmap` 和 `pmap` 创建的两个命名轴。

---

## 4. 网络架构设计（Composable Networks）

文件目录: `stoix/networks/`

### 4.1 组合模式

```
Network = InputLayer → [PreTorso] → [Torso] → Head
```

**示例:**
```yaml
# stoix/configs/network/mlp.yaml
actor_network:
  pre_torso:
    _target_: stoix.networks.torso.MLPTorso
    layer_sizes: [256, 256]
    activation: relu
  action_head:
    _target_: stoix.networks.heads.CategoricalHead

critic_network:
  pre_torso:
    _target_: stoix.networks.torso.MLPTorso
    layer_sizes: [256, 256]
    activation: relu
  critic_head:
    _target_: stoix.networks.heads.ScalarCriticHead
```

### 4.2 核心组件

| 组件 | 文件 | 作用 |
|------|------|------|
| **InputLayer** | `inputs.py` | 处理不同类型输入（数组、结构化观测、嵌入+动作拼接） |
| **Torso** | `torso.py` | 网络主体（MLP、CNN、NoisyMLP） |
| **Head** | `heads.py` | 输出层（Categorical、Normal、Beta、Deterministic、Q-network 等） |
| **Dueling** | `dueling.py` | Value/Advantage 流分离 |
| **Distribution** | `distributions.py` | Tanh 变换分布、Beta 分布、多离散分布 |
| **ResNet** | `resnet.py` | 残差块 + 多种下采样策略 |
| **Layers** | `layers.py` | NoisyLinear（Rainbow 用）、StackedRNN |
| **ModelBased** | `model_based.py` | World Model（MuZero 用） |

### 4.3 Flax nn.Module 约定

所有网络组件都是 `flax.linen.Module`，遵循以下模式:

```python
class MLPTorso(nn.Module):
    layer_sizes: Sequence[int]
    activation: str = "relu"
    use_layer_norm: bool = False

    @nn.compact
    def __call__(self, x: chex.Array) -> chex.Array:
        for size in self.layer_sizes:
            x = nn.Dense(size)(x)
            if self.use_layer_norm:
                x = nn.LayerNorm()(x)
            x = parse_activation_fn(self.activation)(x)
        return x
```

### 4.4 Hydra 动态实例化

```python
# 通过 Hydra _target_ 动态创建网络
actor_pre_torso = hydra.utils.instantiate(config.network.actor_network.pre_torso)
actor_head = hydra.utils.instantiate(config.network.actor_network.action_head)
actor_network = FeedForwardActor(pre_torso=actor_pre_torso, action_head=actor_head)
```

这允许**零代码修改**切换网络架构，只需更改配置文件。

---

## 5. 类型系统（Type System）

文件: `stoix/base_types.py`

### 5.1 核心类型定义

```python
# 基础类型别名
Action: TypeAlias = chex.Array
Value: TypeAlias = chex.Array
Reward: TypeAlias = chex.Array
HiddenState: TypeAlias = chex.Array

# 观测结构
class Observation(NamedTuple):
    agent_view: chex.Array      # 观测特征
    action_mask: chex.Array     # 合法动作掩码
    step_count: Optional[chex.Array]

# 网络参数分组
class ActorCriticParams(NamedTuple):
    actor_params: FrozenDict
    critic_params: FrozenDict

class OnlineAndTarget(NamedTuple):  # DQN/SAC 用
    online: FrozenDict
    target: FrozenDict

# Learner 状态
class OnPolicyLearnerState(NamedTuple):
    params: Parameters
    opt_states: OptStates
    key: chex.PRNGKey
    env_state: WrapperState
    timestep: TimeStep

class OffPolicyLearnerState(NamedTuple):  # 多了 buffer_state
    params: Parameters
    opt_states: OptStates
    buffer_state: BufferState
    key: chex.PRNGKey
    env_state: WrapperState
    timestep: TimeStep
```

### 5.2 函数签名协议

```python
ActorApply = Callable[..., DistributionLike]
CriticApply = Callable[[FrozenDict, Observation], Value]

@runtime_checkable
class EvalFn(Protocol[StoixState]):
    def __call__(self, trained_params, key, running_statistics=None) -> EvaluationOutput: ...
```

**设计意义**: 使用 NamedTuple + TypeAlias 而非 dataclass，确保所有状态在 JAX 变换（jit/vmap/pmap/scan）中可被正确追踪为 PyTree。

---

## 6. 配置系统（Hydra Config）

目录: `stoix/configs/`

### 6.1 配置层级

```
configs/
├── default/          # 算法默认配置入口点
│   ├── anakin/       # Anakin 架构默认
│   └── sebulba/      # Sebulba 架构默认
├── arch/             # 架构级配置 (seed, total_envs, eval 设置)
├── system/           # 算法超参 (lr, gamma, clip_eps...)
├── network/          # 网络架构 (MLP, CNN, Dueling...)
├── env/              # 环境 (gymnax, brax, jumanji, gymnasium...)
├── logger/           # 日志后端 (console, wandb, neptune, tensorboard)
└── launcher/         # SLURM 集群启动配置
```

### 6.2 运行示例

```bash
# 基本运行
python stoix/systems/ppo/anakin/ff_ppo.py env=gymnax/cartpole

# 覆盖超参数
python stoix/systems/ppo/anakin/ff_ppo.py \
    env=brax/ant \
    system.rollout_length=64 \
    system.actor_lr=1e-4 \
    network=mlp \
    arch.total_timesteps=5e7

# 多运行 sweep
python stoix/systems/ppo/anakin/ff_ppo.py -m \
    system.actor_lr=1e-3,3e-4,1e-4 \
    system.ent_coef=0.01,0.001
```

### 6.3 配置组合机制

每个算法文件在 `defaults` 列表中指定要组合的配置:

```yaml
# stoix/configs/default/anakin/ff_ppo.yaml
defaults:
  - /arch: anakin
  - /system: ppo/ff_ppo
  - /network: mlp
  - /env: gymnax/cartpole
  - /logger: logger
```

---

## 7. 关键工程模式

### 7.1 jax.lax.scan 代替 Python for 循环

```python
# ❌ 错误：Python 循环无法被 JIT 编译
for step in range(rollout_length):
    state, transition = env_step(state)

# ✅ 正确：使用 scan，可被 JIT 编译
state, transitions = jax.lax.scan(env_step, state, None, length=rollout_length)
```

这是 JAX RL 框架中最基础也最重要的模式。所有循环都必须用 `scan` 实现才能被编译。

### 7.2 PRNG Key 分裂

```python
# JAX 要求显式管理随机数状态
key, actor_key, env_key = jax.random.split(key, 3)
action = policy.sample(seed=actor_key)
env_state, timestep = env.step(env_state, action, env_key)
```

### 7.3 Truncation 处理

```python
# Episode 被截断（time limit）时需要 bootstrap
# 这和 episode 自然终止 (done) 不同
bootstrap_value = jnp.where(
    timestep.last(),               # episode 结束
    jnp.where(
        truncated,                 # 但是是截断的
        critic(next_obs),          # → 需要 bootstrap
        jnp.zeros_like(value)      # → 自然终止，不需要
    ),
    jnp.zeros_like(value)
)
```

### 7.4 Off-Policy：Flashbax Replay Buffer

```python
# 初始化
buffer = fbx.make_item_buffer(max_length=buffer_size, min_length=warmup_steps)
buffer_state = buffer.init(sample_transition)

# 添加数据
buffer_state = buffer.add(buffer_state, transition_batch)

# 采样
batch = buffer.sample(buffer_state, key)
```

### 7.5 Target 网络 Polyak 平均

```python
# SAC/DQN: 慢更新 target 网络
target_params = optax.incremental_update(online_params, target_params, tau=0.005)
```

### 7.6 观测标准化（Running Statistics）

```python
# Warmup 阶段收集统计信息
running_stats = init_running_statistics(obs_shape)

# 训练中更新并标准化
running_stats = update_running_statistics(running_stats, obs_batch)
normalized_obs = normalize(obs, running_stats, min_std=1e-6, max_std=1e6)
```

---

## 8. 支持的算法一览

| 类别 | 算法 | 文件 |
|------|------|------|
| **Value-Based** | DQN, DDQN, C51, QR-DQN, M-DQN, PQN, Rainbow, R2D2 | `systems/q_learning/` |
| **Policy Gradient** | PPO, PPO-Penalty, DPO, REINFORCE | `systems/ppo/`, `systems/vpg/` |
| **Actor-Critic** | SAC, DDPG, TD3, D4PG | `systems/sac/`, `systems/ddpg/` |
| **Policy Optimization** | MPO, V-MPO, AWR, SPO | `systems/mpo/`, `systems/awr/`, `systems/spo/` |
| **Search-Based** | AlphaZero, MuZero, Sampled-AZ, Sampled-MZ | `systems/search/` |
| **分布式** | IMPALA | `systems/impala/` |

大多数算法都有 **离散动作** 和 **连续动作** 两个版本。

---

## 9. 环境集成

支持 **14 种环境框架**:

| 类型 | 框架 | JAX 原生? |
|------|------|-----------|
| 经典控制 | Gymnax | ✅ |
| 组合优化 | Jumanji | ✅ |
| 物理仿真 | Brax | ✅ |
| 手工制作 | CraftAx | ✅ |
| 网格世界 | XLand-MiniGrid | ✅ |
| 棋盘游戏 | Pgx | ✅ |
| 导航 | Navix | ✅ |
| 机器人 | Kinetix | ✅ |
| MuJoCo | MJC Playground | ✅ |
| 标准 Gym | Gymnasium | ❌ (通过 wrapper) |
| 批量并行 | Envpool | ❌ (通过 wrapper) |

JAX 原生环境可以使用 Anakin 架构实现端到端编译；非 JAX 环境需要使用 Sebulba 架构。

---

## 10. 日志和检查点

### 日志后端
- **Console**: 终端实时输出
- **WandB**: Weights & Biases 实验跟踪
- **Neptune**: Neptune ML 平台
- **TensorBoard**: TensorBoard 日志
- **JSON**: 标准 JSON 文件 + id-marl-eval 兼容

### 检查点
- 基于 **Orbax** 的检查点系统
- 支持训练恢复
- 可配置最大保留数量和保存间隔

---

## 11. 关键依赖

| 库 | 用途 |
|----|------|
| `jax` / `jaxlib` | 核心计算框架 |
| `flax` | 神经网络定义 |
| `optax` | 优化器 |
| `distrax` | 概率分布 |
| `rlax` | RL 损失函数 |
| `chex` | 类型检查和数组验证 |
| `flashbax` | 高效回放缓冲区 |
| `mctx` | 蒙特卡洛树搜索 |
| `hydra-core` | 配置管理 |
| `wandb` / `neptune` | 实验跟踪 |

---

## 12. 给工程师的建议

### 快速上手路径
1. 先读 `stoix/systems/ppo/anakin/ff_ppo.py` — 最清晰的完整训练循环示例
2. 理解 `base_types.py` 中的类型定义
3. 看 `configs/default/anakin/` 中的配置入口
4. 读 `networks/base.py` 理解网络组合方式

### 扩展新算法
1. 在 `systems/` 下创建新目录
2. 复制最接近的算法文件作为起点
3. 定义算法特有的 Transition 类型
4. 实现 `_update_step` 和 `learner_fn`
5. 添加对应的 `configs/system/` 和 `configs/default/` 配置

### 常见陷阱
- **不要在 JIT 区域内使用 Python 控制流**：用 `jax.lax.cond` / `jax.lax.scan` 代替 if/for
- **PRNG key 必须显式分裂**：忘记分裂会导致随机性丧失
- **状态形状必须包含设备维度**：pmap 要求第一维是 n_devices
- **NamedTuple 是不可变的**：更新状态需要 `state._replace(key=new_key)`
- **梯度同步顺序**：先 batch pmean，再 device pmean
