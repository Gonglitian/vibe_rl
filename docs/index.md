---
layout: home

hero:
  name: vibe-rl
  text: Pure JAX Reinforcement Learning
  tagline: Fast, functional, and hackable. End-to-end JIT-compiled training with zero Python overhead.
  actions:
    - theme: brand
      text: Get Started
      link: /guide/getting-started
    - theme: alt
      text: View on GitHub
      link: https://github.com/Gonglitian/vibe_rl

features:
  - title: Pure JAX
    details: Every training loop compiles end-to-end with jax.jit. Rollout, GAE, and SGD all run inside lax.scan — no Python loop overhead.
  - title: 3 Algorithms
    details: PPO, DQN, and SAC with a uniform functional API. Stateless namespaces, explicit state, and frozen dataclass configs.
  - title: Multi-GPU / TPU
    details: Single-line jax.pmap scaling with automatic gradient synchronization via lax.pmean. Zero code changes to the algorithm.
  - title: Data Pipeline
    details: Offline RL dataset loading, transforms, and normalization. Compatible with LeRobot and HuggingFace datasets.
  - title: Inference Serving
    details: Export trained policies and serve them over WebSocket with msgpack serialization for real-time robot control.
  - title: Reward Plotting
    details: Built-in reward curve visualization with smoothing, multi-run comparison, and automatic plot generation after training.
---
