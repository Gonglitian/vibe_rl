import { defineConfig } from 'vitepress'

export default defineConfig({
  title: 'vibe-rl',
  description: 'Pure JAX Reinforcement Learning Library',

  themeConfig: {
    nav: [
      { text: 'Guide', link: '/guide/getting-started' },
      { text: 'Algorithms', link: '/algorithms/ppo' },
      { text: 'API', link: '/api/runner' },
    ],

    sidebar: [
      {
        text: 'Guide',
        items: [
          { text: 'Getting Started', link: '/guide/getting-started' },
          { text: 'Configuration', link: '/guide/configuration' },
          { text: 'Training', link: '/guide/training' },
          { text: 'Checkpointing', link: '/guide/checkpointing' },
          { text: 'Multi-GPU', link: '/guide/multi-gpu' },
          { text: 'Metrics & Logging', link: '/guide/metrics' },
          { text: 'Plotting', link: '/guide/plotting' },
        ],
      },
      {
        text: 'Algorithms',
        items: [
          { text: 'PPO', link: '/algorithms/ppo' },
          { text: 'DQN', link: '/algorithms/dqn' },
          { text: 'SAC', link: '/algorithms/sac' },
        ],
      },
      {
        text: 'Environments',
        items: [
          { text: 'Overview', link: '/environments/overview' },
          { text: 'Built-in Envs', link: '/environments/builtin' },
          { text: 'Wrappers', link: '/environments/wrappers' },
        ],
      },
      {
        text: 'Data Pipeline',
        items: [
          { text: 'Dataset & DataLoader', link: '/data/dataset' },
          { text: 'Transforms', link: '/data/transforms' },
          { text: 'Normalization', link: '/data/normalization' },
        ],
      },
      {
        text: 'Deployment',
        items: [
          { text: 'Policy Wrapper', link: '/deployment/policy-wrapper' },
          { text: 'WebSocket Serving', link: '/deployment/websocket' },
        ],
      },
      {
        text: 'API Reference',
        items: [
          { text: 'Runner', link: '/api/runner' },
          { text: 'Agent Protocol', link: '/api/agent-protocol' },
          { text: 'Types', link: '/api/types' },
          { text: 'Networks', link: '/api/networks' },
        ],
      },
    ],

    socialLinks: [
      { icon: 'github', link: 'https://github.com/Gonglitian/vibe_rl' },
    ],
  },
})
