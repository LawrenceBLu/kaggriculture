import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

def run():
  env = gym.make("CartPole-v1", render_mode="rgb_array")
  input_len = env.observation_space.shape[0]
  action_dim = int(env.action_space.n)
  hidden_len = 32
  actor_lr = 5e-4
  critic_lr = 1e-4
  value_c = 0.5
  entr_c = 0.01
  eps = 0.2
  epochs = 2
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  ppo = PPO(input_len, action_dim, hidden_len, actor_lr, critic_lr, eps, epochs, value_c, entr_c, device)

  episodes = 0
  max_episodes = 1000
  rewards = []

  for _ in range(max_episodes):
    state, info = env.reset()
    curr_reward = 0

    while True:
      action = ppo.get_action(state)
      state, reward, terminated, truncated, _ = env.step(action)
      done = terminated or truncated
      ppo.store(reward, done)

      curr_reward += reward

      if done:
        ppo.train()
        break

    rewards.append(curr_reward)

  env.close()

  return rewards