from kaggle_environments import make
import numpy as np
import torch
import torch.nn as nn
from ppo import PPO
from env import KaggleEnvWrapper
import gymnasium as gym
from action_space import farmhand_action_space, market_action_space

def make_env(env_name, num_channels, non_tile_len, opponent_agent="random"):
    def _init():
        env = KaggleEnvWrapper(env_name, num_channels, non_tile_len, opponent_agent)
        return env
    return _init

def run():
  grid_size = 31
  cnn_dim = 256
  non_tile_len = 97
  non_tile_dim = 64
  linear_dim = 256
  lstm_dim = 256
  lstm_layers = 1
  farm_action_dim = len(farmhand_action_space())
  num_hands = 5
  market_action_dim = len(market_action_space())
  num_orders = 3
  chunk_len = 72
  batch_size = 10
  lr = 1e-4
  clipping_eps = 0.2
  adam_eps = 1e-5
  epochs = 3
  value_c = 0.5
  entr_c = 0.01
  gae_lambda = 0.99
  gamma = 0.95
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  num_envs = 8
  envs = gym.vector.AsyncVectorEnv([
      make_env("kaggriculture", grid_size, non_tile_len, opponent_agent="starter")
      for i in range(num_envs)
  ])
  
  ppo = PPO(envs, num_envs, grid_size, cnn_dim, non_tile_len, non_tile_dim, linear_dim, lstm_dim, lstm_layers, farm_action_dim, num_hands, market_action_dim, num_orders, lr, adam_eps, clipping_eps, epochs, chunk_len, batch_size, value_c, entr_c, gae_lambda, gamma, device)

  num_rollouts = 1
  max_steps = 720 * num_rollouts
  num_iterations = 10


  for i in range(num_iterations):
    ppo.collect_rollout(max_steps)
    ppo.train()
    ppo.save_model()

if __name__ == "__main__":
  run()