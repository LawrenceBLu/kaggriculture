from kaggle_environments import make
import numpy as np
import torch
import torch.nn as nn
from ppo import PPO
from env import KaggleEnvWrapper
import gymnasium as gym

def make_env(env_name, num_channels, non_tile_len, opponent_agent="random"):
    def _init():
        env = KaggleEnvWrapper(env_name, num_channels, non_tile_len, opponent_agent)
        return env
    return _init

def run():
  grid_size = 26
  cnn_dim = 128
  non_tile_len = 97
  linear_dim = 128
  lstm_dim = 256
  lstm_layers = 1
  farm_action_dim = 113
  num_hands = 5
  market_action_dim = 41
  num_orders = 10
  batch_size = 360
  lr = 1e-4
  eps = 0.2
  epochs = 2
  value_c = 0.5
  entr_c = 0.01
  gae_lambda = 0.99
  gamma = 0.95
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  num_envs = 5
  envs = gym.vector.AsyncVectorEnv([
      make_env("kaggriculture", grid_size, non_tile_len, opponent_agent="random")
      for i in range(num_envs)
  ])
  
  ppo = PPO(envs, grid_size, cnn_dim, non_tile_len, linear_dim, lstm_dim, lstm_layers, farm_action_dim, num_hands, market_action_dim, num_orders, lr, eps, epochs, batch_size, value_c, entr_c, gae_lambda, gamma, device)

  num_rollouts = 5
  max_steps = (720 * num_rollouts) - 1
  num_iterations = 20


  for _ in range(num_iterations):
    ppo.collect_rollout(max_steps)
    ppo.train()
    ppo.save_model()

if __name__ == "__main__":
  run()