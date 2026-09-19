from kaggle_environments import make
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from ppo import PPO

def run():
  env = make("kaggriculture", debug=True)

  grid_size = 11
  cnn_dim = 128
  non_tile_len = 9
  linear_dim = 128
  lstm_dim = 256
  lstm_layers = 1
  farm_action_dim = 111
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
  
  ppo = PPO(env, grid_size, cnn_dim, non_tile_len, linear_dim, lstm_dim, lstm_layers, farm_action_dim, num_hands, market_action_dim, num_orders, lr, eps, epochs, batch_size, value_c, entr_c, gae_lambda, gamma, device)

  max_steps = 1000
  num_iterations = 2



  for _ in range(num_iterations):
    ppo.collect_rollout(max_steps)
    ppo.train()

  env.close()


if __name__ == "__main__":
  run()