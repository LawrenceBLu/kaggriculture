import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from models import ActorCritic
from copy import deepcopy

class ReplayBuffer():
  def __init__(self):
    self.state_grid = []
    self.state_non_tiles = []
    self.actions = []
    self.rewards = []
    self.dones = []
    self.state_values = []
    self.log_probs = []
    self.lstm_hidden = []
    self.advantages = []
    self.returns = []

  def clear(self):
    del self.state_grid[:]
    del self.state_non_tiles[:]
    del self.actions[:]
    del self.rewards[:]
    del self.dones[:]
    del self.state_values[:]
    del self.log_probs[:]
    del self.lstm_hidden[:]
    del self.advantages[:]
    del self.returns[:]

  def compute_gae(self, next_value, next_done, gae_lambda, gamma):
      '''
      gae_lambda: future discount
      gamma: smoothing paramater
  
      '''
      values = torch.squeeze(torch.stack(self.state_values, dim=0)).detach().to(self.device)

      advantage = torch.zeros_like(values)
      last_gae = 0
  
      for t in reversed(range(len(self.rewards))):
        if t == len(self.rewards) - 1:
          next_val = next_value
          mask = 1.0 - next_done
        else:
          next_val = values[t+1]
          mask = 0 if self.dones[t] else 1
        delta = self.rewards[t] + gamma * next_val * mask - values[t]
        advantage[t] = last_gae = delta + gae_lambda * gamma * mask * last_gae
  
      returns = advantage + values

      self.advantages = advantage
      self.returns = returns  
  


class PPO():
  def __init__(self, env, grid_size, cnn_dim, non_tile_len, linear_dim, lstm_dim, lstm_layers, action_dim, lr, eps, epochs, value_c, entr_c, gae_lambda, gamma, device):
    self.replay_buffer = ReplayBuffer()
    self.model = ActorCritic(grid_size, cnn_dim, non_tile_len, linear_dim, lstm_dim, lstm_layers, action_dim).to(device)
    self.env = env
    
    self.optim = torch.optim.AdamW(self.model.parameters(), lr=lr)

    # self.old_model = ActorCritic(grid_size, cnn_dim, non_tile_len, linear_dim, lstm_dim, lstm_layers, action_dim).to(device)
    # self.old_model.load_state_dict(self.model.state_dict())
    self.mse = nn.MSELoss()
    self.hidden = (torch.zeros(lstm_layers, 1, lstm_dim, devic=device), torch.zeros(lstm_layers, 1, lstm_dim, devic=device))

    self.eps = eps
    self.device = device
    self.value_c = value_c
    self.entr_c = entr_c
    self.epochs = epochs
    self.gae_lambda = gae_lambda
    self.gamma = gamma

    self.last_obs = None
    self.last_hidden = None

    self.init_env()

  def init_env(self):
    state, _ = self.env.reset()
    self.last_obs = state
    self.last_ep_starts = [True]

  def get_action(self, state):
    with torch.no_grad():
      state = torch.FloatTensor(state).to(self.device)
      action, log_prob, state_value = self.model(state)

    self.replay_buffer.states.append(state)
    self.replay_buffer.actions.append(action)
    self.replay_buffer.state_values.append(state_value)
    self.replay_buffer.log_probs.append(log_prob)

    return action.item()

  def store(self, reward, done):
    self.replay_buffer.rewards.append(reward)
    self.replay_buffer.dones.append(done)

  def collect_rollout(self, n_rollout_steps):
    n_steps = 0
    self.replay_buffer.clear()
    lstm_hidden = deepcopy(self.last_hidden)

    while n_steps < n_rollout_steps:
      with torch.no_grad():
        grid, non_tiles = self.process_state(self.last_obs)
        grid = torch.tensor(grid, device=self.device)
        non_tiles = torch.tensor(non_tiles, device=self.device)
        ep_starts = torch.tensor(self.last_ep_starts, dtype=torch.float32, device=self.device)
        action, log_prob, state_value, lstm_hidden = self.model(grid, non_tiles, lstm_hidden, ep_starts)

      new_state, reward, terminated, truncated, info = self.env.step(action)
      n_steps += 1

      if terminated == False and truncated == True and info.get("terminal_observation") is not None:
        terminal_grid, terminal_non_tiles = self.process_state(info.get("terminal_observation"))
        with torch.no_grad():
          ep_starts = torch.tensor([False], dtype=torch.float32, device=self.device)
          terminal_value = self.model.predict_values(terminal_grid, terminal_non_tiles, lstm_hidden, ep_starts)
        reward += self.gamma * terminal_value

      self.replay_buffer.store(
        self.last_obs,
        action,
        reward,
        self.last_ep_starts,
        state_value,
        log_prob,
        self.last_hidden
      )

      self.last_obs = new_state
      self.last_ep_starts = terminated or truncated
      self.last_hidden = lstm_hidden

      with torch.no_grad():
        grid, non_tiles = self.process_state(new_state)
        done = terminated or truncated
        ep_starts = torch.tensor(done, dtype=torch.float32, device=self.device)
        state_value = self.model.predict_values(grid, non_tiles, lstm_hidden, ep_starts)

      self.replay_buffer.compute_gae(state_value, done, self.gae_lambda, self.gamma)


  def train(self):
    old_states = torch.squeeze(torch.stack(self.replay_buffer.states, dim=0)).detach().to(self.device)
    old_actions = torch.squeeze(torch.stack(self.replay_buffer.actions, dim=0)).detach().to(self.device)
    old_log_probs = torch.squeeze(torch.stack(self.replay_buffer.log_probs, dim=0)).detach().to(self.device)

    advantage = self.replay_buffer.advantages
    returns = self.replay_buffer.returns

    for _ in range(self.epochs):
      log_probs, entropy, state_values = self.model.evaluate(old_states, old_actions)

      ratio = torch.exp(log_probs - old_log_probs)

      surr1 = ratio * advantage
      surr2 = torch.clamp(ratio, 1-self.eps, 1+self.eps) * advantage

      loss = -torch.min(surr1, surr2) + self.value_c * self.mse(state_values, returns) - self.entr_c * entropy

      self.optim.zero_grad()
      loss.mean().backward()
      self.optim.step()

    self.old_model.load_state_dict(self.model.state_dict())

    self.replay_buffer.clear()