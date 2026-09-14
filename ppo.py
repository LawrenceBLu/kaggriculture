import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from models import ActorCritic, CNN

class ReplayBuffer():
  def __init__(self):
    self.states = []
    self.actions = []
    self.rewards = []
    self.dones = []
    self.state_values = []
    self.log_probs = []
    self.lstm_hidden = []

  def clear(self):
    del self.states[:]
    del self.actions[:]
    del self.rewards[:]
    del self.dones[:]
    del self.state_values[:]
    del self.log_probs[:]
    del self.lstm_hidden[:]


class PPO():
  def __init__(self, grid_size, cnn_dim, non_tile_len, linear_dim, lstm_dim, lstm_layers, action_dim, lr, eps, epochs, value_c, entr_c, device):
    self.replay_buffer = ReplayBuffer()
    self.model = ActorCritic(grid_size, cnn_dim, non_tile_len, linear_dim, lstm_dim, lstm_layers, action_dim).to(device)
    
    self.optim = torch.optim.AdamW(self.model.parameters(), lr=lr)

    self.old_model = ActorCritic(grid_size, cnn_dim, non_tile_len, linear_dim, lstm_dim, lstm_layers, action_dim).to(device)
    self.old_model.load_state_dict(self.model.state_dict())
    self.mse = nn.MSELoss()
    self.hidden = (torch.zeros(lstm_layers, 1, lstm_dim, devic=device), torch.zeros(lstm_layers, 1, lstm_dim, devic=device))

    self.eps = eps
    self.device = device
    self.value_c = value_c
    self.entr_c = entr_c
    self.epochs = epochs

  def get_action(self, state):
    with torch.no_grad():
      state = torch.FloatTensor(state).to(self.device)
      action, log_prob, state_value = self.old_model(state)

    self.replay_buffer.states.append(state)
    self.replay_buffer.actions.append(action)
    self.replay_buffer.state_values.append(state_value)
    self.replay_buffer.log_probs.append(log_prob)

    return action.item()

  def store(self, reward, done):
    self.replay_buffer.rewards.append(reward)
    self.replay_buffer.dones.append(done)


  def gae(self, values, dones, rewards, gae_lambda=0.99, gamma=0.95):
    '''
    gae_lambda: future discount
    gamma: smoothing paramater

    '''
    advantage = torch.zeros_like(values)
    last_gae = 0

    for t in reversed(range(len(rewards))):
      if t == len(rewards) - 1:
        next_val = 0
      else:
        next_val = values[t+1]
      mask = 0 if dones[t] else 1
      delta = rewards[t] + gamma * next_val * mask - values[t]
      advantage[t] = last_gae = delta + gae_lambda * gamma * mask * last_gae

    returns = advantage + values

    return advantage, returns


  def train(self):
    old_states = torch.squeeze(torch.stack(self.replay_buffer.states, dim=0)).detach().to(self.device)
    old_actions = torch.squeeze(torch.stack(self.replay_buffer.actions, dim=0)).detach().to(self.device)
    old_state_values = torch.squeeze(torch.stack(self.replay_buffer.state_values, dim=0)).detach().to(self.device)
    old_log_probs = torch.squeeze(torch.stack(self.replay_buffer.log_probs, dim=0)).detach().to(self.device)

    advantage, returns = self.gae(old_state_values, self.replay_buffer.dones, self.replay_buffer.rewards)

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