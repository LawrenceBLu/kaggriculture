import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from models import ActorCritic
from copy import deepcopy
from kaggyculture import invalid_action, setup_non_tile_observation, setup_tile_obs
from action_space import farmhand_action_space, market_action_space


class ReplayBuffer():
  def __init__(self):
    self.states = []
    self.farm_actions = []
    self.market_actions = []
    self.rewards = []
    self.dones = []
    self.state_values = []
    self.log_probs = []
    self.lstm_hidden = []
    self.advantages = []
    self.returns = []
    self.farm_masks = []
    self.market_masks = []
    self.internal_obs = {}

  def clear(self):
    del self.states[:]
    del self.farm_actions[:]
    del self.market_actions[:]
    del self.rewards[:]
    del self.dones[:]
    del self.state_values[:]
    del self.log_probs[:]
    del self.lstm_hidden[:]
    del self.advantages[:]
    del self.returns[:]
    del self.farm_masks[:]
    del self.market_masks[:]
    del internal_obs

  def store(self, state, farm_actions, market_actions, reward, done, state_value, log_prob, lstm_hidden, farm_masks, market_masks):
    self.states.append(state)
    self.farm_actions.append(farm_actions)
    self.market_actions.append(market_actions)
    self.rewards.append(reward)
    self.dones.append(done)
    self.state_values.append(state_value)
    self.log_probs.append(log_prob)
    self.lstm_hidden.append(lstm_hidden)
    self.farm_masks.append(farm_masks)
    self.market_masks.append(market_masks)

  def get_batch(self, batch_size):
    batch_data = ReplayBuffer()
    for i in range(0, len(self.states), batch_size):
      batch_data.states = self.states[i:i+batch_size]
      batch_data.farm_actions = self.farm_actions[i:i+batch_size]
      batch_data.market_actions = self.market_actions[i:i+batch_size]
      batch_data.log_probs = self.log_probs[i:i+batch_size]
      batch_data.lstm_hidden = self.lstm_hidden[i:i+batch_size]
      batch_data.advantage = self.advantages[i:i+batch_size]
      batch_data.returns = self.advantages[i:i+batch_size]

      yield batch_data
    
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
  def __init__(self, env, grid_size, cnn_dim, non_tile_len, linear_dim, lstm_dim, lstm_layers, farm_action_dim, num_hands, market_action_dim, num_orders, lr, eps, epochs, batch_size, value_c, entr_c, gae_lambda, gamma, device):
    self.replay_buffer = ReplayBuffer()
    self.model = ActorCritic(grid_size, cnn_dim, non_tile_len, linear_dim, lstm_dim, lstm_layers, farm_action_dim, num_hands, market_action_dim, num_orders).to(device)
    self.env = env
    
    self.optim = torch.optim.AdamW(self.model.parameters(), lr=lr)

    # self.old_model = ActorCritic(grid_size, cnn_dim, non_tile_len, linear_dim, lstm_dim, lstm_layers, action_dim).to(device)
    # self.old_model.load_state_dict(self.model.state_dict())
    self.mse = nn.MSELoss()
    self.hidden = (torch.zeros(lstm_layers, 1, lstm_dim, device=device), torch.zeros(lstm_layers, 1, lstm_dim, device=device))

    self.eps = eps
    self.device = device
    self.value_c = value_c
    self.entr_c = entr_c
    self.epochs = epochs
    self.batch_size = batch_size
    self.gae_lambda = gae_lambda
    self.gamma = gamma

    self.last_obs = None
    self.last_hidden = None

    self.all_farm_actions = farmhand_action_space()
    self.all_market_actions = market_action_space()

    self.init_env()

  def init_env(self):
    state, _ = self.env.reset()
    self.last_obs = state
    self.last_ep_starts = [True]

  def action_idx_to_string(self, farmer_action_idx, market_action_idx):
    action_dict = {}
    action_dict['farmer'] = [self.farmer_action_space(self.farmer_action_idx[0])]
    
    farmhand_actions = []
    if len(farmer_action_idx > 0):
      for farmhand_action in farmer_action_idx[1:]:
        farmhand_actions.append([self.farmer_action_space[farmhand_action]]) 
      
      action_dict['hands'] = farmhand_actions
    
    market_actions = []
    for market_action in market_action_idx:
      if self.market_action_space[market_action] != "PASS": market_actions.append([self.market_action_space[market_action]])
    
    action_dict['market'] = market_actions
    
    return action_dict

  def process_obs(self, obs):
    # Add function to split up obs into grid and non_tiles
    grid = setup_tile_obs(obs)
    non_tiles = setup_non_tile_observation(obs)

    return grid, non_tiles

  def create_masks(self, obs):
    farm_mask = list(map(lambda x: invalid_action(obs, x), self.all_farm_actions))
    market_mask = list(map(lambda x: invalid_action(obs, x), self.all_market_actions))

    return farm_mask, market_mask

  def collect_rollout(self, n_rollout_steps):
    n_steps = 0
    self.replay_buffer.clear()
    lstm_hidden = deepcopy(self.last_hidden)

    while n_steps < n_rollout_steps:
      with torch.no_grad():
        grid, non_tiles = self.process_obs(self.last_obs)
        farm_mask, market_mask = self.create_masks(self.last_obs)
        grid = torch.tensor(grid, device=self.device)
        non_tiles = torch.tensor(non_tiles, device=self.device)
        ep_starts = torch.tensor(self.last_ep_starts, dtype=torch.float32, device=self.device)
        farm_actions, market_actions, farm_masks, market_masks, log_prob, state_value, lstm_hidden = self.model(grid, non_tiles, lstm_hidden, ep_starts, farm_mask, market_mask, self.replay_buffer.internal_obs)
        # turn action indices into string

      action = self.action_idx_to_string(farm_actions, market_actions)
      new_state, reward, terminated, truncated, info = self.env.step(action)
      n_steps += 1

      if terminated == False and truncated == True and info.get("terminal_observation") is not None:
        terminal_grid, terminal_non_tiles = self.process_obs(info.get("terminal_observation"))
        with torch.no_grad():
          ep_starts = torch.tensor([False], dtype=torch.float32, device=self.device)
          terminal_value = self.model.predict_values(terminal_grid, terminal_non_tiles, lstm_hidden, ep_starts)
        reward += self.gamma * terminal_value

      self.replay_buffer.store(
        state = self.last_obs,
        farm_actions = farm_actions,
        market_actions = market_actions,
        reward = reward,
        done = self.last_ep_starts,
        state_value = state_value,
        log_prob = log_prob,
        lstm_hidden = self.last_hidden,
        farm_masks = farm_masks,
        market_masks = market_masks
      )

      self.last_obs = new_state
      self.last_ep_starts = terminated or truncated
      self.last_hidden = lstm_hidden

      with torch.no_grad():
        grid, non_tiles = self.process_obs(new_state)
        done = terminated or truncated
        ep_starts = torch.tensor(done, dtype=torch.float32, device=self.device)
        state_value = self.model.predict_values(grid, non_tiles, lstm_hidden, ep_starts)

      self.replay_buffer.compute_gae(state_value, done, self.gae_lambda, self.gamma)


  def train(self):
    for _ in range(self.epochs):
      for batch_data in self.replay_buffer.get_batches(self.batch_size):
        grids, non_tiles = zip(*[self.process_obs(obs) for obs in batch_data.states])
        grids, non_tiles = list(grids), list(non_tiles)
        old_grid = torch.squeeze(torch.stack(grids, dim=0)).detach().to(self.device)
        old_non_tiles = torch.squeeze(torch.stack(non_tiles, dim=0)).detach().to(self.device)
        old_farm_actions = torch.squeeze(torch.stack(batch_data.farm_actions, dim=0)).detach().to(self.device)
        old_market_actions = torch.squeeze(torch.stack(batch_data.market_actions, dim=0)).detach().to(self.device)
        old_log_probs = torch.squeeze(torch.stack(batch_data.log_probs, dim=0)).detach().to(self.device)

        log_probs, entropy, state_values = self.model.evaluate(old_grid, old_non_tiles, batch_data.lstm_hiddens, old_farm_actions, old_market_actions)

        ratio = torch.exp(log_probs - old_log_probs)

        surr1 = ratio * batch_data.advantage
        surr2 = torch.clamp(ratio, 1-self.eps, 1+self.eps) * batch_data.advantage

        loss = -torch.min(surr1, surr2) + self.value_c * self.mse(state_values, batch_data.returns) - self.entr_c * entropy

        self.optim.zero_grad()
        loss.mean().backward()
        self.optim.step()

    self.old_model.load_state_dict(self.model.state_dict())
    self.replay_buffer.clear()