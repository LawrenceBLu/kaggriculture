import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from models import ActorCritic
from copy import deepcopy
from kaggyculture import invalid_action, setup_non_tile_observation, setup_tile_obs, update_internal_obs
from action_space import farmhand_action_space, market_action_space
from env import Env

# torch.autograd.set_detect_anomaly(True)

class ReplayBuffer():
  def __init__(self):
    self.states = []
    self.farm_actions = []
    self.market_actions = []
    self.rewards = []
    self.dones = []
    self.state_values = []
    self.log_probs = []
    self.lstm_hiddens = []
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
    del self.lstm_hiddens[:]
    del self.farm_masks[:]
    del self.market_masks[:]
    self.internal_obs = {}
    self.advantages = []
    self.returns = []

  def store(self, state, farm_actions, market_actions, reward, done, state_value, log_prob, lstm_hidden, farm_masks, market_masks):
    self.states.append(state)
    self.farm_actions.append(farm_actions)
    self.market_actions.append(market_actions)
    self.rewards.append(reward)
    self.dones.append(done)
    self.state_values.append(state_value)
    self.log_probs.append(log_prob)
    self.lstm_hiddens.append(lstm_hidden)
    self.farm_masks.append(farm_masks)
    self.market_masks.append(market_masks)

  def get_batches(self, batch_size):
    batch_data = ReplayBuffer()
    for i in range(0, len(self.states), batch_size):
      batch_data.states = self.states[i:i+batch_size]
      batch_data.farm_actions = self.farm_actions[i:i+batch_size]
      batch_data.market_actions = self.market_actions[i:i+batch_size]
      batch_data.farm_masks = self.farm_masks[i:i+batch_size]
      batch_data.market_masks = self.market_masks[i:i+batch_size]
      batch_data.log_probs = self.log_probs[i:i+batch_size]
      batch_data.lstm_hiddens = self.lstm_hiddens[i:i+batch_size]
      batch_data.advantages = self.advantages[i:i+batch_size]
      batch_data.returns = self.advantages[i:i+batch_size]
      batch_data.dones = self.dones[i:i+batch_size]

      yield batch_data
    
  def compute_gae(self, next_value, next_done, gae_lambda, gamma, device):
      '''
      gae_lambda: future discount
      gamma: smoothing paramater
  
      '''
      values = torch.squeeze(torch.stack(self.state_values, dim=0)).detach().to(device)

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

      self.advantages = (advantage - advantage.mean())/(advantage.std() + 1e-8)
      self.returns = returns  
  

class PPO():
  def __init__(self, env, grid_size, cnn_dim, non_tile_len, linear_dim, lstm_dim, lstm_layers, farm_action_dim, num_hands, market_action_dim, num_orders, lr, eps, epochs, batch_size, value_c, entr_c, gae_lambda, gamma, device):
    self.replay_buffer = ReplayBuffer()
    self.model = ActorCritic(grid_size, cnn_dim, non_tile_len, linear_dim, lstm_dim, lstm_layers, farm_action_dim, num_hands, market_action_dim, num_orders).to(device)
    self.env = Env(env)
    
    self.optim = torch.optim.AdamW(self.model.parameters(), lr=lr)

    # self.old_model = ActorCritic(grid_size, cnn_dim, non_tile_len, linear_dim, lstm_dim, lstm_layers, action_dim).to(device)
    # self.old_model.load_state_dict(self.model.state_dict())
    self.mse = nn.MSELoss()

    self.lstm_layers = lstm_layers
    self.lstm_dim = lstm_dim

    self.eps = eps
    self.device = device
    self.value_c = value_c
    self.entr_c = entr_c
    self.epochs = epochs
    self.batch_size = batch_size
    self.gae_lambda = gae_lambda
    self.gamma = gamma
    self.steps = 0

    self.last_obs = None

    self.farmer_action_space = farmhand_action_space()
    self.market_action_space = market_action_space()

    # self.init_env()

  def init_env(self):
    state = self.env.reset()
    self.last_obs = state
    self.last_ep_starts = True
    self.reset_hidden()

  def reset_hidden(self):
    self.last_hidden = (torch.zeros(self.lstm_layers, 1, self.lstm_dim, device=self.device), torch.zeros(self.lstm_layers, 1, self.lstm_dim, device=self.device))


  def action_idx_to_string(self, farmer_action_idx, market_action_idx):
    action_dict = {}
    farmer_action = self.farmer_action_space[int(farmer_action_idx[0])]
    if type(farmer_action) == str:
      action_dict['farmer'] = [farmer_action]
    else:
      action_dict['farmer'] = list(farmer_action)
    
    farmhand_actions = []
    if len(farmer_action_idx) > 0:
      for farmhand_action in farmer_action_idx[1:]:
        hand_action = self.farmer_action_space[int(farmhand_action)]
        if type(hand_action) == str:
          farmhand_actions.append(hand_action)
        else:
          farmhand_actions.append(list(hand_action))
      
      action_dict['hands'] = farmhand_actions
    
    market_actions = []
    for market_action in market_action_idx:
      market_action = self.market_action_space[int(market_action)]
      if market_action != "PASS":
        if type(market_action) == str:
          market_actions.append([market_action])
        else:
          market_actions.append(list(market_action))
    
    action_dict['market'] = market_actions
    
    return action_dict

  def process_obs(self, obs):
    # Add function to split up obs into grid and non_tiles
    grid = setup_tile_obs(obs['observation'])
    non_tiles = setup_non_tile_observation(obs['observation'])
    self.replay_buffer.internal_obs = deepcopy(obs["observation"])
    update_internal_obs(self.replay_buffer.internal_obs)

    return grid, non_tiles

  def create_masks(self, obs):
    # First farm_mask is for farmer only

    farm_mask = list(map(lambda x: invalid_action(obs["observation"], x), self.farmer_action_space.values()))
    market_mask = list(map(lambda x: invalid_action(obs["observation"], x), self.market_action_space.values()))

    return farm_mask, market_mask

  def collect_rollout(self, n_rollout_steps):
    print("COLLECTING ROLLOUT")
    n_steps = 0
    self.replay_buffer.clear()
    self.init_env()
    lstm_hidden = deepcopy(self.last_hidden)

    while n_steps < n_rollout_steps:
      with torch.no_grad():
        grid, non_tiles = self.process_obs(self.last_obs)
        farm_mask, market_mask = self.create_masks(self.last_obs)

        grid = torch.from_numpy(np.array(grid)).to(dtype=torch.float32, device=self.device).permute(2, 0, 1)
        non_tiles = torch.tensor(non_tiles, dtype=torch.float32, device=self.device).unsqueeze(0)
        ep_starts = torch.tensor(self.last_ep_starts, dtype=torch.float32, device=self.device)

        farm_actions, market_actions, farm_masks, market_masks, log_prob, state_value, lstm_hidden = self.model(grid, non_tiles, lstm_hidden, ep_starts, farm_mask, market_mask, self.replay_buffer.internal_obs)
        # turn action indices into string

      action = self.action_idx_to_string(farm_actions, market_actions)
      actions = [action]
      actions.extend([{"market":[], "farmer": ["PASS"]}])
      new_state, done, reward = self.env.step(actions)
      # new_state, reward, terminated, truncated, info = self.env.step(action)
      n_steps += 1

      # if terminated == False and truncated == True and info.get("terminal_observation") is not None:
      #   terminal_grid, terminal_non_tiles = self.process_obs(info.get("terminal_observation"))
      #   with torch.no_grad():
      #     ep_starts = torch.tensor([False], dtype=torch.float32, device=self.device)
      #     terminal_value = self.model.predict_values(terminal_grid, terminal_non_tiles, lstm_hidden, ep_starts)
      #   reward += self.gamma * terminal_value


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
      self.last_ep_starts = done
      self.last_hidden = lstm_hidden

    with torch.no_grad():
      grid, non_tiles = self.process_obs(new_state)
      grid = torch.from_numpy(np.array(grid)).to(dtype=torch.float32, device=self.device).permute(2, 0, 1)
      non_tiles = torch.tensor(non_tiles, dtype=torch.float32, device=self.device).unsqueeze(0)
      ep_starts = torch.tensor([done], dtype=torch.float32, device=self.device)
      state_value = self.model.predict_value(grid, non_tiles, lstm_hidden, ep_starts)

    self.replay_buffer.compute_gae(state_value, done, self.gae_lambda, self.gamma, self.device)
    print("FINISHED ROLLOUT")


  def train(self):
    print("START TRAINING")
    self.model.train()
    for _ in range(self.epochs):
      for batch_data in self.replay_buffer.get_batches(self.batch_size):
        grids, non_tiles = zip(*[self.process_obs(obs) for obs in batch_data.states])
        grids, non_tiles = list(grids), list(non_tiles)

        old_grid = torch.squeeze(torch.stack([torch.from_numpy(np.array(g)).to(dtype=torch.float32).permute(2, 0, 1) for g in grids], dim=0)).detach().to(self.device)
        old_non_tiles = torch.squeeze(torch.stack([torch.as_tensor(non_tile, dtype=torch.float32).unsqueeze(0) for non_tile in non_tiles], dim=0)).detach().to(self.device)
        old_farm_actions = torch.squeeze(torch.stack([torch.as_tensor(farm_action) for farm_action in batch_data.farm_actions], dim=0)).detach().to(self.device)
        old_market_actions = torch.squeeze(torch.stack([torch.as_tensor(market_action) for market_action in batch_data.market_actions], dim=0)).detach().to(self.device)
        old_log_probs = torch.squeeze(torch.stack(batch_data.log_probs, dim=0)).detach().to(self.device)
        old_dones = torch.tensor(batch_data.dones, dtype=torch.float32).detach()
        old_farm_masks = torch.stack([torch.as_tensor(farm_mask) for farm_mask in batch_data.farm_masks], dim=0).detach().to(self.device)
        old_market_masks = torch.stack([torch.as_tensor(market_mask) for market_mask in batch_data.market_masks], dim=0).detach().to(self.device)
        h_list, c_list = zip(*batch_data.lstm_hiddens)
        h_stack = torch.stack(h_list, dim=1).squeeze(dim=2)
        c_stack = torch.stack(c_list, dim=1).squeeze(dim=2)

        log_probs, entropy, state_values = self.model.evaluate(old_grid, old_non_tiles, (h_stack, c_stack), old_farm_actions, old_market_actions, old_farm_masks, old_market_masks, old_dones)

        # print("log_probs", log_probs)
        # print("old_log_probs", old_log_probs)
        assert torch.isfinite(log_probs).all(), f"log_prob has nan/inf: {log_probs}"
        # assert torch.isfinite(old_log_probs).all(), f"old_log_prob has nan/inf: {old_log_probs}"

        ratio = torch.exp(torch.clamp(log_probs - old_log_probs, min=-20, max=2))
        # assert torch.isfinite(ratio).all(), f"std has nan/inf: {ratio}"

        surr1 = ratio * batch_data.advantages
        surr2 = torch.clamp(ratio, 1-self.eps, 1+self.eps) * batch_data.advantages

        # print("adv std", batch_data.advantages.std())

        loss = -torch.min(surr1, surr2) + self.value_c * self.mse(state_values, batch_data.returns) - self.entr_c * entropy
        # print("loss:", loss)
        # print("policy_loss:", -torch.min(surr1, surr2))
        # print("value_loss:", (self.value_c * self.mse(state_values, batch_data.returns) - self.entr_c * entropy))
        # print("entropy:", entropy.item() if torch.is_tensor(entropy) else entropy)
        
        self.optim.zero_grad()
        loss.mean().backward()
        # for name, p in self.model.cnn.named_parameters():
        #   if p.grad is not None and not torch.isfinite(p.grad).all():
        #       print(f"{name}.grad is already non-finite, BEFORE clipping/step")
        total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
        # print(f"grad norm before clip: {total_norm}")
        self.optim.step()

    print("DONE")


  def save_model(self):
    torch.save(self.model.state_dict(), "./model_checkpoint.pt")

  def print_last_state(self):
    print(self.last_obs)