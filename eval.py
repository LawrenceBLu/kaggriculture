from kaggle_environments import make
import numpy as np
import torch
import torch.nn as nn
from ppo import PPO
import json
from copy import deepcopy
from kaggyculture import invalid_action, setup_non_tile_observation, setup_tile_obs, update_internal_obs
from action_space import farmhand_action_space, market_action_space
from models import ActorCritic

grid_size = 26
cnn_dim = 256
non_tile_len = 97
non_tile_dim = 64
linear_dim = 256
lstm_dim = 256
lstm_layers = 1
farm_action_dim = 113
num_hands = 5
market_action_dim = 41
num_orders = 10
batch_size = 360
lr = 1e-4
clipping_eps = 0.2
adam_eps = 1e-5
epochs = 5
value_c = 0.5
entr_c = 0.01
gae_lambda = 0.99
gamma = 0.95

model = ActorCritic(grid_size, cnn_dim, non_tile_len, non_tile_dim, linear_dim, lstm_dim, lstm_layers, farm_action_dim, num_hands, market_action_dim, num_orders)
weights = torch.load("model_checkpoint.pt", weights_only=True)
model.load_state_dict(weights)
model.eval()

internal_obs = {}
farmer_action_space = farmhand_action_space()
market_action_space = market_action_space()
last_start = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lstm_hidden = (torch.zeros(lstm_layers, 1, lstm_dim, device=device), torch.zeros(lstm_layers, 1, lstm_dim, device=device))


def action_idx_to_string(farmer_action_idx, market_action_idx):
    action_dict = {}
    farmer_action = farmer_action_space[int(farmer_action_idx[0])]
    if type(farmer_action) == str:
      action_dict['farmer'] = [farmer_action]
    else:
      action_dict['farmer'] = list(farmer_action)
    
    farmhand_actions = []
    if len(farmer_action_idx) > 0:
      for farmhand_action in farmer_action_idx[1:]:
        hand_action = farmer_action_space[int(farmhand_action)]
        if type(hand_action) == str:
          farmhand_actions.append(hand_action)
        else:
          farmhand_actions.append(list(hand_action))
      
      action_dict['hands'] = farmhand_actions
    
    market_actions = []
    for market_action in market_action_idx:
      market_action = market_action_space[int(market_action)]
      if market_action != "PASS":
        if type(market_action) == str:
          market_actions.append([market_action])
        else:
          market_actions.append(list(market_action))
    
    action_dict['market'] = market_actions
    
    return action_dict

def process_obs(obs):
    # Add function to split up obs into grid and non_tiles
    global internal_obs
    grid = setup_tile_obs(obs)
    non_tiles = setup_non_tile_observation(obs)
    internal_obs = deepcopy(obs)
    update_internal_obs(internal_obs)

    return grid, non_tiles

def create_masks(obs):
    # First farm_mask is for farmer only

    farm_mask = list(map(lambda x: invalid_action(obs, x), farmer_action_space.values()))
    market_mask = list(map(lambda x: invalid_action(obs, x), market_action_space.values()))

    return farm_mask, market_mask

def agent(obs):
    global lstm_hidden, last_start, internal_obs
    
    grid, non_tiles = process_obs(obs)
    farm_mask, market_mask = create_masks(obs)
    grid = torch.from_numpy(np.array(grid)).to(dtype=torch.float32, device=device).permute(2, 0, 1)
    non_tiles = torch.tensor(non_tiles, dtype=torch.float32, device=device).unsqueeze(0)
    ep_starts = torch.tensor(last_start, dtype=torch.float32, device=device)

    farm_actions, market_actions, _, _, _, _, lstm_hidden = model(grid, non_tiles, lstm_hidden, ep_starts, farm_mask, market_mask, internal_obs)

    last_start = False
    action = action_idx_to_string(farm_actions, market_actions)

    return action
    

if __name__ == "__main__":
    env = make("kaggriculture", configuration={"episodeSteps": 720}, debug=True)
    env.run([agent, "random"])
    with open("replay.json", "w") as f:
        json.dump(env.toJSON(), f)