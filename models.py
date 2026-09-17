import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
import torch.nn.functional as F

#heehee

class CNN(nn.Module):
    def __init__(self, input_channels, output_len):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 20, 3)
        self.conv2 = nn.Conv2d(20, 40, 3)
        self.conv3 = nn.Conv2d(40, 60, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(60 * 10 * 10, output_len)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 60 * 10 * 10)
        x = self.fc1(x)

class ActorCritic(nn.Module):
    def __init__(self, grid_size, cnn_dim, non_tile_len, linear_dim, lstm_dim, num_layers, farm_action_dim, market_action_dim):
        super().__init__()
        self.cnn = CNN(grid_size, cnn_dim)
        self.pre_lstm = nn.Linear(cnn_dim+non_tile_len, linear_dim)
        self.lstm = nn.LSTM(linear_dim, lstm_dim, num_layers, batch_first=True)
        
        self.actor_farm_head = nn.Linear(lstm_dim, farm_action_dim)
        self.actor_market_head = nn.Linear(lstm_dim, market_action_dim)
        self.critic_head = nn.Linear(lstm_dim, 1)
        self.linear_dim = linear_dim

    def _process_sequence(self, grid, non_tiles, hidden, ep_starts):
        # FINISH handling lstm state resets
        grid_state = self.cnn(grid)
        state = torch.cat([grid_state, non_tiles], dim=-1)
        x = self.pre_lstm(state)

        if torch.all(ep_starts == 0.0):
            lstm_out, hidden = self.lstm(x, hidden)
        else:
            lstm_out = []
            n_seq = hidden[0].shape[1]
            ep_starts = ep_starts.reshape((n_seq, -1)).swapaxes(0, 1)
            features = x.reshape((n_seq, -1, self.linear_dim)).swapaxes(0, 1)
            for feats, start in zip(features, ep_starts):
                out, hidden = self.lstm(feats.unsqueeze(dim=1),
                    (
                        (1.0 - start).view(1, n_seq, 1) * hidden[0],
                        (1.0 - start).view(1, n_seq, 1) * hidden[1]
                    ),
                )
                lstm_out += [out]

            lstm_out = torch.flatten(torch.cat(lstm_out, dim=1), start_dim=0, end_dim=1)
        return lstm_out, hidden

    def compute_actions(self, farm_logits, market_logits, num_hands, num_orders, internal_obs):
        hands_vec = farm_logits.reshape(num_hands+1, -1)
        market_vec = market_logits.reshape(num_orders, -1)

        farm_actions = []
        market_actions = []
        log_probs = []

        for hands_logits in hands_vec:
            create_mask(internal_obs)
            farm_dist = MaskedCategorical(logits=hands_logits, mask)
            farm_action = farm_dist.sample()
            log_prob = farm_dist.log_prob(farm_action)
            farm_actions.append(farm_action)
            log_probs.append(log_prob)

        for market_logits in market_vec:
            market_dist = MaskedCategorical(logits=market_logits, mask)
            market_action = market_dist.sample()
            log_prob = market_dist.log_prob(market_action)
            market_actions.append(market_action)
            log_probs.append(log_prob)

        return farm_actions, market_action, log_probs

    def compute_entropy(self, farm_logits, market_logits, actions):
        hands_vec = farm_logits.reshape(num_hands+1, -1)
        market_vec = market_logits.reshape(num_orders, -1)

        farm_actions = []
        market_actions = []
        log_probs = []

        for hands_logits in hands_vec:
            farm_dist = MaskedCategorical(logits=hands_logits, mask)
            farm_action = farm_dist.sample()
            log_prob = farm_dist.log_prob(farm_action)
            farm_actions.append(farm_action)
            log_probs.append(log_prob)

        for market_logits in market_vec:
            market_dist = MaskedCategorical(logits=market_logits, mask)
            market_action = market_dist.sample()
            log_prob = market_dist.log_prob(market_action)
            market_actions.append(market_action)
            log_probs.append(log_prob)
    

    def forward(self, grid, non_tiles, hidden, ep_starts, num_hands, internal_obs):
        # No detach because ran under no_grad()
        lstm_out, hidden = self._process_sequence(grid, non_tiles, hidden, ep_starts, internal_obs)

        farm_logits = self.actor_farm_head(lstm_out)
        market_logits = self.actor_market_head(lstm_out)
        farm_actions, market_actions, log_prob = self.handle_actions(farm_logits, market_logits, num_hands, internal_obs)
        # dist = Categorical(logits = logits)
        # action = dist.sample()
        # log_prob = dist.log_prob(action)

        state_value = self.critic_head(lstm_out)

        return farm_actions, market_actions, log_prob, state_value, hidden

    def evaluate(self, grid, non_tiles, actions, hiddens, ep_starts):
        hidden = hiddens[0]
        lstm_out, _ = self._process_sequence(grid, non_tiles, hidden, ep_starts)
        
        farm_logits = self.actor_farm_head(lstm_out)
        market_logits = self.actor_market_head(lstm_out)
        dist = Categorical(logits = logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        state_values = self.critic_head(lstm_out).squeeze(-1)

        return log_probs, entropy, state_values

    def predict_value(self, grid, non_tiles, hidden, ep_starts):
        lstm_out, _ = self._process_sequence(grid, non_tiles, hidden, ep_starts)
        state_value = self.critic_head(lstm_out)

        return state_value
