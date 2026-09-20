import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchrl.modules import MaskedCategorical
from kaggyculture import update_mask, invalid_action
from action_space import market_action_space, farmhand_action_space

#heehee

class CNN(nn.Module):
    def __init__(self, input_channels, output_len):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 20, 3, padding=1)
        self.conv2 = nn.Conv2d(20, 40, 3, padding=1)
        self.conv3 = nn.Conv2d(40, 60, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(60 * 2 * 2, output_len)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.relu(self.conv3(x))
        x = x.view(-1, 60 * 2 * 2)
        x = self.fc1(x)
        return x

class ActorCritic(nn.Module):
    def __init__(self, grid_size, cnn_dim, non_tile_len, linear_dim, lstm_dim, num_layers, farm_action_dim, num_hands, market_action_dim, num_orders):
        super().__init__()
        self.cnn = CNN(grid_size, cnn_dim)
        self.pre_lstm = nn.Linear(cnn_dim+non_tile_len, linear_dim)
        self.lstm = nn.LSTM(linear_dim, lstm_dim, num_layers, batch_first=True)

        self.num_hands = num_hands
        self.num_orders = num_orders
        self.actor_farm_head = nn.Linear(lstm_dim, farm_action_dim*(num_hands+1))
        self.actor_market_head = nn.Linear(lstm_dim, market_action_dim*num_orders)
        self.critic_head = nn.Linear(lstm_dim, 1)
        self.linear_dim = linear_dim

    def _process_sequence(self, grid, non_tiles, hidden, ep_starts):
        # FINISH handling lstm state resets
        grid_state = self.cnn(grid)
        state = torch.cat([torch.squeeze(grid_state), non_tiles], dim=-1)
        x = self.pre_lstm(state)

        if torch.all(ep_starts == 0.0):
            lstm_out, hidden = self.lstm(x.unsqueeze(0), hidden)
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

    def compute_actions(self, farm_logits, market_logits, farm_mask, market_mask, internal_obs):
        farmer_vec = farm_logits.reshape(self.num_hands+1, -1)
        market_vec = market_logits.reshape(self.num_orders, -1)

        all_farm_actions = farmhand_action_space()
        all_market_actions = market_action_space()

        farm_actions = []
        market_actions = []

        farm_masks = []
        market_masks = []
        log_probs = []
        

        for farmer_index, farmer_logits in enumerate(farmer_vec):
            farm_masks.append(farm_mask)
            farm_dist = MaskedCategorical(logits=farmer_logits, mask=torch.tensor(farm_mask))
            farm_action = farm_dist.sample()
            if farmer_index < (len(farmer_vec) - 1):
                farm_mask = list(map(lambda x: invalid_action(internal_obs, x, farmer_index), all_farm_actions.values()))
                farm_action_str = all_farm_actions[int(farm_action.item())]
                farm_mask = update_mask(farm_mask, internal_obs, farm_action_str, all_farm_actions, index=farmer_index, is_farm=True)

            log_prob = farm_dist.log_prob(farm_action)
            farm_actions.append(farm_action)
            log_probs.append(log_prob)

        for market_index, market_logits in enumerate(market_vec):
            market_masks.append(market_mask)
            market_dist = MaskedCategorical(logits=market_logits, mask=torch.tensor(market_mask))
            market_action = market_dist.sample()
            if market_index < (len(market_vec) - 1):
                market_mask = list(map(lambda x: invalid_action(internal_obs, x), all_market_actions.values()))
                market_action_str = all_market_actions[int(market_action.item())]
                market_mask = update_mask(market_mask, internal_obs, (market_action, market_action_str), all_market_actions, index=market_index)

            log_prob = market_dist.log_prob(market_action)
            market_actions.append(market_action)
            log_probs.append(log_prob)

        log_p = torch.stack(log_probs, dim=-1).sum(dim=-1)
        return farm_actions, market_actions, log_p, farm_masks, market_masks

    def compute_entropy(self, farm_logits, market_logits, farm_actions, market_actions, farm_masks, market_masks):
        hands_vec = farm_logits.reshape(self.num_hands+1, -1)
        market_vec = market_logits.reshape(self.num_orders, -1)

        log_probs = []
        entropies = []
        log_num_valid = []

        for hands_i in range(len(hands_vec)):
            farm_dist = MaskedCategorical(logits=farm_logits[hands_i], mask=torch.tensor(farm_masks[hands_i]))
            farm_entropy = farm_dist.entropy()
            log_prob = farm_dist.log_prob(farm_actions[hands_i])

            num_valid = farm_masks[hands_i].sum(-1).clamp(min=1)
            log_num_valid.append(torch.log(num_valid.float()).clamp(min=1e-8))

            entropies.append(farm_entropy)
            log_probs.append(log_prob)

        for market_i in range(len(market_vec)):
            market_dist = MaskedCategorical(logits=market_logits[market_i], mask=torch.tensor(market_masks[market_i]))
            market_entropy = market_dist.entropy()
            log_prob = market_dist.log_prob(market_actions[market_i])

            num_valid = market_masks[hands_i].sum(-1).clamp(min=1)
            log_num_valid.append(torch.log(num_valid.float()).clamp(min=1e-8))

            entropies.append(market_entropy)
            log_probs.append(log_prob)

        log_p = torch.stack(log_probs, dim=-1).sum(dim=-1)
        normed = torch.stack(entropies, dim=-1)/torch.stack(log_num_valid, dim=-1)
        entropy = normed.mean(dim=-1).mean()

        return log_p, entropy

    def forward(self, grid, non_tiles, hidden, ep_starts, farm_mask, market_mask, internal_obs):
        # No detach because ran under no_grad()
        lstm_out, hidden = self._process_sequence(grid.unsqueeze(0), non_tiles, hidden, ep_starts)

        farm_logits = self.actor_farm_head(lstm_out)
        market_logits = self.actor_market_head(lstm_out)
        farm_actions, market_actions, log_prob, farm_masks, market_masks = self.compute_actions(farm_logits, market_logits, farm_mask, market_mask, internal_obs)
        # dist = Categorical(logits = logits)
        # action = dist.sample()
        # log_prob = dist.log_prob(action)

        state_value = self.critic_head(lstm_out)

        return farm_actions, market_actions, farm_masks, market_masks, log_prob, state_value, hidden

    def evaluate(self, grid, non_tiles, hiddens, farm_actions, market_actions, farm_masks, market_masks, ep_starts):
        hidden = hiddens[0]
        lstm_out, _ = self._process_sequence(grid, non_tiles, hidden, ep_starts)
        
        farm_logits = self.actor_farm_head(lstm_out)
        market_logits = self.actor_market_head(lstm_out)
        log_probs, entropy = self.compute_entropy(farm_logits, market_logits, farm_actions, market_actions, farm_masks, market_masks)

        # dist = Categorical(logits = logits)
        # log_probs = dist.log_prob(actions)
        # entropy = dist.entropy()

        state_values = self.critic_head(lstm_out).squeeze(-1)

        return log_probs, entropy, state_values

    def predict_value(self, grid, non_tiles, hidden, ep_starts):
        lstm_out, _ = self._process_sequence(grid.unsqueeze(0), non_tiles, hidden, ep_starts)
        state_value = self.critic_head(lstm_out)

        return state_value
