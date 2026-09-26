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
        self.conv1 = nn.Conv2d(input_channels, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        # self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(64 * 10 * 10, output_len)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = x.view(-1, 64 * 10 * 10)
        x = self.relu(self.fc(x))
        return x

class ActorCritic(nn.Module):
    def __init__(self, grid_size, cnn_dim, non_tile_len, non_tile_dim, linear_dim, lstm_dim, num_layers, farm_action_dim, num_hands, market_action_dim, num_orders, device):
        super().__init__()
        self.cnn = CNN(grid_size, cnn_dim)
        self.non_tile_feat = nn.Sequential(
            nn.Linear(non_tile_len, 128),
            nn.ReLU(),
            nn.Linear(128, non_tile_dim),
            nn.ReLU()
        )
        self.pre_lstm = nn.Linear(cnn_dim+non_tile_dim, linear_dim)
        self.lstm = nn.LSTM(linear_dim, lstm_dim, num_layers, batch_first=True)
        self.relu = nn.ReLU()

        self.num_hands = num_hands
        self.num_orders = num_orders
        self.actor_farm_head = nn.Linear(lstm_dim, farm_action_dim*(num_hands+1))
        self.actor_market_head = nn.Linear(lstm_dim, market_action_dim*num_orders)
        self.critic_head = nn.Linear(lstm_dim, 1)
        self.linear_dim = linear_dim

        self.device = device

    def _process_sequence(self, grid, non_tiles, hidden, ep_starts):
        # FINISH handling lstm state resets
        grid_state = self.cnn(grid)
        non_tile_state = self.non_tile_feat(non_tiles)
        state = torch.cat([grid_state, non_tile_state], dim=-1)
        x = self.relu(self.pre_lstm(state))


        if torch.all(ep_starts == 0.0):
            lstm_out, hidden = self.lstm(x.unsqueeze(1), hidden)
            lstm_out = lstm_out.squeeze(1)
        else:
            lstm_out = []
            n_seq = hidden[0].shape[1]
            ep_starts = ep_starts.reshape((n_seq, -1)).swapaxes(0, 1)
            features = x.reshape((n_seq, -1, self.linear_dim)).swapaxes(0, 1)
            for feats, start in zip(features, ep_starts):
                # print("start:", start.shape, "n_seq:", n_seq, "hidden[0]:", hidden[0].shape)
                out, hidden = self.lstm(feats.unsqueeze(dim=1),
                    (
                        (1.0 - start).view(1, n_seq, 1) * hidden[0],
                        (1.0 - start).view(1, n_seq, 1) * hidden[1]
                    ),
                )
                lstm_out += [out]

            lstm_out = torch.flatten(torch.cat(lstm_out, dim=1), start_dim=0, end_dim=1)
        return lstm_out, hidden

    def compute_actions(self, farm_logits, market_logits, farm_mask, market_mask, obs):
        farmer_vec = farm_logits.reshape(farm_logits.size(0), self.num_hands+1, -1).permute(1,0,2)
        market_vec = market_logits.reshape(market_logits.size(0), self.num_orders, -1).permute(1,0,2)

        farm_action_space = farmhand_action_space()
        market_space = market_action_space()

        all_farm_actions = []
        all_market_actions = []

        all_farm_masks = []
        all_market_masks = []
        all_log_probs = []
        for farmer_index, farmer_logits in enumerate(farmer_vec):
            all_farm_masks.append(farm_mask)
            farm_dists = MaskedCategorical(logits=farmer_logits, mask=torch.tensor(farm_mask, device=self.device))
            farm_actions = farm_dists.sample()
            if farmer_index < (farmer_vec.shape[0] - 1):
                farm_masks = [list(map(lambda x: invalid_action(internal_obs, x, farmer_index), farm_action_space.values())) for internal_obs in obs]
                farm_action_strs = [farm_action_space[int(farm_action.item())] for farm_action in farm_actions]
                farm_masks = [update_mask(farm_mask, internal_obs, farm_action_str, farm_action_space, index=farmer_index, is_farm=True) for farm_mask, internal_obs, farm_action_str in zip(farm_masks, obs, farm_action_strs)]

            log_probs = farm_dists.log_prob(farm_actions)
            all_farm_actions.append(farm_actions)
            all_log_probs.append(log_probs)

        for market_index, market_logits in enumerate(market_vec):
            all_market_masks.append(market_mask)
            market_dists = MaskedCategorical(logits=market_logits, mask=torch.tensor(market_mask, device=self.device))
            market_actions = market_dists.sample()
            if market_index < (market_vec.shape[0] - 1):
                market_masks = [list(map(lambda x: invalid_action(internal_obs, x), market_space.values())) for internal_obs in obs]
                market_action_strs = [market_space[int(market_action.item())] for market_action in market_actions]
                market_masks = [update_mask(market_mask, internal_obs, (market_index, market_action_str), market_space, index=market_index) for market_mask, internal_obs, market_action_str in zip(market_masks, obs, market_action_strs)]

            log_probs = market_dists.log_prob(market_actions)
            all_market_actions.append(market_actions)
            all_log_probs.append(log_probs)

        all_log_p = torch.stack(all_log_probs, dim=-1).sum(dim=-1)
        all_farm_actions = torch.stack(all_farm_actions, dim=-1)
        all_market_actions = torch.stack(all_market_actions, dim=-1)
        all_farm_masks = [list(mask) for mask in zip(*all_farm_masks)]
        all_market_masks = [list(mask) for mask in zip(*all_market_masks)]
        
        return all_farm_actions, all_market_actions, all_log_p, all_farm_masks, all_market_masks

    def compute_entropy(self, farm_logits, market_logits, farm_actions, market_actions, farm_masks, market_masks):
        hands_vec = farm_logits.reshape(farm_logits.size(0), self.num_hands+1, -1)
        market_vec = market_logits.reshape(market_logits.size(0), self.num_orders, -1)

        log_probs = []
        entropies = []
        log_num_valid = []

        # print("hands_vec has nan:", torch.isnan(hands_vec).any().item())
        # print("hands_vec has inf:", torch.isinf(hands_vec).any().item())
        # print("hands_vec min/max:", hands_vec.min().item(), hands_vec.max().item())

        
        farm_dist = MaskedCategorical(logits=hands_vec, mask=farm_masks)
        farm_entropy = farm_dist.entropy()
        farm_log_prob = farm_dist.log_prob(farm_actions)

        farm_num_valid = farm_masks.sum(-1).clamp(min=1)
        farm_log_valid = torch.log(farm_num_valid.float())
        safe_divisor = torch.where(farm_num_valid > 1, farm_log_valid, torch.ones_like(farm_log_valid))

        # log_num_valid.append(farm_log_valid.clamp(min=1e-8))
        log_num_valid.append(safe_divisor)

        entropies.append(farm_entropy)
        log_probs.append(farm_log_prob)

        market_dist = MaskedCategorical(logits=market_vec, mask=market_masks)
        market_entropy = market_dist.entropy()
        market_log_prob = market_dist.log_prob(market_actions)

        market_num_valid = market_masks.sum(-1).clamp(min=1)
        market_log_valid = torch.log(market_num_valid.float())

        safe_divisor = torch.where(market_num_valid > 1, market_log_valid, torch.ones_like(market_log_valid))
        log_num_valid.append(safe_divisor)

        entropies.append(market_entropy)
        log_probs.append(market_log_prob)

        log_p = torch.cat(log_probs, dim=-1).sum(dim=-1)
        
        normed = torch.cat(entropies, dim=-1)/torch.cat(log_num_valid, dim=-1)
        entropy = normed.mean(dim=-1).mean()

        # print("farm_num_valid min:", farm_num_valid.min().item())
        # print("log_num_valid min:", torch.cat(log_num_valid, dim=-1).min().item())

        return log_p, entropy

    def forward(self, grid, non_tiles, hidden, ep_starts, farm_mask, market_mask, internal_obs):
        # No detach because ran under no_grad()
        lstm_out, hidden = self._process_sequence(grid, non_tiles, hidden, ep_starts)

        farm_logits = self.actor_farm_head(lstm_out)
        market_logits = self.actor_market_head(lstm_out)
        farm_actions, market_actions, log_prob, farm_masks, market_masks = self.compute_actions(farm_logits, market_logits, farm_mask, market_mask, internal_obs)
        # dist = Categorical(logits = logits)
        # action = dist.sample()
        # log_prob = dist.log_prob(action)

        state_value = self.critic_head(lstm_out)

        return farm_actions, market_actions, farm_masks, market_masks, log_prob, state_value, hidden

    def evaluate(self, grid, non_tiles, hiddens, farm_actions, market_actions, farm_masks, market_masks, ep_starts):
        # for name, p in self.cnn.named_parameters():
        #     if not torch.isfinite(p).all():
        #         print(f"{name} is already non-finite BEFORE this forward pass")
        lstm_out, _ = self._process_sequence(grid, non_tiles, hiddens, ep_starts)
        # assert torch.isfinite(lstm_out).all(), "lstm_out has nan/inf"
        
        farm_logits = self.actor_farm_head(lstm_out)
        market_logits = self.actor_market_head(lstm_out)
        log_probs, entropy = self.compute_entropy(farm_logits, market_logits, farm_actions, market_actions, farm_masks, market_masks)

        # dist = Categorical(logits = logits)
        # log_probs = dist.log_prob(actions)
        # entropy = dist.entropy()

        state_values = self.critic_head(lstm_out).squeeze(-1)

        return log_probs, entropy, state_values

    def predict_value(self, grid, non_tiles, hidden, ep_starts):
        lstm_out, _ = self._process_sequence(grid, non_tiles, hidden, ep_starts)
        state_value = self.critic_head(lstm_out)

        return state_value
