import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchrl.modules import MaskedCategorical
from kaggyculture import update_mask, invalid_action
from action_space import market_action_space, farmhand_action_space

class CNN(nn.Module):
    def __init__(self, input_channels, output_len):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, 3, padding=1)
        # self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(128 * 10 * 10, output_len)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = x.view(-1, 128 * 10 * 10)
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
        if grid.dim() == 4:
            rollout = True

            grid = grid.unsqueeze(1)

            if non_tiles.dim() == 2:
                non_tiles = non_tiles.unsqueeze(1)

            if ep_starts.dim() == 1:
                ep_starts = ep_starts.unsqueeze(1)

        elif grid.dim() == 5:
            rollout = False

        B, T, C, H, W = grid.shape

        if non_tiles.dim() == 2:
            non_tiles = non_tiles.reshape(B, T, -1)

        ep_starts = ep_starts.reshape(B, T)

        grid_flat = torch.flatten(grid, start_dim=0, end_dim=1)

        grid_state = self.cnn(grid_flat)
        grid_state = grid_state.reshape(B, T, -1)
        non_tile_state = self.non_tile_feat(non_tiles)
        state = torch.cat([grid_state, non_tile_state], dim=-1)
        x = self.relu(self.pre_lstm(state))
        h, c = hidden

        if h.shape[1] != B:
            raise ValueError(
                f"Hidden-state batch dimension ({h.shape[1]}) "
                f"does not match sequence batch size ({B}). "
                f"Hidden shape: {h.shape}, grid shape: {grid.shape}"
            )

        if not torch.any(ep_starts):
                lstm_out, hidden = self.lstm(x, hidden)
        else:
            outputs = []

            for t in range(T):
                keep_mask = (1.0 - ep_starts[:, t].float()).view(1, B, 1)

                keep_mask = keep_mask.to(device=h.device,dtype=h.dtype)

                h = h * keep_mask
                c = c * keep_mask

                step_x = x[:, t:t + 1, :]

                step_out, (h, c) = self.lstm(step_x, (h, c))

                outputs.append(step_out)

            lstm_out = torch.cat(outputs, dim=1)
            hidden = (h, c)

        if rollout:
            lstm_out = lstm_out.squeeze(1)

        else:
            lstm_out = torch.flatten(lstm_out, start_dim=0, end_dim=1)

        return lstm_out, hidden


    def compute_actions(self, farm_logits, market_logits, farm_masks, market_masks, obs):
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
            all_farm_masks.append(farm_masks)
            farm_dists = MaskedCategorical(logits=farmer_logits, mask=torch.tensor(farm_masks, device=self.device))
            farm_actions = farm_dists.sample()
            if farmer_index < (farmer_vec.shape[0] - 1):
                farm_masks = [list(map(lambda x: invalid_action(internal_obs, x, farmer_index), farm_action_space.values())) for internal_obs in obs]
                farm_action_strs = [farm_action_space[int(farm_action.item())] for farm_action in farm_actions]
                farm_masks = [update_mask(farm_mask, internal_obs, farm_action_str, farm_action_space, index=farmer_index, is_farm=True) for farm_mask, internal_obs, farm_action_str in zip(farm_masks, obs, farm_action_strs)]

            log_probs = farm_dists.log_prob(farm_actions)
            all_farm_actions.append(farm_actions)
            all_log_probs.append(log_probs)

        persistent_market_masks = np.array(market_masks)
        for market_index, market_logits in enumerate(market_vec):
            all_market_masks.append(persistent_market_masks)
            market_dists = MaskedCategorical(logits=market_logits, mask=torch.from_numpy(persistent_market_masks).to(device=self.device))
            market_actions = market_dists.sample()
            if market_index < (market_vec.shape[0] - 1):
                market_masks = [list(map(lambda x: invalid_action(internal_obs, x), market_space.values())) for internal_obs in obs]
                market_action_strs = [market_space[int(market_action.item())] for market_action in market_actions]
                market_masks = [update_mask(market_mask, internal_obs, (market_action.item(), market_action_str), market_space, index=market_index) for market_mask, internal_obs, market_action, market_action_str in zip(market_masks, obs, market_actions, market_action_strs)]
                persistent_market_masks = persistent_market_masks * market_masks

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

        farm_actions = torch.flatten(farm_actions, 0, 1)
        market_actions = torch.flatten(market_actions, 0, 1)
        farm_masks = torch.flatten(farm_masks, 0, 1)
        market_masks = torch.flatten(market_masks, 0, 1)

        log_probs = []
        entropies = []
        log_num_valid = []

        farm_dist = MaskedCategorical(logits=hands_vec, mask=farm_masks)
        farm_entropy = farm_dist.entropy()
        farm_log_prob = farm_dist.log_prob(farm_actions)

        farm_num_valid = farm_masks.sum(-1).clamp(min=1)
        farm_log_valid = torch.log(farm_num_valid.float())
        safe_divisor = torch.where(farm_num_valid > 1, farm_log_valid, torch.ones_like(farm_log_valid))

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

        return log_p, entropy

    def forward(self, grid, non_tiles, hidden, ep_starts, farm_mask, market_mask, internal_obs):
        # No detach because ran under no_grad()
        lstm_out, hidden = self._process_sequence(grid, non_tiles, hidden, ep_starts)

        farm_logits = self.actor_farm_head(lstm_out)
        market_logits = self.actor_market_head(lstm_out)
        farm_actions, market_actions, log_prob, farm_masks, market_masks = self.compute_actions(farm_logits, market_logits, farm_mask, market_mask, internal_obs)

        state_value = self.critic_head(lstm_out)

        return farm_actions, market_actions, farm_masks, market_masks, log_prob, state_value, hidden

    def evaluate(self, grid, non_tiles, hiddens, farm_actions, market_actions, farm_masks, market_masks, ep_starts):
        lstm_out, _ = self._process_sequence(grid, non_tiles, hiddens, ep_starts)
        
        farm_logits = self.actor_farm_head(lstm_out)
        market_logits = self.actor_market_head(lstm_out)
        
        log_probs, entropy = self.compute_entropy(farm_logits, market_logits, farm_actions, market_actions, farm_masks, market_masks)

        state_values = self.critic_head(lstm_out).squeeze(-1)

        return log_probs, entropy, state_values

    def predict_value(self, grid, non_tiles, hidden, ep_starts):
        lstm_out, _ = self._process_sequence(grid, non_tiles, hidden, ep_starts)
        state_value = self.critic_head(lstm_out)

        return state_value
