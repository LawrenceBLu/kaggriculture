import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
import torch.nn.functional as F

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
    def __init__(self, grid_size, cnn_dim, non_tile_len, linear_dim, lstm_dim, num_layers, action_dim):
        super().__init__()
        self.cnn = CNN(grid_size, cnn_dim)
        self.pre_lstm = nn.Linear(cnn_dim+non_tile_len, linear_dim)
        self.lstm = nn.LSTM(linear_dim, lstm_dim, num_layers, batch_first=True)
        
        self.actor_head = nn.Linear(lstm_dim, action_dim)
        self.critic_head = nn.Linear(lstm_dim, 1)

    def forward(self, grid, non_tiles, hidden):
        grid_state = self.cnn(grid)
        state = torch.cat([grid_state, non_tiles], dim=-1)
        x = self.pre_lstm(state)
        lstm_out, hidden = self.lstm(x, hidden)
        logits = self.actor_head(lstm_out)
        cat = Categorical(logits = logits)
        action = cat.sample()
        log_prob = cat.log_prob(action)

        state_value = self.critic_head(lstm_out)

        return action.detach(), log_prob.detach(), state_value.detach(), (hidden[0].detach(), hidden[1].detach())

    def evaluate(self, states, actions, hidden, last_episode_dones):
        # No detach because this is used for optimization
        # FINISH handling lstm state resets
        # FINISH

        grid_state = self.cnn(grid)
        state = torch.cat([grid_state, non_tiles], dim=-1)
        x = self.pre_lstm(state)
        lstm_out = self.lstm(x, hidden)
        logits = self.actor_head(lstm_out)
        cat = Categorical(logits = logits)
        log_probs = cat.log_prob(actions)
        entropy = cat.entropy()

        state_values = self.critic_head(lstm_out).squeeze(-1)

        return log_probs, entropy, state_values