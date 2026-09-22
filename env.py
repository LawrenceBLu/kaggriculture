from kaggle_environments.envs.kaggriculture.kaggriculture import MARKET_PARAMS
import numpy as np

class env():
    def __init__(self, env):
        self.env = env
        self.past_obs = None
        self.past_action = None
        self.reset()
        self.max_money = 0
        
    def reset(self):
        obs, _ = self.env.reset()
        self.past_obs = obs
        
    def step(self, action):
        reward = 0
        self.past_action = action['market']
        if self.steps == 720:
            obs, _ = self.reset()
            self.steps = 0
            self.past_obs = obs
            self.past_market_action = None
            return obs, False, reward
        obs, _ = self.env.step(action)
        reward = self.get_reward(obs)
        self.steps += 1
        if self.steps == 719:
            return obs, True, reward
        return obs, False, reward

    def get_reward(self, obs):
        reward = 0
        player_num = obs["player"]
        
        self.max_money = max(self.max_money, obs['farm'][player_num]['money'])

        if self.steps == 719:
            # rewards for win
            reward += 5
            # rewards for total money/25k
            reward += self.max_money/25000
            
        else:
            # rewards for first farm unlock
            if self.past_obs['farm'][player_num]['tiles'][0][9] == 'LOCKED' and obs['farm'][player_num]['tiles'][0][9] == None:
                reward += 3
            # rewards for second farm unlock
            if self.past_obs['farm'][player_num]['tiles'][9][0] == 'LOCKED' and obs['farm'][player_num]['tiles'][9][0] == None:
                reward += 3
            # penalizes for third farm unlock
            if self.past_obs['farm'][player_num]['tiles'][9][9] == 'LOCKED' and obs['farm'][player_num]['tiles'][9][9] == None:
                reward -= 5
            # rewards for buying low
            if self.past_action in type(list, tuple) and "BUY" in self.past_action[0]:
                if self.past_obs['market']['inventory'][self.past_action[1]] > 10000:
                    reward += (self.past_obs['market']['inventory'][self.past_action[1]]-10000 / MARKET_PARAMS[self.past_action[1]]["T"])* 0.02 * self.past_action[2] 
            # rewards for selling high
            if self.past_action in type(list, tuple) and self.past_action[0] == "SELL":
                if self.past_obs['market']['inventory'][self.past_action[1]] < 10000:
                    reward += (10000-self.past_obs['market']['inventory'][self.past_action[1]] / MARKET_PARAMS[self.past_action[1]]["T"])* 0.02 * self.past_action[2] 
            tiles = [tile for column in obs['farm'][player_num]['tiles'] for tile in column]
            # penalizes for weed on a tile, disincentivizes underwatering
            if "WEED" in tiles:
                reward -= 0.5
            # rewards animals near the center tile
            for y in range(len(obs['farm'][player_num]['tiles'])):
                for x in range(len(obs['farm'][player_num]['tiles'][y])):
                    tile = obs['farm'][player_num]['tiles'][y][x]
                    if isinstance(tile, dict):
                        if tile.get("animal") is not None and tile.get("kind") in ("COOP", "PASTURE"):
                            if np.linalg.norm(np.array([y, x]) - (4,4)) <= 3:
                                reward += (1/(1+np.pow(1.5, np.linalg.norm(np.array([y, x]) - (4,4))))) * 0.25
        return reward