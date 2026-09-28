from kaggle_environments.envs.kaggriculture.kaggriculture import MARKET_PARAMS
import numpy as np
from kaggyculture import load_dict_of_lists
import gymnasium as gym
from kaggle_environments import make
from gymnasium import spaces
from kaggyculture import invalid_action, setup_non_tile_observation, setup_tile_obs, update_internal_obs
from global_vars import MAX_FARMHAND_HIRES


class KaggleEnvWrapper(gym.Env):
    def __init__(self, env_name, num_channels, non_tile_len, opponent_agent="random"):
        self.env = make(env_name, debug=False)
        self.trainer = self.env.train([None, opponent_agent])
        self.observation_space = spaces.Dict({
            "grid": spaces.Box(low=0, high=1e6, shape=(num_channels, 10, 10), dtype=np.float32),
            "non_tiles": spaces.Box(low=0, high=1e6, shape=(non_tile_len,), dtype=np.float32)
        })
        self.action_space = spaces.Dict({
            "farmer": spaces.Discrete(1),
            "hands": spaces.Discrete(5),
            "market": spaces.Discrete(3)
        })

        self.past_obs = None
        self.past_action = None
        self.done = False
        self.dict_of_lists = load_dict_of_lists()
        self.max_money = 0
        self.curr_step = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs = self.trainer.reset()
        self.past_obs = obs
        state = self._extract_obs(obs)
        self.done = False
        self.curr_step = 0
        return state, {"raw_obs": obs}

    def step(self, action):
        # Build the joint action list: your agent's action + opponents' actions
        reward = 0
        self.past_action = action
        if self.done == True:
            obs, info = self.reset()
            state = self._extract_obs(obs)
            self.past_market_action = None
            return state, 0, False, False, info
        if self.curr_step % 24 == 0:
            for _ in range(5):
                action["market"].append(["HIRE"])
        obs, _, self.done, _ = self.trainer.step(action)
        self.curr_step += 1
        state = self._extract_obs(obs)
        reward = self.get_reward(obs)
        self.past_obs = obs
        if self.done == True:
            return state, reward, True, False, {"raw_obs": obs}
        return state, reward, False, False, {"raw_obs": obs}

    def _extract_obs(self, obs):
        grid = setup_tile_obs(obs)
        non_tiles = setup_non_tile_observation(obs)
        observation = {"grid": grid, "non_tiles": non_tiles}
        return observation

    def get_internal_obs(self):
        return self.past_obs

    def get_reward(self, obs):
        reward = 0
        player_num = obs["player"]
        # print(self.past_obs)
        self.max_money = max(self.max_money, obs['farms'][player_num]['money'])
        if self.done == True:
            # rewards for win
            if obs['farms'][player_num]['money'] > obs['farms'][player_num^1]['money']:
                reward += 5
            # rewards for total money/25k
            reward += self.max_money/10000
            
        else:
            # rewards for first farms unlock
            if self.past_obs['farms'][player_num]['tiles'][0][9] == 'LOCKED' and obs['farms'][player_num]['tiles'][0][9] == None:
                # print("FIRST QUADRANT HAS BEEN UNLOCKED")
                reward += 3
            # rewards for second farms unlock
            if self.past_obs['farms'][player_num]['tiles'][9][0] == 'LOCKED' and obs['farms'][player_num]['tiles'][9][0] == None:
                reward += 3
                # print("SECOND QUADRANT HAS BEEN UNLOCKED")
            # # penalizes for third farms unlock
            # if self.past_obs['farms'][player_num]['tiles'][9][9] == 'LOCKED' and obs['farms'][player_num]['tiles'][9][9] == None:
            #     reward -= 5
            for action in self.past_action['market']:
                if type(action) == list and len(action) > 1:
                    if action[0] == "SELL":
                        if self.past_obs['market']['inventory'][action[1]] < 10000:
                            reward += (10000-self.past_obs['market']['inventory'][action[1]] / MARKET_PARAMS[action[1]]["T"])* 0.02 * action[2] 
                # rewards for buying low
            """                     if action[0] == "BUY":
                                    if self.past_obs['market']['inventory'][self.past_action[1]] > 10000:
                                        reward += (self.past_obs['market']['inventory'][action[1]]-10000 / MARKET_PARAMS[self.past_action[1]]["T"])* 0.02 * self.past_action[2] 
                """                    # rewards for selling high
            tiles = [tile for column in obs['farms'][player_num]['tiles'] for tile in column]
            # penalizes for weed on a tile, disincentivizes underwatering
            
            # if
            farmer_actions = self.past_action['farmer']

            farmhand_actions = self.past_action['hands']
            if len(farmer_actions) == 1: combined_farmer_farmhand_actions = farmer_actions + (farmhand_actions)
            else: combined_farmer_farmhand_actions = [farmer_actions] + (farmhand_actions)
            pos_farmer_farmhand = []
            pos_farmer_farmhand.append(self.past_obs['farms'][player_num]['farmer'])
            
            for i in range(MAX_FARMHAND_HIRES):
                if i < len(self.past_obs['farms'][player_num]['hands']):
                    pos_farmer_farmhand.append(self.past_obs['farms'][player_num]['hands'][i])
                else:
                    pos_farmer_farmhand.append(None)
            combined_ffa_w_pos = []
            for i in range(len(combined_farmer_farmhand_actions)):
                combined_ffa_w_pos.append([combined_farmer_farmhand_actions[i], pos_farmer_farmhand[i]])
            if obs['step'] % 24 == 0:
                for action in combined_farmer_farmhand_actions:
                    if action == "HIRE": 
                        # print("HIRE HAS BEEN MADE")
                        reward += 0.5
            for action in combined_farmer_farmhand_actions:
                if action == "PASS": 
                    reward -= 0.01

            for t in tiles: 
                if t is not None and t == "WEED":
                    # print("WEED ON BOARD")
                    reward -= 0.5
            for y in range(10):
                for x in range(10):
                    tile = obs['farms'][player_num]['tiles'][y][x]
                    past_tile = self.past_obs['farms'][player_num]['tiles'][y][x]
                    if ["BUILD_PASTURE", [y, x]] in combined_ffa_w_pos or ["BUILD_COOP", [y, x]] in combined_ffa_w_pos:
                        # print("placed?")
                        if past_tile == None and np.linalg.norm(np.array([y, x]) - (4,4)) <= 3:
                        # NEED TO ADD POSITION OF ALL COMBINED FARMER HAND AND LINK WHICH ONES TO WHICH
                            # print("COOP/PASTURE HAS BEEN PLACED")
                            reward += (1/(1+np.pow(1.5, np.linalg.norm(np.array([y, x]) - (4,4))))) * 3
                    if isinstance(past_tile, dict):
                        if past_tile.get("animal") is not None and past_tile.get("kind") in ("COOP", "PASTURE"):
                            if [["PLACE", "COW"], [y, x]] in combined_ffa_w_pos or [["PLACE", "GOOSE"], [y, x]] in combined_ffa_w_pos or [["PLACE", "SHEEP"], [y,x]] in combined_ffa_w_pos:
                                if np.linalg.norm(np.array([y, x]) - (4,4)) <= 3:
                                    # print("ANIMAL HAS BEEN PLACED IN COOP")
                                    reward += (1/(1+np.pow(1.5, np.linalg.norm(np.array([y, x]) - (4,4))))) * 3
                    elif past_tile is None and ["PLANT", [y,x]] in combined_ffa_w_pos:
                        if np.linalg.norm(np.array([y, x]) - (4,4)) > 3:
                            # print("PLANT HAS BEEN PLANTED OUTSIDE OF RADIUS 3")
                            reward += (1/(1+np.pow(1.5, np.linalg.norm(np.array([y, x]) - (4,4)))))

                    elif type(past_tile) == str and past_tile == "WEED" and ["DIG", [y,x]] in combined_ffa_w_pos:
                        # print("WEED HAS BEEN DUG")
                        reward += 0.5
                    if type(past_tile) == dict:
                        if "plant" in past_tile.keys() and past_tile['watered_today'] == False and ["WATER", [y,x]] in combined_ffa_w_pos:
                            # print("PLANT HAS BEEN WATERED")
                            reward += 0.5
                        elif "plant" in past_tile.keys() and (past_tile['fertilized_until_day'] - obs['day']) == 0 and ["FERTILIZE", [y,x]] in combined_ffa_w_pos:
                            reward += 0.2
                        #elif "plant" in past_tile.keys():
                            #and self.past_action == "HARVEST": reward += 11111111
                        #elif "plant" in past_tile.keys() and self.past_action == "HARVEST" and past_tile['yield_units']/self.dict_of_lists['list_of_max_yield_crops'][self.dict_of_lists['list_of_all_crops'][past_tile['crop']]] > 0.5:
                            #reward += 0.25
                            #if self.past_action == "HARVEST" and past_tile['yield_units']/self.dict_of_lists['list_of_max_yield_crops'][self.dict_of_lists['list_of_all_crops'][past_tile['crop']]] == 1:
                                #reward += 0.5
                        elif "plant" in past_tile.keys() and ["HARVEST", [y,x]] in combined_ffa_w_pos and past_tile['yield_units']/self.dict_of_lists['list_of_max_yield_crops'][self.dict_of_lists['list_of_all_crops'][past_tile['crop']]] > 0.5:
                            reward += 0.5
                            if ["HARVEST", [y,x]] in combined_ffa_w_pos and past_tile['yield_units']/self.dict_of_lists['list_of_max_yield_crops'][self.dict_of_lists['list_of_all_crops'][past_tile['crop']]] == 1:
                                reward += 0.5
                                
                        # CHECK IF THIS IS CORRECT
                        elif "plant" in past_tile.keys() and past_tile['consecutive_unwatered'] >= 1:
                            reward -= 0.5
                            
                        elif "animal" in past_tile.keys() and past_tile['yield_units']/self.dict_of_lists['list_of_max_yield_animals'][self.dict_of_lists['list_of_all_animals'][past_tile['animal']]] > 0.5:
                            reward += 0.25
                            if past_tile['yield_units']/self.dict_of_lists['list_of_max_yield_animals'][self.dict_of_lists['list_of_all_animals'][past_tile['animal']]] == 1:
                                reward += 0.5
                        
                        elif "animal" in past_tile.keys() and past_tile['fed_today'] == False and ["FEED", [y,x]] in combined_ffa_w_pos:
                            # print("ANIMAL HAS BEEN FED")
                            reward += 0.35
                        elif "animal" in past_tile.keys() and past_tile['cared_today'] == False and ["CARE", [y,x]] in combined_ffa_w_pos:
                            # print("ANIMAL HAS BEEN CARED FOR")
                            reward += 0.35
                        elif "animal" in past_tile.keys() and past_tile['consecutive_unfed'] >= 1:
                            reward -= 0.5
                        
        return reward