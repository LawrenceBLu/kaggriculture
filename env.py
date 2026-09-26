from kaggle_environments.envs.kaggriculture.kaggriculture import MARKET_PARAMS
import numpy as np
from kaggyculture import load_dict_of_lists
from global_vars import MAX_FARMHAND_HIRES

class Env():
    def __init__(self, env):
        self.env = env
        self.past_obs = None
        self.past_action = None
        self.reset()
        self.max_money = 0
        self.dict_of_lists = load_dict_of_lists()
        
    def reset(self):
        obs, _ = self.env.reset()
        self.past_obs = obs["observation"]
        self.steps = 0
        return obs
        
    def step(self, action):
        reward = 0
        self.past_action = action[0]
        self.steps += 1
        if self.steps == 720:
            obs = self.reset()
            self.past_market_action = None
            return obs, False, reward
        obs, _ = self.env.step(action)
        reward = self.get_reward(obs["observation"])
        self.past_obs = obs["observation"]
        if self.steps == 719:
            return obs, True, reward
        return obs, False, reward

    def get_reward(self, obs):
        reward = 0
        player_num = obs["player"]
        
        self.max_money = max(self.max_money, obs['farms'][player_num]['money'])
        if self.steps == 719:
            # rewards for win
            if obs['farms'][player_num]['money'] > obs['farms'][player_num^1]['money']:
                reward += 5
            # rewards for total money/25k
            reward += self.max_money/10000
            
        else:
            # rewards for first farms unlock
            if self.past_obs['farms'][player_num]['tiles'][0][9] == 'LOCKED' and obs['farms'][player_num]['tiles'][0][9] == None:
                reward += 3
            # rewards for second farms unlock
            if self.past_obs['farms'][player_num]['tiles'][9][0] == 'LOCKED' and obs['farms'][player_num]['tiles'][9][0] == None:
                reward += 3
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
            combined_farmer_farmhand_actions = [farmer_actions] + (farmhand_actions)
            pos_farmer_farmhand = []
            pos_farmer_farmhand.append(self.past_obs['farms'][player_num]['farmer'])
            
            for i in range(MAX_FARMHAND_HIRES):
                if i+1 <= len(self.past_obs['farms'][player_num]['hands']):
                    pos_farmer_farmhand.append(self.past_obs['farms'][player_num]['hands'][i])
                else:
                    pos_farmer_farmhand.append(None)
                    
            combined_ffa_w_pos = []
            for i in range(len(combined_farmer_farmhand_actions)):
                combined_ffa_w_pos.append([combined_farmer_farmhand_actions[i], pos_farmer_farmhand[i]])
            if obs['step'] % 24 == 0:
                for action in combined_farmer_farmhand_actions:
                    if action == "HIRE": reward += 0.5
            
            for action in combined_farmer_farmhand_actions:
                if action == "PASS": reward -= 0.15

            for t in tiles: 
                if t is not None and t == "WEED":
                    reward -= 0.05
                
            for y in range(10):
                for x in range(10):
                    tile = obs['farms'][player_num]['tiles'][y][x]
                    past_tile = self.past_obs['farms'][player_num]['tiles'][y][x]
                    if tile == None and np.linalg.norm(np.array([y, x]) - (4,4)) <= 3:
                        # NEED TO ADD POSITION OF ALL COMBINED FARMER HAND AND LINK WHICH ONES TO WHICH
                        if [["PLACE", "COOP"], [y, x]] in combined_ffa_w_pos or [['PLACE', "PASTURE"], [y, x]] in combined_ffa_w_pos:
                            reward += (1/(1+np.pow(1.5, np.linalg.norm(np.array([y, x]) - (4,4))))) * 3
                    if isinstance(tile, dict):
                        if tile.get("animal") is not None and tile.get("kind") in ("COOP", "PASTURE"):
                            if [["PLACE", "COW"], [y, x]] in combined_ffa_w_pos or [["PLACE", "GOOSE"], [y, x]] in combined_ffa_w_pos or [["PLACE", "SHEEP"], [y,x]] in combined_ffa_w_pos:
                                if np.linalg.norm(np.array([y, x]) - (4,4)) <= 3:
                                    reward += (1/(1+np.pow(1.5, np.linalg.norm(np.array([y, x]) - (4,4))))) * 3
                    elif tile is None and ["PLANT", [y,x]] in combined_ffa_w_pos:
                        if np.linalg.norm(np.array([y, x]) - (4,4)) > 3:
                            reward += (1/(1+np.pow(1.5, np.linalg.norm(np.array([y, x]) - (4,4)))))

                    elif type(past_tile) == str and past_tile == "WEED" and ["DIG", [y,x]] in combined_ffa_w_pos:
                        reward += 0.5
                    if type(past_tile) == dict:
                        if "plant" in past_tile.keys() and past_tile['watered_today'] == False and ["WATER", [y,x]] in combined_ffa_w_pos:
                            reward += 11
                        elif "plant" in past_tile.keys() and (past_tile['fertilized_until_day'] - obs['day']) == 0 and ["FERTILIZE", [y,x]] in combined_ffa_w_pos:
                            reward += 0.2
                        #elif "plant" in past_tile.keys():
                            #and self.past_action == "HARVEST": reward += 11111111
                        #elif "plant" in past_tile.keys() and self.past_action == "HARVEST" and self.past_tile['yield_units']/self.dict_of_lists['list_of_max_yield_crops'][self.dict_of_lists['list_of_all_crops'][past_tile['crop']]] > 0.5:
                            #reward += 0.25
                            #if self.past_action == "HARVEST" and past_tile['yield_units']/self.dict_of_lists['list_of_max_yield_crops'][self.dict_of_lists['list_of_all_crops'][past_tile['crop']]] == 1:
                                #reward += 0.5
                                
                        # CHECK IF THIS IS CORRECT
                        elif "plant" in past_tile.keys() and self.past_tile['consecutive_unwatered'] >= 1:
                            reward -= 0.5
                            
                        elif "animal" in past_tile.keys() and self.past_tile['yield_units']/self.dict_of_lists['list_of_max_yield_animals'][self.dict_of_lists['list_of_all_animals'][past_tile['animal']]] > 0.5:
                            reward += 0.25
                            if past_tile['yield_units']/self.dict_of_lists['list_of_max_yield_animals'][self.dict_of_lists['list_of_all_animals'][past_tile['animal']]] == 1:
                                reward += 0.5
                        
                        elif "animal" in past_tile.keys() and self.past_tile['fed_today'] == False and ["FEED", [y,x]] in combined_ffa_w_pos:
                            reward += 0.35
                        elif "animal" in past_tile.keys() and self.past_tile['cared_today'] == False and ["CARE", [y,x]] in combined_ffa_w_pos:
                            reward += 0.35
                        elif "animal" in past_tile.keys() and self.past_tile['consecutive_unfed'] >= 1:
                            reward -= 0.5
                        
        return reward