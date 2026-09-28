from kaggle_environments import make
import numpy as np
from torch.distributions.categorical import Categorical
import global_vars
import torch

# HELPFUL DICTIONARIES

list_of_all_sellable_resources = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER"]
list_of_all_crops = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"]
list_of_all_animals = ['GOOSE', 'COW', 'SHEEP']
list_of_all_shops = ["BAKERY", "PIZZA_SHOP", "BRUNCH_SPOT", "YARN_STORE", "ICE_CREAM_SHOP", "PET_CAFE", "SMOOTHIE_SHOP", "FARMERS_MARKET"]
list_of_all_harvestable = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "GOOSE", "COW", "SHEEP"]

# market info
list_of_market_deviation = [400, 450, 200, 100, 300, 332, 122, 105, 200]
list_of_original_market_prices_resources_sell = [25, 35, 60, 120, 250, 50, 160, 200, 100]
list_of_original_market_prices_crops_buy = [10, 20, 50, 100, 80]
list_of_original_market_prices_animals_buy = [300, 400, 500]

# crop info
list_of_first_yield_day_crops = [2, 4, 8, 10, 10]
list_of_first_yield_day_animals = [4, 8, 6]
list_of_max_yield_steps_crops = [96, 72, 264, 384, 240]
list_of_max_yield_crops = [6, 4, 4, 4, 6]
list_of_max_yield_animals = [4, 6, 6]
list_of_first_yield_combined = [2, 4, 8, 10, 10, 4, 8, 6]

list_of_shop_types = ['BAKERY', "PIZZA SHOP", "BRUNCH SPOT", "YARN STORE", "ICE CREAM SHOP", "PET CAFE", "SMOOTHIE SHOP", "FARMERS MARKET"]

# HELPFUL FUNCTIONS
def _fib(n):
    a, b = 1, 1
    for _ in range(n):
        a, b = b, a + b
    return a


# GLOBAL VARS
MAX_INVEN_CAP = 25
MAX_FARMHANDS = 5
MAX_HIRE_PRICE = _fib(MAX_FARMHANDS)


# PRECOMPUTES
# pre-computed one-hot for all shops
unlocked_shop_one_hot = {}
none_one_hot = np.eye(9)[8]

for i, shop in enumerate(list_of_all_shops):
    vec = [0] * 9
    vec[i] = 1
    unlocked_shop_one_hot[shop] = vec


def setup_non_tile_observation(obs):
    player_num = obs["player"]
    market_curr_inven = obs['market']['inventory']
    market_inventory = [(market_curr_inven[inven] - 10000) / list_of_market_deviation[list_of_all_sellable_resources.index(inven) ] for inven in market_curr_inven]
    # market_prices = [i for i in obs['market']['prices'].values() / list_of_original_market_prices_resources_sell[i] for i in list_of_original_market_prices_resources_sell]
    market_prices = [obs['market']['prices'][key]/list_of_original_market_prices_resources_sell[list_of_all_sellable_resources.index(key)] for key in obs['market']['prices'].keys()]


    def encode_inventory(inv, max_inv, resource_arr):
        if max_inv == 0:
            return [0.0] * len(resource_arr)

        return [inv.get(resource, 0) / MAX_INVEN_CAP for resource in resource_arr]

    shed_inven = encode_inventory(obs['private']['shed'], MAX_INVEN_CAP, list_of_all_sellable_resources)
    seed_inven = encode_inventory(obs['private']['seeds'], MAX_INVEN_CAP, list_of_all_crops)

    inventory_obs = []
    num_hands = 0
    for inv in obs["private"]["inventories"]:
        # print("MORE HANDS 1")
        num_hands += 1
        inventory_obs.extend(encode_inventory(inv, MAX_INVEN_CAP, list_of_all_sellable_resources))

    for i in range(num_hands, MAX_FARMHANDS+1):
        # print("MORE HANDS 2")
        inventory_obs.extend([0] * len(list_of_all_sellable_resources))

    money = (obs['farms'][player_num]['money'] - 3000) / 3000

    curr_num_hires = len(obs['farms'][player_num]['hands'])

    cost_of_next_hire = MAX_HIRE_PRICE
    # maybe cumulative sum of hired already?

    unlocked_shops = [0] * 8
    for shop in obs['town']['unlocked_shops']:
        unlocked_shops[list_of_all_shops.index(shop)] += 1
    unlocked_shops = [shop/8 for shop in unlocked_shops]
        
    obs_arr = [money] + market_inventory + market_prices + shed_inven + seed_inven + inventory_obs + [curr_num_hires] + [curr_num_hires/cost_of_next_hire] + unlocked_shops
    return obs_arr

MAX_UNWATERED_DAYS = 30
def setup_tile_obs(obs):
    player_num = obs["player"]
    # list_of_all_animals_w_none = list_of_all_animals + [None]
    # tile_options = [None, "LOCKED"] + list_of_all_crops + ["WEED"] + list_of_all_animals
    
    # list_of_layers = []
    
    zero_matrix_copy = [[0 for _ in range(10)] for _ in range(10)]
    # plant_num = 0
    # animal_num = 0
    for y in range(10):
        for x in range(10):
            tile = obs['farms'][player_num]['tiles'][y][x]
            feature_vector = np.zeros(31)
            if type(tile) == str or tile is None:
                if tile == None:
                    feature_vector[0] = 1
                elif tile == 'LOCKED':
                    feature_vector[1] = 1
                elif tile == "WEED":
                    feature_vector[2] = 1
            else:
                if tile['kind'] == "PLANT":
                    feature_vector[3+list_of_all_crops.index(tile['crop'])] = 1
                    steps_since_planted_norm = (obs['step'] - (tile['planted_day']*24))/list_of_max_yield_steps_crops[list_of_all_crops.index(tile['crop'])]
                    watered_today = int(tile['watered_today'])
                    # consecutive_unwatered = tile['consecutive_unwatered']/MAX_UNWATERED_DAYS
                    yield_units = tile['yield_units']/list_of_max_yield_crops[list_of_all_crops.index(tile['crop'])]                    
                    max_lifespan_step = (tile['max_lifespan_step'] - obs['step']) / list_of_max_yield_steps_crops[list_of_all_crops.index(tile['crop'])]
                    fertilized_until_day = max(0, (tile['fertilized_until_day'] - obs['day'])/3)
                    feature_vector[len(list_of_all_crops)+3:len(list_of_all_crops)+8] = [steps_since_planted_norm, watered_today, yield_units, max_lifespan_step, fertilized_until_day]
                elif (tile['kind'] == 'COOP' or tile['kind'] == 'PASTURE'):
                    if tile['kind'] == 'COOP':
                        feature_vector[len(list_of_all_crops)+9] = 1
                    elif tile['kind'] == 'PASTURE':
                        feature_vector[len(list_of_all_crops)+10] = 1
                    if tile.get("animals", None) is not None:
                        feature_vector[len(list_of_all_crops)+11+list_of_all_animals.index(tile['animals'])] = 1

                        steps_since_placed_norm = (obs['step'] - (tile['placed_day']*24))/720
                        yield_units_animal = tile['yield_units']/list_of_max_yield_animals[list_of_all_animals.index(tile['animal'])]                    
                        fed_today = int(tile['fed_today'])
                        consecutive_unfed = tile['consecutive_unfed']/2
                        cared_today = int(tile['cared_today'])
                        fertilizer_available = int(tile['fertilizer_available'])
                        pending_care_bonus = tile['pending_care_bonus']/list_of_max_yield_animals[list_of_all_animals.index(tile['animal'])]                    
                        animal_vector_ind = len(list_of_all_crops)+11+len(list_of_all_animals)
                        feature_vector[animal_vector_ind:animal_vector_ind+6] = [steps_since_placed_norm, yield_units_animal, fed_today, consecutive_unfed, cared_today, fertilizer_available, pending_care_bonus]
            zero_matrix_copy[y][x] = feature_vector
    farmer_position = obs['farms'][player_num]['farmer']
    zero_matrix_copy[farmer_position[1], farmer_position[0]][len(feature_vector)-6] = 1
    for i in range(len(obs['farms'][player_num]['hands'])):
        farmhand_position = obs['farms'][player_num]['hands'][i]
        zero_matrix_copy[farmhand_position[1], farmhand_position[0]][len(feature_vector)-6+i+1] = 1
    return zero_matrix_copy

# action mask

def invalid_action(obs, action, farmhand_index=None):
    # print(action)
    invalid, valid = False, True
    if type(action) in [list, tuple]: type_action = action[0]
    else: type_action = action

    if type_action == "PASS":
        return valid
    # obs = obs["observation"]
    player_num = obs["player"]
    money = obs['farms'][player_num]['money']
    shed = obs['private']['shed']

    if farmhand_index is None:
        x, y = obs['farms'][player_num]['farmer']
        inventory = obs['private']['inventories'][0]
    else:
        if farmhand_index < len(obs['farms'][player_num]['hands']):
            x, y = obs['farms'][player_num]['hands'][farmhand_index]
            inventory = obs['private']['inventories'][farmhand_index + 1]
        else:
            if action != "PASS":
                return invalid

    tile = obs['farms'][player_num]['tiles'][y][x]
    shed_adjacent = (x, y) in [(4, 4), (5, 4), (4, 5), (5, 5)]
    if type(action) is tuple and len(action) >= 2:
        item = action[1]
        num = 1
        if len(action) == 3:
            num = action[2]
    
    if type_action == "NORTH" and y == 0: return invalid
    if type_action == "SOUTH" and y == 9: return invalid
    elif type_action == "SOUTH" and obs['farms'][player_num]['tiles'][min(y+1, 9)][x] == "LOCKED" and obs['farms'][player_num]['tiles'][y][x] == "LOCKED": return invalid
    if type_action == "EAST" and x == 9: return invalid
    elif type_action == "EAST" and obs['farms'][player_num]['tiles'][y][min(x+1, 9)] == "LOCKED" and obs['farms'][player_num]['tiles'][y][x] == "LOCKED": return invalid
    if type_action == "WEST" and x == 0: return invalid

    if type_action not in ["NORTH", "SOUTH", "EAST", "WEST"] and tile == "LOCKED": return invalid
            
    if type_action == "PICKUP":
        # if len(action) < 2: return invalid
        if obs['private']['shed'].get(item, 0) == 0: return invalid
        if isinstance(tile, dict) and tile.get("kind") in ["COOP", "PASTURE"]: return invalid
        if not shed_adjacent: return invalid
        # if num <= 0: return invalid

    if type_action == "PLACE":
        # if len(action) < 2: return invalid
        if inventory.get(item, 0) == 0: return invalid
        # if num <= 0: return invalid

        if isinstance(tile, dict) and (tile.get("kind") in ["COOP", "PASTURE"]):
            if item in list_of_all_animals:
                if tile["kind"] == "COOP" and item != "GOOSE": return invalid
                if tile["kind"] == "PASTURE" and item == "GOOSE": return invalid
                if tile.get("animal") is not None: return invalid

            """ elif not shed_adjacent:
                return invalid """

        elif not shed_adjacent:
            return invalid
        
    if type_action == "DROP":
        if not shed_adjacent: return invalid
        if sum(inventory.values()) == 0: return invalid

    if type_action == "PLANT":
        # does not exist in action space
        # if len(action) < 2: return invalid
        
        # action should not exist in viable actions
        #   if item not in list_of_all_crops: return invalid
        if tile is not None: return invalid
        if obs['private']['seeds'].get(item, 0) == 0: return invalid

    if type_action == "WATER":
        if not isinstance(tile, dict): return invalid
        if tile.get("kind") != "PLANT": return invalid
        if tile.get("watered_today", True): return invalid

    if type_action == "HARVEST":
        if not isinstance(tile, dict): return invalid
        if tile.get('kind') == "PLANT":
            if obs['day'] - tile.get('planted_day', 0) < list_of_first_yield_combined[list_of_all_harvestable.index(tile.get('crop'))]: return invalid
        elif tile.get('kind') in ["COOP", "PASTURE"]:
            if tile.get("animal", None) is not None and obs['day'] - tile.get('planted_day', 0) < list_of_first_yield_combined[list_of_all_harvestable.index(tile.get('animal', None))]: return invalid            
        # if tile.get("kind") == "PLANT":
        #     if tile.get("yield_units") <= 0: return invalid
        # elif tile.get("kind") in ["COOP", "PASTURE"]:
        #     if tile.get("yield_units") <= 0: return invalid
        """ first_yield_day = list_of_first_yield_day_crops[list_of_all_crops.index(tile.get("crop"))]
            if obs["day"] - tile["planted_day"] < first_yield_day:
                return invalid """

    if type_action == "FERTILIZE":
        if not isinstance(tile, dict): return invalid
        if tile.get("kind") != "PLANT": return invalid
        if (tile.get("fertilized_until_day") - obs['day']) > 0: return invalid
        if inventory.get("FERTILIZER", 0) == 0: return invalid

    if type_action == "BUILD_COOP":
        if tile is not None: return invalid

    if type_action == "BUILD_PASTURE":
        if tile is not None: return invalid

    if type_action == "FEED":
        if not isinstance(tile, dict): return invalid
        if tile.get("animal") is None: return invalid
        if tile.get("fed_today", True): return invalid
        if inventory.get("WHEAT", 0) == 0: return invalid

    if type_action == "COLLECT_FERTILIZER":
        if not isinstance(tile, dict): return invalid
        if tile.get("animal") is None: return invalid
        if tile.get("fertilizer_available") == False: return invalid

    if type_action == "CARE":
        if not isinstance(tile, dict): return invalid
        if tile.get("animal") is None: return invalid
        if tile.get("cared_today") == True: return invalid

    if type_action == "DIG":
        if tile is None: return invalid
        # redundant
        # if tile == "LOCKED": return invalid
        if isinstance(tile, dict):
            if tile.get("kind") in ["COOP", "PLANT", "PASTURE"]: return invalid
            # if tile.get("kind") not in ["PLANT", "WEED", "COOP", "PASTURE"]: return invalid
            # if tile.get("kind") in ["COOP", "PASTURE"]:
                # if tile.get("animal") is None: return invalid
         # if isinstance(tile, dict) and tile.get("animal") is not None: return invalid

    if type_action == "BUY_SEED":
        # should not have to exist, operation does not exist in action space
        # if item not in list_of_all_crops: return invalid
        # if len(action) < 3: return invalid
        # if num <= 0: return invalid

        ### list_of_original_market_prices_crops_sell[action[2]]
        if money + global_vars.MIN_MONEY < obs['market']['prices'][item] * num: return invalid
        # if money < list_of_original_market_prices_crops_buy[list_of_all_crops.index(item)] * action[2]: return invalid


    if type_action == "BUY_PRODUCT":
        # should not have to exist
        # if len(action) < 3: return invalid
        # if num <= 0: return invalid
        # if item not in ["WHEAT", "FERTILIZER"]: return invalid
        if money + global_vars.MIN_MONEY < obs['market']['prices'][item] * num: return invalid

    if type_action == "BUY_ANIMAL":
        # should not have to exist
        # if len(action) < 3: return invalid
        # if num <= 0: return invalid
        # if item not in list_of_all_animals: return invalid
        if money + global_vars.MIN_MONEY < list_of_original_market_prices_animals_buy[list_of_all_animals.index(item)] * num: return invalid

    if type_action == "SELL":
        # should not have to exist
        # if len(action) < 3: return invalid
        # if num <= 0: return invalid
        if item not in list_of_all_sellable_resources: return invalid
        
        # AN ISSUE EXISTS WITH SELL, if we assume we take into account all inven of farmers
        inven = 0
        for i in range(len(obs["farms"][player_num]["hands"])):
            inven += obs['private']['inventories'][i+1].get(item, 0)
        if shed.get(item, 0) + inven == 0: return invalid

    if type_action == "HIRE":
        hire_cost = _fib(obs['farms'][player_num]['hires_today'])
        if money < hire_cost: return invalid
        
        # redundant but which one is right, why do u need both?
        if len(obs["farms"][player_num]["hands"]) >= MAX_FARMHANDS: return invalid
        if obs.get("num_hands", 0) >= MAX_FARMHANDS: return invalid

    if type_action == "BUY_LAND":
        num_unlocked = len(obs['farms'][player_num]['unlocked_quadrants'])
        land_cost = [1000, 2000, 4000][num_unlocked-1]
        if money < land_cost: return invalid

    return valid

def update_internal_obs(internal_obs):
    all_pos = []
    obs = internal_obs
    player_num = obs["player"]
    x, y = obs["farms"][player_num]["farmer"]
    all_pos.append((x,y))
    for i in range(len(obs["farms"][player_num]["hands"])):
        x, y = obs["farms"][player_num]["hands"][i]
        all_pos.append((x,y))

    internal_obs["all_pos"] = all_pos
    internal_obs["num_hands"] = internal_obs["farms"][player_num]["hires_today"]
    # internal_obs["seeds"] = obs["private"]["seeds"]
    # internal_obs["balance"] = obs["farms"][player_num]["money"]


def update_mask(mask, internal_obs, action, action_space, index = None, is_farm=False):
    player_num = internal_obs["player"]
    if is_farm == True:
        # Update mask if on same tile as previous
        if index >= 0:
            past_pos = set(internal_obs["all_pos"][:index])
            if index < len(internal_obs["all_pos"]) and internal_obs["all_pos"][index] in past_pos:
                mask[4:-1] = [False] * len(mask[4:-1])
                return mask

        # Update internal_obs player inventory if farmer plants seeds
        if "PLANT" in action:
            item = action[1]
            internal_obs["private"]["seeds"][item] -= 1
            mask = list(map(lambda x: invalid_action(internal_obs, x, index), action_space.values()))
            return mask
    else:
        action_idx, action_str = action
        action_idx = int(action_idx)
        if index == 0: internal_obs["past_market_actions"] = set()
                    
        if action_str == "HIRE":
            internal_obs["num_hands"] += 1
            if internal_obs["num_hands"] >= 5:
                return list(map(lambda x: invalid_action(internal_obs, x), action_space.values()))

        # Update money after buying something
        if type(action_str) == tuple and "BUY" in action_str[0]:
            if action_str[0] == "BUY_PRODUCT":
                internal_obs["farms"][player_num]["money"] -= internal_obs['market']['prices'][action_str[1]] * action_str[2]
            elif action_str[0] == "BUY_ANIMAL":
                internal_obs["farms"][player_num]["money"] -= list_of_original_market_prices_animals_buy[list_of_all_animals.index(action_str[1])]* action_str[2]                
            elif action_str[0] == "BUY_SEED":
                internal_obs["farms"][player_num]["money"] -= global_vars.MARKET_SEED_PRICES[action_str[1]] * action_str[2]
            mask = list(map(lambda x: invalid_action(internal_obs, x), action_space.values()))

        offset = 0
        # No duplicate action
        # print("ACTION:", action)
        # print("action_str:", action_str)
        if type(action_str) == tuple and action_str[0] in ["BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL", "SELL"]:
            used_actions = [act[1] for act in internal_obs["past_market_actions"]]
            # print("USED ACTION: ", used_actions)
            if (action_str[0], action_str[1]) not in internal_obs["past_market_actions"]:
                # print("ACTION_STR[2]: ", action_str[2])
                if "BUY_SEED" == action_str[0]:
                    if global_vars.BUY_SEED_INTERVAL == 1: 
                        offset = action_str[2]//global_vars.BUY_SEED_INTERVAL - 1
                        mask[action[0]-offset:action[0]+(global_vars.MAX_NUM_BUY_SEED//global_vars.BUY_SEED_INTERVAL)-offset] = [False] * (global_vars.MAX_NUM_BUY_SEED//global_vars.BUY_SEED_INTERVAL)
                    else:
                        offset = action_str[2]//global_vars.BUY_SEED_INTERVAL
                        # print("first:", action[0]-offset)
                        # print("second:", action[0]+(global_vars.MAX_NUM_BUY_SEED//global_vars.BUY_SEED_INTERVAL) + 1-offset)
                        mask[action[0]-offset:action[0]+(global_vars.MAX_NUM_BUY_SEED//global_vars.BUY_SEED_INTERVAL) + 1-offset] = [False] * (global_vars.MAX_NUM_BUY_SEED//global_vars.BUY_SEED_INTERVAL + 1)
                        # print(mask)
                elif "BUY_PRODUCT" == action_str[0]:
                    if global_vars.BUY_PRODUCT_INTERVAL == 1: 
                        offset = action_str[2]//global_vars.BUY_PRODUCT_INTERVAL - 1
                        mask[action[0]-offset:action[0]+(global_vars.MAX_NUM_BUY_PRODUCT//global_vars.BUY_PRODUCT_INTERVAL)-offset] = [False] * (global_vars.MAX_NUM_BUY_PRODUCT//global_vars.BUY_PRODUCT_INTERVAL)
                    else:
                        offset = action_str[2]//global_vars.BUY_PRODUCT_INTERVAL
                        mask[action[0]-offset:action[0]+(global_vars.MAX_NUM_BUY_PRODUCT//global_vars.BUY_PRODUCT_INTERVAL) + 1-offset] = [False] * (global_vars.MAX_NUM_BUY_PRODUCT//global_vars.BUY_PRODUCT_INTERVAL + 1)
                elif "BUY_ANIMAL" == action_str[0]:
                    if global_vars.BUY_ANIMAL_INTERVAL == 1: 
                        offset = action_str[2]//global_vars.BUY_ANIMAL_INTERVAL - 1
                        mask[action[0]-offset:action[0]+(global_vars.MAX_NUM_BUY_ANIMAL//global_vars.BUY_ANIMAL_INTERVAL)-offset] = [False] * (global_vars.MAX_NUM_BUY_ANIMAL//global_vars.BUY_ANIMAL_INTERVAL)
                    else:
                        offset = action_str[2]//global_vars.BUY_ANIMAL_INTERVAL
                        mask[action[0]-offset:action[0]+(global_vars.MAX_NUM_BUY_ANIMAL//global_vars.BUY_ANIMAL_INTERVAL) + 1-offset] = [False] * (global_vars.MAX_NUM_BUY_ANIMAL//global_vars.BUY_ANIMAL_INTERVAL + 1)
                elif "SELL" == action_str[0]:
                    if global_vars.SELL_INTERVAL == 1: 
                        offset = action_str[2]//global_vars.SELL_INTERVAL  - 1
                        mask[action[0]-offset:action[0]+global_vars.MAX_NUM_SELL//global_vars.MAX_NUM_SELL-offset] = [False] * (global_vars.MAX_NUM_SELL//global_vars.MAX_NUM_SELL)
                    else:
                        offset = action_str[2]//global_vars.SELL_INTERVAL
                        mask[action[0]-offset:action[0]+global_vars.MAX_NUM_SELL//global_vars.MAX_NUM_SELL + 1-offset] = [False] * (global_vars.MAX_NUM_SELL//global_vars.MAX_NUM_SELL + 1)
            # else: print("MASKED")
            if type(action_str) == tuple:   
                # print(((action_str)[0], (action_str)[1], (action_str)[2]))
                # print(offset)
                internal_obs["past_market_actions"].add(((action_str)[0], (action_str)[1]))
                # print(internal_obs['past_market_actions'])

        return mask
        
    return mask


def load_dict_of_lists():
    dict_of_lists = {
        "list_of_all_sellable_resources": ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER"),
        "list_of_all_crops": ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"),
        "list_of_all_animals": ("GOOSE", "COW", "SHEEP"),
        "list_of_all_shops": ("BAKERY", "PIZZA_SHOP", "BRUNCH_SPOT", "YARN_STORE", "ICE_CREAM_SHOP", "PET_CAFE", "SMOOTHIE_SHOP", "FARMERS_MARKET"),

        # market info
        "list_of_market_deviation": (400, 450, 200, 100, 300, 332, 122, 105, 200),
        "list_of_original_market_prices_resources_sell": (25, 35, 60, 120, 250, 50, 160, 200, 100),
        "list_of_original_market_prices_crops_buy": (10, 20, 50, 100, 80),
        "list_of_original_market_prices_animals_buy": (300, 400, 500),

        # crop info
        "list_of_first_yield_day_crops": (2, 4, 8, 10, 10),
        "list_of_max_yield_steps_crops": (96, 72, 264, 384, 240),
        "list_of_max_yield_crops": (6, 4, 4, 4, 6),
        "list_of_max_yield_animals": (4, 6, 6),

        "list_of_shop_types": ("BAKERY", "PIZZA SHOP", "BRUNCH SPOT", "YARN STORE", "ICE CREAM SHOP", "PET CAFE", "SMOOTHIE SHOP", "FARMERS MARKET")
    }
    return dict_of_lists
