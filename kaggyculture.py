from kaggle_environments import make
import numpy as np
from torch.distributions.categorical import Categorical
from global_vars import MARKET_SEED_PRICES
import torch

# HELPFUL DICTIONARIES
list_of_all_sellable_resources = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER"]
list_of_all_crops = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"]
list_of_all_animals = ['GOOSE', 'COW', 'SHEEP']
list_of_all_shops = ["BAKERY", "PIZZA_SHOP", "BRUNCH_SPOT", "YARN_STORE", "ICE_CREAM_SHOP", "PET_CAFE", "SMOOTHIE_SHOP", "FARMERS_MARKET"]

# market info
list_of_market_deviation = [400, 450, 200, 100, 300, 332, 122, 105, 200]
list_of_original_market_prices_resources_sell = [25, 35, 60, 120, 250, 50, 160, 200, 100]
list_of_original_market_prices_crops_buy = [10, 20, 50, 100, 80]
list_of_original_market_prices_animals_buy = [300, 400, 500]

# crop info
list_of_first_yield_day_crops = [2, 4, 8, 10, 10]
list_of_max_yield_steps_crops = [96, 72, 264, 384, 240]
list_of_max_yield_crops = [6, 4, 4, 4, 6]
list_of_max_yield_animals = [4, 6, 6]
list_of_max_yield_steps_animals = []

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
            feature_vector = np.zeros(26)
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
                    consecutive_unwatered = tile['consecutive_unwatered']/MAX_UNWATERED_DAYS
                    yield_units = tile['yield_units']/list_of_max_yield_crops[list_of_all_crops.index(tile['crop'])]                    
                    max_lifespan_step = (tile['max_lifespan_step'] - obs['step']) / list_of_max_yield_steps_crops[list_of_all_crops.index(tile['crop'])]
                    fertilized_until_day = max(0, (tile['fertilized_until_day'] - obs['day'])/3)
                    feature_vector[len(list_of_all_crops)+3:len(list_of_all_crops)+10] = [steps_since_planted_norm, watered_today, consecutive_unwatered, yield_units, max_lifespan_step, fertilized_until_day]
                elif (tile['kind'] == 'COOP' or tile['kind'] == 'PASTURE'):
                    if tile['kind'] == 'COOP':
                        feature_vector[len(list_of_all_crops)+10] = 1
                    elif tile['kind'] == 'PASTURE':
                        feature_vector[len(list_of_all_crops)+11] = 1
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
                        feature_vector[animal_vector_ind:animal_vector_ind+7] = [steps_since_placed_norm, yield_units_animal, fed_today, consecutive_unfed, cared_today, fertilizer_available, pending_care_bonus]
            zero_matrix_copy[y][x] = feature_vector
    return zero_matrix_copy

# action mask

def invalid_action(obs, action, farmhand_index=None):
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

    if tile == "LOCKED": return invalid

    if len(action) >= 2:
        item = action[1]
        num = 1
        if len(action) == 3:
            num = action[2]

    if type_action == "PICKUP":
        if len(action) < 2: return invalid
        if obs['private']['shed'].get(item, 0) == 0: return invalid
        if not shed_adjacent: return invalid
        if num <= 0: return invalid

    if type_action == "PLACE":
        if len(action) < 2: return invalid
        if inventory.get(item, 0) == 0: return invalid
        if num <= 0: return invalid

        if isinstance(tile, dict) and tile.get("kind") in ["COOP", "PASTURE"]:
            if item in list_of_all_animals:
                if tile.get("animal") is None: return invalid
                if tile["kind"] == "COOP" and item != "GOOSE": return invalid
                if tile["kind"] == "PASTURE" and item == "GOOSE": return invalid
            elif not shed_adjacent:
                return invalid

        elif not shed_adjacent:
            return invalid

    if type_action == "NORTH" and y == 0: return invalid
    if type_action == "SOUTH" and y == 9: return invalid
    if type_action == "EAST" and x == 9: return invalid
    if type_action == "WEST" and x == 0: return invalid

    if type_action == "DROP":
        if not shed_adjacent: return invalid
        if sum(inventory.values()) == 0: return invalid

    if type_action == "PLANT":
        if len(action) < 2: return invalid
        if item not in list_of_all_crops: return invalid
        if tile is not None: return invalid
        if obs['private']['seeds'].get(item, 0) == 0: return invalid

    if type_action == "WATER":
        if not isinstance(tile, dict): return invalid
        if tile.get("kind") != "PLANT": return invalid
        if tile.get("watered_today", False): return invalid

    if type_action == "HARVEST":
        if not isinstance(tile, dict): return invalid
        if tile.get("kind") == "PLANT":
            if tile.get("yield_units", 0) <= 0: return invalid
            first_yield_day = list_of_first_yield_day_crops[list_of_all_crops.index(tile.get("crop"))]
            if obs["day"] - tile["planted_day"] < first_yield_day:
                return invalid

    if type_action == "FERTILIZE":
        if not isinstance(tile, dict): return invalid
        if tile.get("kind") != "PLANT": return invalid
        if inventory.get("FERTILIZER", 0) == 0: return invalid

    if type_action == "BUILD_COOP":
        if tile is not None: return invalid

    if type_action == "BUILD_PASTURE":
        if tile is not None: return invalid

    if type_action == "FEED":
        if not isinstance(tile, dict): return invalid
        if tile.get("animal") is None: return invalid
        if tile.get("fed_today", False): return invalid
        if inventory.get("WHEAT", 0) == 0: return invalid

    if type_action == "COLLECT_FERTILIZER":
        if not isinstance(tile, dict): return invalid
        if tile.get("animal") is None: return invalid
        if not tile.get("fertilizer_available"): return invalid

    if type_action == "CARE":
        if not isinstance(tile, dict): return invalid
        if tile.get("animal") is None: return invalid
        if tile.get("cared_today") is None or tile.get("cared_today") == True: return invalid

    if type_action == "DIG":
        if tile is None: return invalid
        if tile == "LOCKED": return invalid
        if not (tile.get("kind") in ("PLANT", "WEED") or (tile.get("kind") in ("COOP", "PASTURE") and tile.get("animal") is None)): return invalid
        if isinstance(tile, dict) and tile.get("animal") is not None: return invalid

    if type_action == "BUY_SEED":
        if len(action) < 3: return invalid
        if num <= 0: return invalid
        if item not in list_of_all_crops: return invalid
        ### list_of_original_market_prices_crops_sell[action[2]]
        if money < list_of_original_market_prices_crops_buy[list_of_all_crops.index(item)] * action[2]: return invalid


    if type_action == "BUY_PRODUCT":
        if len(action) < 3: return invalid
        if num <= 0: return invalid
        if item not in ["WHEAT", "FERTILIZER"]: return invalid
        buy_price = obs['market']['prices'][item]
        if money < buy_price * action[2]: return invalid

    if type_action == "BUY_ANIMAL":
        if len(action) < 3: return invalid
        if num <= 0: return invalid
        if item not in list_of_all_animals: return invalid
        if money < list_of_original_market_prices_animals_buy[list_of_all_animals.index(item)] * action[2]: return invalid

    if type_action == "SELL":
        if len(action) < 3: return invalid
        if num <= 0: return invalid
        if item not in list_of_all_sellable_resources: return invalid
        if shed.get(item, 0) == 0: return invalid

    if type_action == "HIRE":
        hire_cost = _fib(obs['farms'][player_num]['hires_today'])
        if money < hire_cost: return invalid
        if len(obs["farms"][player_num]["hands"]) >= MAX_FARMHANDS: return invalid
        if obs.get("num_hands", 0) >= MAX_FARMHANDS: return invalid

    if type_action == "BUY_LAND":
        num_unlocked = len(obs['farms'][player_num]['unlocked_quadrants'])
        if num_unlocked == 4: return invalid
        land_cost = [1000, 2000, 4000][num_unlocked - 1]
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
        if index == 0:
            internal_obs["past_market_actions"] = set()

        if action_str == "HIRE":
            internal_obs["num_hands"] += 1
            if internal_obs["num_hands"] >= 5:
                return list(map(lambda x: invalid_action(internal_obs, x), action_space.values()))

        # Update money after buying something
        if type(action_str) == tuple:
            if "BUY_PRODUCT" in action_str[0]:
                internal_obs["farms"][player_num]["money"] -= internal_obs["market"]["prices"][action_str[1]] * action_str[2]
            else:
                internal_obs["farms"][player_num]["money"] -= MARKET_SEED_PRICES[action_str[1]] * action_str[2]
            mask = list(map(lambda x: invalid_action(internal_obs, x), action_space.values()))

        # No duplicate action
        if index > 0:
            for act in internal_obs["past_market_actions"]:
                if "BUY_SEED" in act[1]:
                    mask[act[0]:act[0]+5] = [False] * 5
                elif "BUY_PRODUCT" in act[1]:
                    mask[act[0]:act[0]+3] = [False] * 3
                elif "BUY_ANIMAL" in act[1]:
                    mask[act[0]:act[0]+2] = [False] * 2
                
        internal_obs["past_market_actions"].add((action_idx, action_str))

        return mask
        
    return mask