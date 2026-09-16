import global_vars
import numpy as np

# FARM ACTIONS
""" 
["NORTH"]
["SOUTH"]
["EAST"]
["WEST"]
["PICKUP", item, n]      # 9 items
["PLACE", item]          # 8 placeable items
["HARVEST"]
["FERTILIZE"]
["BUILD_COOP"]
["BUILD_PASTURE"]
["FEED"]
["COLLECT_FERTILIZER"]
["CARE"]
["DIG"]
["PASS"]                 # 15th
"""

num_market_actions = 6
def farmhand_action_space():    
    actions = []
    actions.extend(np.arange(0,4))
    actions.extend([[4] + [n, y] for n in range(9) for y in range(global_vars.MAX_ITEMS_PICKUP)])
    actions.extend([[5] + [n] for n in range(8)])
    actions.extend(np.arange(6, 15))
    return actions


# MARKET ACTIONS
""" 
["BUY_SEED", crop, n]       # 5 crops
["BUY_PRODUCT", item, n]    # 4 products
["BUY_ANIMAL", animal, n]   # 3 animals
["SELL", item, n]           # 9 sellable resources
["HIRE"]
["BUY_LAND"]
 """

def market_action_space():
    actions = []
    actions.extend([[15] + [n, y] for n in range(5) for y in range(global_vars.MAX_NUM_BUY_SEED)])
    actions.extend([[16] + [n, y] for n in range(4) for y in range(global_vars.MAX_NUM_BUY_PRODUCT)])
    actions.extend([[17] + [n, y] for n in range(3) for y in range(global_vars.MAX_NUM_BUY_ANIMAL)])
    actions.extend([[18] + [n, y] for n in range(9) for y in range(global_vars.MAX_NUM_SELL)])
    actions.extend([19] + [n] for n in range(global_vars.MAX_HIRE_FARMHANDS_PER_TURN))
    actions.extend([20])
    
    return actions