import global_vars
import numpy as np

list_of_all_pickupable_resources = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER", "GOOSE", "COW", "SHEEP"]
list_of_all_placeable_resources = ["WHEAT","CARROT","TOMATO","STRAWBERRY","MELON","GOOSE", "COW", "SHEEP"]
list_of_all_sellable_resources = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER"]
list_of_all_crops = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"]
list_of_all_animals = ['GOOSE', 'COW', 'SHEEP']
list_of_all_products = ["EGG", "MILK", "WOOL", "FERTILIZER"]
#
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
["PLANT", item, n]       # 5 placeable items
["PASS"]                 
"""
def farmhand_action_space():    
    actions = {}
    ct = 0
    directions = ["NORTH", "SOUTH", "EAST", "WEST"]
    for i in range(4):
        actions[directions[i]] = ct
        ct += 1
    
    for i in range(len(list_of_all_pickupable_resources)):
        for a in range(global_vars.MAX_ITEMS_PICKUP):
            t = tuple(('PICKUP', list_of_all_pickupable_resources[i], a+1))
            actions[t] = ct
            ct += 1
            
    # place resource either in shed or coop
    for i in range(len(list_of_all_pickupable_resources)):
        t = tuple(('PLACE', list_of_all_pickupable_resources[i]))
        actions[t] = ct
        ct += 1
    
    for i in range(len(list_of_all_crops)):
        t = tuple(('PLANT', list_of_all_crops[i]))
        actions[t] = ct
        ct += 1
    
    remaining = ["WATER", "HARVEST", "FERTILIZE","BUILD_COOP","BUILD_PASTURE","FEED","COLLECT_FERTILIZER","CARE","DIG","PASS"]
    for i in range(len(remaining)):
        actions[remaining[i]] = ct     
        ct += 1
        
    reversed_dict = {value: key for key, value in actions.items()}
    return reversed_dict

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
    actions = {}
    list_all_actions = []
    ct = 0
    for i in range(5):
        for a in range(0, global_vars.MAX_NUM_BUY_SEED, global_vars.BUY_SEED_INTERVAL):
            actions[tuple(("BUY_SEED", list_of_all_crops[i], a+1))] = ct
            ct += 1
    
    for i in range(1):
        for a in range(0, global_vars.MAX_NUM_BUY_PRODUCT, global_vars.BUY_PRODUCT_INTERVAL):
            actions[tuple(("BUY_PRODUCT", "WHEAT", a+1))] = ct
            actions[tuple(("BUY_PRODUCT", "FERTILIZER", a+1))] = ct
            ct += 1
            
    for i in range(3):
        for a in range(0, global_vars.MAX_NUM_BUY_ANIMAL, global_vars.BUY_ANIMAL_INTERVAL):
            actions[tuple(("BUY_ANIMAL", list_of_all_animals[i], a+1))] = ct
            ct += 1
            
    for i in range(len(list_of_all_sellable_resources)):
        for a in range(0, global_vars.MAX_NUM_SELL, global_vars.SELL_INTERVAL):
            actions[tuple(("SELL", list_of_all_sellable_resources[i], a+1))] = ct
            ct += 1
    
    actions['HIRE'] = ct 
    ct += 1
    actions['BUY_LAND'] = ct
    ct += 1
    actions["PASS"] = ct
    
    reversed_dict = {value: key for key, value in actions.items()}
    return reversed_dict