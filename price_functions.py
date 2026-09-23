import numpy as np

def hinge(x, T):
    u = x / T
    return u + 8 * np.power(max(0, u - 1), 2)

# WHEAT: sqrt below 0.8, log above 0.2
def wheat_adjusted_price(curr_inventory, deviation):
    total_deviation = (10000 - curr_inventory + deviation)
    ghost_price = 0
    if total_deviation > 0:
        ghost_price = 25 + (0.8 * 25) / np.sqrt(400) * np.sqrt(total_deviation)
    else:
        ghost_price = max(1, 25 - (0.2 * 25) / np.log(1 + 400) * np.log(1 + np.abs(total_deviation)))
    return np.round(ghost_price, 4)

# CARROT: hinge below 1.0, sqrt above 0.7
def carrot_adjusted_price(curr_inventory, deviation):
    total_deviation = (10000 - curr_inventory + deviation)
    ghost_price = 0
    if total_deviation > 0:
        ghost_price = 35 + (1.0 * 35) / 1 * hinge(np.abs(total_deviation), 450)
    else:
        ghost_price = max(1, 35 - (0.7 * 35) / np.sqrt(450) * np.sqrt(np.abs(total_deviation)))
    return np.round(ghost_price, 4)

# TOMATO: hinge below 0.4, sqrt above 0.6
def tomato_adjusted_price(curr_inventory, deviation):
    total_deviation = (10000 - curr_inventory + deviation)
    ghost_price = 0
    if total_deviation > 0:
        ghost_price = 60 + (0.4 * 60) / 1 * hinge(np.abs(total_deviation), 200)
    else:
        ghost_price = max(1, 60 - (0.6 * 60) / np.sqrt(200) * np.sqrt(np.abs(total_deviation)))
    return np.round(ghost_price, 4)

# STRAWBERRY: sqrt below 0.7, linear above 1.6
def strawberry_adjusted_price(curr_inventory, deviation):
    total_deviation = (10000 - curr_inventory + deviation)
    ghost_price = 0
    if total_deviation > 0:
        ghost_price = 120 + 0.7 * 120 / np.sqrt(100) * np.sqrt(np.abs(total_deviation))
    else:
        ghost_price = max(1, 120 - (1.6 * 120) / 100 * np.abs(total_deviation))
    return np.round(ghost_price, 4)

# MELON: log below 0.2, sq above 3.6
def melon_adjusted_price(curr_inventory, deviation):
    total_deviation = (10000 - curr_inventory + deviation)
    ghost_price = 0
    if total_deviation > 0:
        ghost_price = 250 + 0.2 * 250 / np.log(1 + 300) * np.log(1 + np.abs(total_deviation))
    else:
        ghost_price = max(1, 250 - 3.6 * 250 / np.power(300, 2) * np.power(np.abs(total_deviation), 2))
    return np.round(ghost_price, 4)

# EGG: hinge below 0.4, log above 0.2
def egg_adjusted_price(curr_inventory, deviation):
    total_deviation = (10000 - curr_inventory + deviation)
    ghost_price = 0
    if total_deviation > 0:
        ghost_price = 50 + 0.4 * 50 / hinge(332, 332) * hinge(np.abs(total_deviation), 332)
    else:
        ghost_price = max(1, 50 - (0.2 * 50) / np.log(1 + 332) * np.log(1 + np.abs(total_deviation)))
    return np.round(ghost_price, 4)

# MILK: sqrt below 0.6, linear above 1.6
def milk_adjusted_price(curr_inventory, deviation):
    total_deviation = (10000 - curr_inventory + deviation)
    ghost_price = 0
    if total_deviation > 0:
        ghost_price = 160 + 0.6 * 160 / np.sqrt(122) * np.sqrt(np.abs(total_deviation))
    else:
        ghost_price = max(1, 160 - (1.6 * 160) / 122 * np.abs(total_deviation))
    return np.round(ghost_price, 4)

# WOOL: log below 0.2, sq above 3.2
def wool_adjusted_price(curr_inventory, deviation):
    total_deviation = (10000 - curr_inventory + deviation)
    ghost_price = 0
    if total_deviation > 0:
        ghost_price = 200 + 0.2 * 200 / np.log(1 + 105) * np.log(1 + np.abs(total_deviation))
    else:
        ghost_price = max(1, 200 - (3.2 * 200) / np.power(105, 2) * np.power(np.abs(total_deviation), 2))
    return np.round(ghost_price, 4)

# FERTILIZER: linear below 0.4, linear above 0.4
def fertilizer_adjusted_price(curr_inventory, deviation):
    total_deviation = (10000 - curr_inventory + deviation)
    ghost_price = 0
    if total_deviation > 0:
        ghost_price = 100 + 0.4 * 100 / 200 * np.abs(total_deviation)
    else:
        ghost_price = max(1, 100 - 0.4 * 100 / 200 * np.abs(total_deviation))
    return np.round(ghost_price, 4)