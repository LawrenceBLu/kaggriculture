import numpy as np

def adjusted_price(item, curr_inventory, deviation):
    def hinge(x, T):
        u = x / T
        return u + 8 * np.power(max(0, u - 1), 2)

    cfg = {
        "WHEAT":      (25,  np.sqrt,                 1.0,               lambda x: np.log(1 + x), 5 / np.log(401)),
        "CARROT":     (35,  lambda x: hinge(x, 450), 35.0,              np.sqrt,                 24.5 / np.sqrt(450)),
        "TOMATO":     (60,  lambda x: hinge(x, 200), 24.0,              np.sqrt,                 36 / np.sqrt(200)),
        "STRAWBERRY": (120, np.sqrt,                 8.4,               lambda x: x,             1.92),
        "MELON":      (250, lambda x: np.log(1 + x), 50 / np.log(301),  lambda x: x * x,         0.01),
        "EGG":        (50,  lambda x: hinge(x, 332), 20.0,              lambda x: np.log(1 + x), 10 / np.log(333)),
        "MILK":       (160, np.sqrt,                 96 / np.sqrt(122), lambda x: x,             256 / 122),
        "WOOL":       (200, lambda x: np.log(1 + x), 40 / np.log(106),  lambda x: x * x,         640 / 11025),
        "FERTILIZER": (100, lambda x: x,             0.2,               lambda x: x,             0.2),
    }

    td = 10000 - curr_inventory + deviation
    x = abs(td)
    base, pos_fn, pos_k, neg_fn, neg_k = cfg[item.upper()]

    if td > 0:
        price = base + pos_k * pos_fn(x)
    else:
        price = max(1, base - neg_k * neg_fn(x))

    return np.round(price, 4)