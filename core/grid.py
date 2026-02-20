# 2 columns × 3 rows
#
# 0 1
# 2 3
# 4 5

LEFT = {
    0: None, 1: 0,
    2: None, 3: 2,
    4: None, 5: 4,
}

RIGHT = {
    0: 1, 1: None,
    2: 3, 3: None,
    4: 5, 5: None,
}

ABOVE = {
    0: None, 1: None,
    2: 0,    3: 1,
    4: 2,    5: 3,
}

UNDER = {
    0: 2, 1: 3,
    2: 4, 3: 5,
    4: None, 5: None,
}

TOUCHING = {
    0: [1, 2],
    1: [0, 3],
    2: [0, 3, 4],
    3: [1, 2, 5],
    4: [2, 5],
    5: [3, 4],
}

NOT_TOUCHING = {
    s: [i for i in range(6) if i != s and i not in TOUCHING[s]]
    for s in range(6)
}