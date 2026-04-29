"""
Preset floor maps for quick testing.
Each returns a 2D numpy array: 0=wall, 1=floor, 2=door/opening
"""

import numpy as np


def l_shaped_office():
    """L-shaped office with two connected zones."""
    f = np.zeros((14, 14), dtype=int)
    f[1:7, 1:10] = 1
    f[7:13, 1:6] = 1
    f[7, 3] = 2  # door
    return f, "L-Shaped Office (14×14)"


def multi_room():
    """Three rooms connected by doors."""
    f = np.zeros((16, 20), dtype=int)
    # Left room
    f[1:15, 1:8] = 1
    # Top-right room
    f[1:8, 9:19] = 1
    # Bottom-right room
    f[9:15, 9:19] = 1
    # Internal walls
    f[1:15, 8] = 0
    f[8, 9:19] = 0
    # Doors
    f[4, 8] = 2
    f[11, 8] = 2
    f[8, 14] = 2
    return f, "Multi-Room Office (16×20)"


def long_corridor():
    """Two rooms connected by a narrow corridor."""
    f = np.zeros((10, 24), dtype=int)
    # Left room
    f[1:9, 1:7] = 1
    # Corridor
    f[4:6, 7:17] = 1
    # Right room
    f[1:9, 17:23] = 1
    # Doors
    f[4, 7] = 2
    f[4, 17] = 2
    return f, "Long Corridor (10×24)"


def open_plan():
    """Large open plan with interior pillars."""
    f = np.zeros((18, 18), dtype=int)
    f[1:17, 1:17] = 1
    # Pillars (wall blocks)
    for r in [5, 11]:
        for c in [5, 11]:
            f[r, c] = 0
            f[r+1, c] = 0
            f[r, c+1] = 0
            f[r+1, c+1] = 0
    return f, "Open Plan with Pillars (18×18)"


def u_shaped():
    """U-shaped floor wrapping around a courtyard."""
    f = np.zeros((16, 16), dtype=int)
    # Left wing
    f[1:15, 1:5] = 1
    # Bottom wing
    f[10:15, 5:12] = 1
    # Right wing
    f[1:15, 11:15] = 1
    # Doors
    f[10, 3] = 2
    f[10, 12] = 2
    return f, "U-Shaped Floor (16×16)"


ALL_PRESETS = {
    "1": l_shaped_office,
    "2": multi_room,
    "3": long_corridor,
    "4": open_plan,
    "5": u_shaped,
}
