import numpy as np


def to_rad(x):
    return x * (np.pi / 180)


def to_deg(x):
    return x * (180 / np.pi)


def hours_to_rad(hours):
    return (np.pi / 12) * hours


def scale(x, min_val=0, max_val=255):
    x_min = np.min(x)
    x_max = np.max(x)
    scaled_x = min_val + (max_val - min_val) * (x - x_min) / (x_max - x_min)
    return scaled_x


def abs_log_scale(x):
    return np.log(np.abs(x) + 1)
