import numpy as np


def to_rad(x):
    return x * (np.pi / 180)


def to_deg(x):
    return x * (180 / np.pi)


def deg_to_rad(degrees, arcmin, arcsecs):
    return to_rad(total(degrees, arcmin, arcsecs))


def hours_to_rad(hours, mins, secs):
    return (np.pi / 12) * total(hours, mins, secs)


def total(a, b, c):
    return a + (b / 60) + (c / 3600)


def scale(x, max_val=255):
    return (x / np.max(x)) * max_val


def log_scale(x):
    return np.log(np.abs(x) + 1)
