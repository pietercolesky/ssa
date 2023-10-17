import numpy as np


def to_rad(x):
    return x * (np.pi / 180)


def to_deg(x):
    return x * (180 / np.pi)


def deg_to_rad(degrees, arcmin=0, arcsecs=0):
    return to_rad(degrees + (arcmin / 60) + (arcsecs / 3600))


def hours_to_rad(hours, mins=0, secs=0):
    return (np.pi / 12) * (hours + (mins / 60) + (secs / 3600))
