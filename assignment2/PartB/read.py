from json import load
from pathlib import Path

import numpy as np
from pandas import read_csv

from utils import hours_to_rad, to_rad

input_dir = Path(__file__).parent / 'input'


def _get_total(x):
    is_negative = x.startswith("-")
    a, b, c = map(lambda val: abs(float(val)), x.split(":"))
    magnitude = a + (b / 60) + (c / 3600)
    return -magnitude if is_negative else magnitude


def _calculate_l(src):
    dec = src["dec"]
    ra_delta = src["ra_delta"]
    return np.cos(dec) * np.sin(ra_delta)


def _calculate_m(src, dec_0):
    dec = src["dec"]
    ra_delta = src["ra_delta"]
    return np.sin(dec) * np.cos(dec_0) - np.cos(dec) * np.sin(dec_0) * np.cos(ra_delta)


def read_enu_coords():
    return np.loadtxt(input_dir / 'antennae.txt', dtype=float)


def read_config():
    with open(input_dir / 'configurations.json', "rb") as file:
        config = load(file)
    config['obs_freq'] *= 10 ** 9
    config['lat'] = to_rad(_get_total(config['lat']))
    config['ra'] = hours_to_rad(_get_total(config['ra']))
    config['dec'] = to_rad(_get_total(config['dec']))
    config['hour_angle_range'] = list(map(lambda h: hours_to_rad(_get_total(h)), config['hour_angle_range']))
    return config


def read_sky_model_df():
    df = read_csv(input_dir / 'skymodel.csv', usecols=["name", "flux", "ra", "dec"])
    df["ra"] = df["ra"].map(lambda ra: hours_to_rad(_get_total(ra)))
    df["dec"] = df["dec"].map(lambda dec: to_rad(_get_total(dec)))
    field_center = df.iloc[0]
    ra_0 = field_center["ra"]
    dec_0 = field_center["dec"]
    df["ra_delta"] = df["ra"] - ra_0
    df["l"] = df[["dec", "ra_delta"]].apply(_calculate_l, axis=1)
    df["m"] = df[["dec", "ra_delta"]].apply(lambda src: _calculate_m(src, dec_0), axis=1)
    return df


def read_img_config():
    with open(input_dir / 'image.json', "rb") as file:
        config = load(file)
    config["cell_size"] = to_rad(_get_total(config["cell_size"]))
    return config

