from json import load
from pathlib import Path

import numpy as np
from pandas import read_csv

from utils import deg_to_rad, hours_to_rad

input_dir = Path(__file__).parent / 'input'


def _extract_parts(x):
    a, b, c = map(float, x.split(":"))
    is_negative = a < 0
    return a, -b if is_negative else b, -c if is_negative else c


def _calculate_l(src):
    dec = src["dec"]
    ra_delta = src["ra_delta"]
    return np.cos(dec) * np.sin(ra_delta)


def _calculate_m(src, dec_0):
    dec = src["dec"]
    ra_delta = src["ra_delta"]
    return np.sin(dec) * np.cos(dec_0) - np.cos(dec) * np.sin(dec_0) * np.cos(ra_delta)


def read_enu_coords():
    return np.loadtxt(input_dir / 'antennae.txt')


def read_config():
    with open(input_dir / 'configurations.json', "rb") as file:
        config = load(file)
    config['lat'] = deg_to_rad(*_extract_parts(config['lat']))
    config['ra'] = hours_to_rad(*_extract_parts(config['ra']))
    config['dec'] = deg_to_rad(*_extract_parts(config['dec']))
    config['hour_angle_range'] = list(map(lambda h: hours_to_rad(*_extract_parts(h)), config['hour_angle_range']))
    return config


def read_sky_model_df():
    df = read_csv(input_dir / 'skymodel.csv', usecols=["name", "flux", "ra", "dec"])
    df["ra"] = df["ra"].map(lambda ra: hours_to_rad(*_extract_parts(ra)))
    df["dec"] = df["dec"].map(lambda dec: deg_to_rad(*_extract_parts(dec)))
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
    config["cell_size"] = deg_to_rad(*_extract_parts(config["cell_size"]))
    return config

