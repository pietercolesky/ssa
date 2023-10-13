from pathlib import Path
from json import load

import numpy as np
from pandas import read_csv

from utils import deg_to_rad, hours_to_rad, calculate_l, calculate_m


input_dir = Path(__file__).parent / 'input'


def read_enu_coords():
    return np.loadtxt(input_dir / 'antennae.txt')


def read_config():
    with open(input_dir / 'configurations.json', "rb") as file:
        config = load(file)
    config['lat'] = deg_to_rad(*config['lat'])
    config['ra'] = hours_to_rad(*config['ra'])
    config['dec'] = deg_to_rad(*config['dec'])
    config['hour_angle_range'] = list(map(hours_to_rad, config['hour_angle_range']))
    return config


def read_sky_model_df():
    df = read_csv(input_dir / 'skymodel.csv', usecols=["name", "flux", "ra", "dec"])
    df["ra"] = df["ra"].map(lambda ra: hours_to_rad(*list(map(float, ra.split(',')))))
    df["dec"] = df["dec"].map(lambda dec: deg_to_rad(*list(map(float, dec.split(',')))))
    df["ra_delta"] = df["ra"] - df[df["name"] == "Papino"]["ra"].values[0]
    df["l"] = df[["dec", "ra_delta"]].apply(calculate_l, axis=1)
    df["m"] = df[["dec", "ra_delta"]].apply(lambda row: calculate_m(row, df[df["name"] == "Papino"]["dec"].values[0]),
                                            axis=1)
    return df
