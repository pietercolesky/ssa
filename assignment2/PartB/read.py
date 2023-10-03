from pathlib import Path
from json import load

import numpy as np
from pandas import read_csv

input_dir = Path(__file__).parent / 'input'

enu_coords = np.loadtxt(input_dir / 'antennae.txt')

with open(input_dir / 'configurations.json', "rb") as file:
    config = load(file)


def get_arr(row):
    return row["right_asc"].split(','), row["declination"].split(',')


skymodel_df = read_csv(input_dir / 'skymodel.csv', usecols=["name", "flux", "right_asc", "declination"], delimiter=";")
skymodel_df["right_asc"], skymodel_df["declination"] = skymodel_df[["right_asc", "declination"]].apply(get_arr, axis=1)

