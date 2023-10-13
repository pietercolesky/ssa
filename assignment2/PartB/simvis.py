from itertools import combinations

import numpy as np
import numpy.linalg as la
import pandas as pd

from plot import plot_sky_model, plot_visibilities, plot_antennas_2D, plot_uv
from read import read_config, read_enu_coords, read_sky_model_df
from utils import delta, to_deg, calculate_visibilities


def get_skymodel(df):
    sigma = 0.1
    plane_size = 10
    pixel_count = 500
    lm_range = np.linspace(-plane_size / 2, plane_size / 2, pixel_count)
    l, m = np.meshgrid(lm_range, lm_range[::-1])

    l_deg = df["l"].map(to_deg).values
    m_deg = df["m"].map(to_deg).values
    model = delta(
        df['flux'].values[:, np.newaxis, np.newaxis],
        l, l_deg[:, np.newaxis, np.newaxis],
        m, m_deg[:, np.newaxis, np.newaxis], sigma
    )

    plot_sky_model(model, plane_size)
    return model


def get_visibilities(df):
    u_range = np.linspace(-4000, 4000, 500)
    v_range = np.linspace(-3000, 3000, 500)
    u, v = np.meshgrid(u_range, v_range[::-1])

    visibilities = calculate_visibilities(
        df['flux'].values[:, np.newaxis, np.newaxis],
        u, df['l'].values[:, np.newaxis, np.newaxis],
        v, df['m'].values[:, np.newaxis, np.newaxis]
    )

    u_min, u_max, v_min, v_max = u_range.min(), u_range.max(), v_range.min(), v_range.max()
    plot_visibilities(visibilities, u_min, u_max, v_min, v_max)

    return visibilities


def get_xyz(baseline, lat):
    D, A, E = baseline["D"], baseline["A"], baseline["E"]
    return D * np.array(
        [
            [np.cos(lat) * np.sin(E) - np.sin(lat) * np.cos(E) * np.cos(A)],
            [np.cos(E) * np.sin(A)],
            [np.sin(lat) * np.sin(E) + np.cos(lat) * np.cos(E) * np.cos(A)]
        ]
    )


def get_uvw(xyz, hour_angle_range, dec, wavelength):
    return np.array(list(map(lambda h: np.array(
        [
            [np.sin(h), np.cos(h), 0],
            [-np.sin(dec) * np.cos(h), np.sin(dec) * np.sin(h), np.cos(dec)],
            [np.cos(dec), -np.cos(dec) * np.sin(h), np.sin(dec)]
        ]
    ).dot(xyz) / wavelength, hour_angle_range)))


config = read_config()
enu_coords = read_enu_coords()
skymodel_df = read_sky_model_df()

skymodel = get_skymodel(skymodel_df)
visibilities = get_visibilities(skymodel_df)

plot_antennas_2D(enu_coords)

num_steps = config["num_steps"]
hour_angle_range = np.linspace(config["hour_angle_range"][0], config["hour_angle_range"][1], num_steps)

wavelength = 299792458 / (config['obs_freq'] * 10 ** 9)

baselines = list(combinations(enu_coords, 2))
baselines_df = pd.DataFrame(baselines, columns=["X1", "X2"])
baselines_df["b"] = baselines_df["X2"] - baselines_df["X1"]
baselines_df["D"] = baselines_df["b"].map(la.norm)
baselines_df["A"] = baselines_df["b"].map(lambda b: np.arctan(b[0] / b[1]))
baselines_df["E"] = baselines_df["b"].map(lambda b: np.arctan(b[2] / la.norm(b[:2])))
baselines_df["XYZ"] = baselines_df[["D", "A", "E"]].apply(lambda baseline: get_xyz(baseline, config["lat"]), axis=1)
baselines_df["UVW"] = baselines_df["XYZ"].map(lambda xyz: get_uvw(xyz, hour_angle_range, config["dec"], wavelength))

plot_uv(baselines_df["UVW"].values)
