import numpy as np


def to_rad(x):
    return x * (np.pi / 180)


def to_deg(x):
    return x * (180 / np.pi)


def deg_to_rad(degrees, arcmin=0, arcsecs=0):
    return to_rad(degrees + (arcmin / 60) + (arcsecs / 3600))


def hours_to_rad(hours, mins=0, secs=0):
    return (np.pi / 12) * (hours + (mins / 60) + (secs / 3600))


def calculate_l(src):
    dec = src["dec"]
    ra_delta = src["ra_delta"]
    return np.cos(dec) * np.sin(ra_delta)


def calculate_m(src, dec_0):
    dec = src["dec"]
    ra_delta = src["ra_delta"]
    return np.sin(dec) * np.cos(dec_0) - np.cos(dec) * np.sin(dec_0) * np.cos(ra_delta)


def delta(flux, l, l0, m, m0, sigma):
    return np.sum(flux * np.exp(-((l - l0) ** 2 + (m - m0) ** 2) / (2 * sigma ** 2)), axis=0)


def calculate_visibilities(flux, u, l0, v, m0):
    return np.sum(flux * np.exp(-2 * np.pi * (l0 * u + m0 * v) * 1j), axis=0)


def calculate_xyz(lat, d, a, e):
    return d * np.array(
        [
            [np.cos(lat) * np.sin(e) - np.sin(lat) * np.cos(e) * np.cos(a)],
            [np.cos(e) * np.sin(a)],
            [np.sin(lat) * np.sin(e) + np.cos(lat) * np.cos(e) * np.cos(a)]
        ]
    )


def calculate_uvw(h, dec, xyz, wavelength):
    return np.array(
        [
            [np.sin(h), np.cos(h), 0],
            [-np.sin(dec) * np.cos(h), np.sin(dec) * np.sin(h), np.cos(dec)],
            [np.cos(dec), -np.cos(dec) * np.sin(h), np.sin(dec)]
        ]
    ).dot(xyz) / wavelength
