from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, Any

import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
import pandas as pd

from read import read_config, read_enu_coords, read_sky_model_df
from utils import to_deg


@dataclass
class SimVis:
    sigma: float = 0.1
    plane_size: int = 10
    pixel_count: int = 500
    u_range: np.ndarray = np.linspace(-4000, 4000, 500)
    v_range: np.ndarray = np.linspace(3000, -3000, 500)

    config: Dict[str, Any] = field(default_factory=lambda: read_config())
    enu_coords: np.ndarray = field(default_factory=lambda: read_enu_coords())
    skymodel_df: pd.DataFrame = field(default_factory=lambda: read_sky_model_df())

    def __post_init__(self):
        self.obs_wavelength = 299792458 / (self.config['obs_freq'] * 10 ** 9)
        self.hour_angle_range = np.linspace(self.config["hour_angle_range"][0], self.config["hour_angle_range"][1],
                                            self.config["num_steps"])

        uv = np.vstack(self.baselines["UVW"].values)[:, :2]
        self.uv = np.concatenate([uv, -uv])

    @property
    def skymodel(self):
        lm_range = np.linspace(-self.plane_size / 2, self.plane_size / 2, self.pixel_count)
        l, m = np.meshgrid(lm_range, lm_range[::-1])

        l_deg = self.skymodel_df["l"].map(to_deg).values
        m_deg = self.skymodel_df["m"].map(to_deg).values
        model = self._delta(
            self.skymodel_df['flux'].values[:, np.newaxis, np.newaxis],
            l, l_deg[:, np.newaxis, np.newaxis],
            m, m_deg[:, np.newaxis, np.newaxis], self.sigma
        )
        return model

    @property
    def visibilities(self):
        u, v = np.meshgrid(self.u_range, self.v_range)

        vis = self._calculate_visibilities(
            self.skymodel_df['flux'].values[:, np.newaxis, np.newaxis],
            u, self.skymodel_df['l'].values[:, np.newaxis, np.newaxis],
            v, self.skymodel_df['m'].values[:, np.newaxis, np.newaxis]
        )

        return vis

    @property
    def baselines(self):
        baselines = list(combinations(self.enu_coords, 2))
        baselines_df = pd.DataFrame(baselines, columns=["X1", "X2"])
        baselines_df["b"] = baselines_df["X2"] - baselines_df["X1"]
        baselines_df["D"] = baselines_df["b"].map(la.norm)
        baselines_df["A"] = baselines_df["b"].map(lambda b: np.arctan(b[0] / b[1]))
        baselines_df["E"] = baselines_df["b"].map(lambda b: np.arctan(b[2] / la.norm(b[:2])))
        baselines_df["XYZ"] = baselines_df[["D", "A", "E"]].apply(
            lambda baseline: self._get_xyz(baseline),axis=1
        )
        baselines_df["UVW"] = baselines_df["XYZ"].map(
            lambda xyz: self._get_uvw(xyz, self.hour_angle_range, self.config["dec"], self.obs_wavelength)
        )
        return baselines_df

    def _delta(self, flux, l, l0, m, m0, sigma):
        return np.sum(flux * np.exp(-((l - l0) ** 2 + (m - m0) ** 2) / (2 * sigma ** 2)), axis=0)

    def _calculate_visibilities(self, flux, u, l0, v, m0):
        return np.sum(flux * np.exp(-2 * np.pi * (l0 * u + m0 * v) * 1j), axis=0)

    def _get_xyz(self, baseline):
        L = self.config["lat"]
        D, A, E = baseline["D"], baseline["A"], baseline["E"]
        return D * np.array([
            np.cos(L) * np.sin(E) - np.sin(L) * np.cos(E) * np.cos(A),
            np.cos(E) * np.sin(A),
            np.sin(L) * np.sin(E) + np.cos(L) * np.cos(E) * np.cos(A)
        ])

    def _get_uvw(self, xyz, h_range, dec, wavelength):
        return list(map(
            lambda h: np.array([
                [np.sin(h), np.cos(h), 0],
                [-np.sin(dec) * np.cos(h), np.sin(dec) * np.sin(h), np.cos(dec)],
                [np.cos(dec), -np.cos(dec) * np.sin(h), np.sin(dec)]
            ]).dot(xyz) / wavelength, h_range
        ))

    def _search(self, coords, col, min_val, max_val):
        uv_sorted = coords[coords[:, col].argsort()]
        start_idx = np.searchsorted(uv_sorted[:, col], min_val, side='left')
        end_idx = np.searchsorted(uv_sorted[:, col], max_val, side='right')
        return uv_sorted[start_idx:end_idx]

    def plot_sky_model(self):
        plt.imshow(self.skymodel, extent=(-self.plane_size / 2, self.plane_size / 2, -self.plane_size / 2,
                                          self.plane_size / 2), cmap='jet')
        plt.colorbar(label='Brightness')
        plt.xlabel(r'l ($^{\circ}$)')
        plt.ylabel(r'm ($^{\circ}$)')
        plt.title('Skymodel (in brightness)')
        plt.show()

    def plot_visibilities(self):
        u_min, u_max, v_min, v_max = self.u_range.min(), self.u_range.max(), self.v_range.min(), self.v_range.max()

        fig, axes = plt.subplots(1, 2, figsize=(20, 6))

        im1 = axes[0].imshow(self.visibilities.real, extent=(u_min, u_max, v_min, v_max), cmap="jet")
        axes[0].set_xlabel(r"u (rad$^{-1})$")
        axes[0].set_ylabel(r"v (rad$^{-1})$")
        axes[0].set_title('Real part of Visibilities')
        cbar = fig.colorbar(im1, ax=axes[0], orientation='vertical')
        cbar.set_label('Magnitude')

        im2 = axes[1].imshow(self.visibilities.imag, extent=(u_min, u_max, v_min, v_max), cmap='jet')
        axes[1].set_xlabel(r"u (rad$^{-1})$")
        axes[1].set_ylabel(r"v (rad$^{-1})$")
        axes[1].set_title('Imaginary part of Visibilities')
        cbar = fig.colorbar(im2, ax=axes[1], orientation='vertical')
        cbar.set_label('Magnitude')

        plt.tight_layout()
        plt.show()

    def plot_antennas_2D(self):
        E = self.enu_coords[:, 0]
        N = self.enu_coords[:, 1]

        max_E = np.max(np.abs(E))
        max_N = np.max(np.abs(N))

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(E, N)

        ax.set_xlabel("W-E (m)")
        ax.set_ylabel("S-N (m)")

        ax.set_xlim([-max_E - (1 / 3) * max_E, max_E + (1 / 3) * max_E])
        ax.set_ylim([-max_N - (1 / 3) * max_N, max_N + (1 / 3) * max_N])

        plt.show()

    def plot_uv(self):
        mid = len(self.uv) // 2
        plt.scatter(self.uv[:mid, 0], self.uv[:mid, 1], s=2, c="b", label="Baselines")
        plt.scatter(self.uv[mid:, 0], self.uv[mid:, 1], s=2, c="r", label="Conjugate Baselines")

        plt.xlabel(r"u (rad$^{-1})$")
        plt.ylabel(r"v (rad$^{-1})$")
        plt.title('uv-tracks')
        plt.legend(["Baselines", "Conjugate Baselines"])
        plt.show()


if __name__ == "__main__":
    simvis = SimVis()

    print("Baselines:")
    print(simvis.baselines.b)

    print()
    print("Distance:")
    print(simvis.baselines.D)

    print()
    print("Azimuth:")
    print(simvis.baselines.A)

    print()
    print("Elevation:")
    print(simvis.baselines.E)

    print()
    print("XYZ:")
    print(simvis.baselines.XYZ)

    print()
    print("UV:")
    print(simvis.uv.shape)
    print(simvis.uv)

    # Get uv-coords with u in range [0, 1)
    pts = simvis._search(simvis.uv, 0, 0, 1)
    print()
    print("BINARY SEARCH")
    print(pts.shape)
    if len(pts) > 0:
        print(f"u min: {pts[:, 0].min()}")
        print(f"u max: {pts[:, 0].max()}")
        print(f"v min: {pts[:, 1].min()}")
        print(f"v max: {pts[:, 1].max()}")

        # Get uv-coords of the points above with v in range [-416, -415)
        filtered = simvis._search(pts, 1, -416, -415)
        print(filtered.shape)
        if len(filtered) > 0:
            print(f"u min: {filtered[:, 0].min()}")
            print(f"u max: {filtered[:, 0].max()}")
            print(f"v min: {filtered[:, 1].min()}")
            print(f"v max: {filtered[:, 1].max()}")

    simvis.plot_antennas_2D()
    simvis.plot_sky_model()
    simvis.plot_visibilities()
    simvis.plot_uv()
