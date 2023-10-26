from dataclasses import dataclass, field
from itertools import combinations
from json import dumps
from pathlib import Path, PosixPath
from typing import Dict, Any

import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
import pandas as pd

from read import read_config, read_enu_coords, read_sky_model_df, read_img_config, input_dir
from utils import to_deg, scale, abs_log_scale


def calculate_vis(flux, l0, m0, u, v, axis=None):
    return np.sum(flux * np.exp(-2 * np.pi * (l0 * u + m0 * v) * 1j), axis=axis)


@dataclass
class SimVis:
    results_dir: PosixPath = Path(__file__).parent / 'results'
    config: Dict[str, Any] = field(default_factory=lambda: read_config())
    img_conf: Dict[str, Any] = field(default_factory=lambda: read_img_config())
    enu_coords: np.ndarray[float] = field(default_factory=lambda: read_enu_coords())
    skymodel_df: pd.DataFrame = field(default_factory=lambda: read_sky_model_df())

    def __post_init__(self):
        self._create_results_directory()

        self.obs_wavelength = self._calculate_wavelength()
        self.hour_angle_range = self._get_hour_angle_range()
        self.lm_min, self.lm_max, self.uv_min, self.uv_max = self._get_min_max()

        self.skymodel = self._get_skymodel()
        self.baselines = self._get_baselines()
        self.uv = self._get_uv()
        self.scaled_uv = self._scale_uv()
        self.gridded_uv, self.gridded_vis = self._grid()
        self.psf = self._get_psf()
        self.obs_img = self._get_restored_img()

    def _create_results_directory(self):
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def _calculate_wavelength(self):
        return 299792458 / self.config['obs_freq']

    def _get_hour_angle_range(self):
        return np.linspace(*self.config["hour_angle_range"], self.config["num_steps"])

    def _get_min_max(self):
        N = self.img_conf["N"]
        cell_size = self.img_conf["cell_size"]
        lm_max = to_deg(0.5 * N * cell_size)
        uv_max = 0.5 * N * (1 / (N * cell_size))
        return -lm_max, lm_max, -uv_max, uv_max

    def _get_skymodel(self, sigma=0.1):
        N = self.img_conf["N"]

        lm_range = np.linspace(self.lm_min, self.lm_max, N)
        l, m = np.meshgrid(lm_range, lm_range[::-1])

        l_deg = self.skymodel_df["l"].map(to_deg).values
        m_deg = self.skymodel_df["m"].map(to_deg).values

        flux = self.skymodel_df['flux'].values[:, np.newaxis, np.newaxis]
        l0 = l_deg[:, np.newaxis, np.newaxis]
        m0 = m_deg[:, np.newaxis, np.newaxis]
        return np.sum(flux * np.exp(-((l - l0) ** 2 + (m - m0) ** 2) / (2 * sigma ** 2)), axis=0)

    def _get_xyz(self, baseline):
        L = self.config["lat"]
        D, A, E = baseline["D"], baseline["A"], baseline["E"]
        return D * np.array([
            np.cos(L) * np.sin(E) - np.sin(L) * np.cos(E) * np.cos(A),
            np.cos(E) * np.sin(A),
            np.sin(L) * np.sin(E) + np.cos(L) * np.cos(E) * np.cos(A)
        ])

    def _get_uvw(self, xyz):
        dec = self.config["dec"]
        return list(map(
            lambda h: np.array([
                [np.sin(h), np.cos(h), 0],
                [-np.sin(dec) * np.cos(h), np.sin(dec) * np.sin(h), np.cos(dec)],
                [np.cos(dec), -np.cos(dec) * np.sin(h), np.sin(dec)]
            ]).dot(xyz) / self.obs_wavelength, self.hour_angle_range
        ))

    def _get_baselines(self):
        baselines = list(combinations(enumerate(self.enu_coords, 1), 2))
        baselines_df = pd.DataFrame(baselines, columns=["A1", "A2"])
        baselines_df["name"] = baselines_df.apply(lambda row: f'{row["A1"][0]}_{row["A2"][0]}', axis=1)
        baselines_df["conj_name"] = baselines_df.name.map(lambda name: '_'.join(reversed(name.split('_'))))
        baselines_df["A1"] = baselines_df["A1"].apply(lambda x: x[1])
        baselines_df["A2"] = baselines_df["A2"].apply(lambda x: x[1])
        baselines_df["b"] = baselines_df["A2"] - baselines_df["A1"]
        baselines_df["D"] = baselines_df["b"].map(la.norm)
        baselines_df["A"] = baselines_df["b"].map(lambda b: np.arctan(b[0] / b[1]))
        baselines_df["E"] = baselines_df["b"].map(lambda b: np.arctan(b[2] / la.norm(b[:2])))
        baselines_df["XYZ"] = baselines_df[["D", "A", "E"]].apply(
            lambda baseline: self._get_xyz(baseline), axis=1
        )
        baselines_df["UVW"] = baselines_df["XYZ"].map(
            lambda xyz: self._get_uvw(xyz)
        )
        return baselines_df

    def _get_uv(self):
        uv = np.vstack(self.baselines["UVW"].values)[:, :2]
        return np.concatenate([uv, -uv])

    def _scale_uv(self):
        N = self.img_conf["N"]
        scaled_uv = np.copy(self.uv)
        scaled_uv /= (2 * self.uv_max / N)
        scaled_uv += N / 2
        return np.round(scaled_uv).astype(int)

    def _get_sources(self):
        sources = self.skymodel_df[["flux", "l", "m"]].values
        return sources[:, 0], sources[:, 1], sources[:, 2]

    def _filter_valid_uv(self):
        N = self.img_conf["N"]

        is_valid = ((0 <= self.scaled_uv[:, 0]) & (self.scaled_uv[:, 0] < N) &
                    (0 <= self.scaled_uv[:, 1]) & (self.scaled_uv[:, 1] < N))

        ind = self.scaled_uv[is_valid]
        uv = self.uv[is_valid][:, np.newaxis, :]

        return ind[:, 0], -ind[:, 1], uv[:, :, 0], uv[:, :, 1]

    def _grid(self):
        N = self.img_conf["N"]

        gridded_uv = np.zeros((N, N), dtype=float)
        gridded_vis = np.zeros((N, N), dtype=complex)

        flux, l, m = self._get_sources()
        x, y, u, v = self._filter_valid_uv()

        np.add.at(gridded_uv, (y, x), 1)
        np.add.at(gridded_vis, (y, x), calculate_vis(flux, l, m, u, v, 1))

        if self.img_conf["weighting"] == "uniform":
            gridded_vis[y, x] /= gridded_uv[y, x]
            gridded_uv[y, x] /= gridded_uv[y, x]

        return gridded_uv, gridded_vis

    def _get_psf(self):
        return np.abs(np.fft.fftshift(np.fft.ifft2(self.gridded_uv)))

    def _get_restored_img(self):
        obs_img = np.abs(np.fft.fftshift(np.fft.ifft2(self.gridded_vis)))
        return scale(obs_img, max_val=self.skymodel_df["flux"].max())

    def print_info(self):
        print(f"Configurations: {dumps(self.config, indent=4)}")

        print(f"\nImager configurations: {dumps(self.img_conf, indent=4)}")

        print(f"\nSkymodel:")
        print(self.skymodel_df.to_string())

        print("\nENU Coordinates:")
        print(self.enu_coords)

        print(f"\nLM-range: [{self.lm_min}, {self.lm_max}]")
        print(f"UV-range: [{self.uv_min}, {self.uv_max}]")

        print(f"\nObservational wavelength: {self.obs_wavelength}m")

        print("\nBaselines:")
        print(self.baselines[["name", "A1", "A2"]].to_string())
        print(self.baselines[["name", "b", "D", "A", "E", "XYZ"]].to_string())

        print("\nUV:")
        print(self.uv)

    def plot_sky_model(self):
        print("\nPlotting sky model")
        plt.imshow(self.skymodel, extent=(self.lm_min, self.lm_max, self.lm_min, self.lm_max), cmap='jet')
        l_rads = self.skymodel_df["l"].values
        m_rads = self.skymodel_df["m"].values
        for l_rad, m_rad in zip(l_rads, m_rads):
            src = self.skymodel_df[(self.skymodel_df.l == l_rad) & (self.skymodel_df.m == m_rad)]
            name = src.name.values[0]
            flux = src.flux.values[0]
            l_deg = to_deg(l_rad)
            m_deg = to_deg(m_rad)
            plt.annotate(f"{name}: {flux}", xy=(l_deg, m_deg), xytext=(0.95 * self.lm_max, m_deg),
                         fontsize='medium', ha='right', va='center', color='white',
                         arrowprops=dict(facecolor='white', edgecolor='white', arrowstyle='->'))
        plt.colorbar(label='Brightness')
        plt.xlabel(r'l ($^{\circ}$)')
        plt.ylabel(r'm ($^{\circ}$)')
        plt.title('Skymodel (in brightness)')
        plt.savefig(self.results_dir / "skymodel.png")
        plt.close()
        print("Done.")

    def plot_antennas_2D(self):
        print("\nPlotting antennae")
        E = self.enu_coords[:, 0]
        N = self.enu_coords[:, 1]

        max_E = np.max(np.abs(E))
        max_N = np.max(np.abs(N))

        marker = plt.imread(input_dir / 'telescope.png')
        marker_size = 10

        fig = plt.figure()
        ax = fig.add_subplot(111)

        for i in range(len(E)):
            e_min = float(E[i] - marker_size / 2)
            e_max = float(E[i] + marker_size / 2)
            n_min = float(N[i] - marker_size / 2)
            n_max = float(N[i] + marker_size / 2)
            ax.imshow(marker, extent=(e_min, e_max, n_min, n_max))

        ax.set_xlabel("W-E (m)")
        ax.set_ylabel("S-N (m)")

        ax.set_xlim([-max_E - (1 / 3) * max_E, max_E + (1 / 3) * max_E])
        ax.set_ylim([-max_N - (1 / 3) * max_N, max_N + (1 / 3) * max_N])

        plt.savefig(self.results_dir / "antennae.png")
        plt.close()
        print("Done.")

    def plot_uv(self):
        print("\nPlotting UV tracks")
        mid = len(self.uv) // 2
        plt.figure(figsize=(6, 5))
        plt.scatter(self.uv[:mid, 0], self.uv[:mid, 1], s=2, c="b", label="Baselines")
        plt.scatter(self.uv[mid:, 0], self.uv[mid:, 1], s=2, c="r", label="Conjugate Baselines")
        plt.xlabel(r"u (rad$^{-1})$")
        plt.ylabel(r"v (rad$^{-1})$")
        plt.title('uv-tracks')
        plt.legend(["Baselines", "Conjugate Baselines"])
        plt.savefig(self.results_dir / "uv-coverage.png")
        plt.close()
        print("Done.")

    def plot_psf(self):
        print("\nPlotting PSF")
        plt.imshow(self.psf, cmap="jet", extent=(self.lm_min, self.lm_max, self.lm_min, self.lm_max))
        plt.xlabel(r"l ($^{\circ})$")
        plt.ylabel(r"m ($^{\circ})$")
        plt.title('PSF')
        plt.colorbar()
        plt.savefig(self.results_dir / "psf.png")
        plt.close()
        print("Done.")

    def plot_gridded_vis(self):
        print("\nPlotting gridded visibilities")
        plt.imshow(abs_log_scale(self.gridded_vis), extent=(self.uv_min, self.uv_max, self.uv_min, self.uv_max),
                   cmap="jet")
        plt.xlabel(r"u (rad$^{-1})$")
        plt.ylabel(r"v (rad$^{-1})$")
        plt.title('Gridded Visibilities (Amplitude)')
        plt.colorbar(label="Magnitude", orientation='vertical')
        plt.savefig(self.results_dir / "gridded_amp.png")
        plt.close()

        plt.imshow(np.angle(self.gridded_vis), extent=(self.uv_min, self.uv_max, self.uv_min, self.uv_max), cmap="jet")
        plt.xlabel(r"u (rad$^{-1})$")
        plt.ylabel(r"v (rad$^{-1})$")
        plt.title('Gridded Visibilities (Phase)')
        plt.colorbar(label="Magnitude", orientation='vertical')
        plt.savefig(self.results_dir / "gridded_phase.png")
        plt.close()
        print("Done.")

    def plot_obs_img(self):
        print("\nPlotting observed image")
        plt.imshow(self.obs_img, cmap="jet", extent=(self.lm_min, self.lm_max, self.lm_min, self.lm_max))
        plt.xlabel(r"l ($^{\circ})$")
        plt.ylabel(r"m ($^{\circ})$")
        plt.title('Observed Image')
        plt.colorbar()
        plt.savefig(self.results_dir / "image.png")
        plt.close()
        print("Done.")

    def _get_vis(self, u_range, v_range):
        u, v = np.meshgrid(u_range, v_range[::-1])
        flux = self.skymodel_df['flux'].values[:, np.newaxis, np.newaxis]
        l0 = self.skymodel_df['l'].values[:, np.newaxis, np.newaxis]
        m0 = self.skymodel_df['m'].values[:, np.newaxis, np.newaxis]
        return calculate_vis(flux, l0, m0, u, v, 0)

    def plot_vis(self):
        print("\nPlotting visibilities")
        u_max = np.max(self.uv[:, 0])
        v_max = np.max(self.uv[:, 1])
        u_min = -u_max
        v_min = -v_max

        u_range = np.linspace(u_min, u_max, self.config["num_steps"])
        v_range = np.linspace(v_min, v_max, self.config["num_steps"])

        vis = self._get_vis(u_range, v_range)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 8))
        im1 = ax1.imshow(vis.real, extent=(u_min, u_max, v_min, v_max), cmap="jet", aspect="auto")
        ax1.set_title('Real part of Visibilities')
        ax1.set_xlabel(r"u (rad$^{-1})$")
        ax1.set_ylabel(r"v (rad$^{-1})$")
        fig.colorbar(im1, ax=ax1, label='Magnitude')

        im2 = ax2.imshow(vis.imag, extent=(u_min, u_max, v_min, v_max), cmap="jet", aspect="auto")
        ax2.set_title('Imaginary part of Visibilities')
        ax2.set_xlabel(r"u (rad$^{-1})$")
        ax2.set_ylabel(r"v (rad$^{-1})$")
        fig.colorbar(im2, ax=ax2, label='Magnitude')

        plt.tight_layout()
        plt.savefig(self.results_dir / "vis.png")
        plt.close()
        print("Done.")

    def _get_baseline_vis(self, name):
        baselines = np.concatenate([self.baselines.name.values, self.baselines.conj_name.values])
        if name not in baselines:
            return []

        flux, l, m = self._get_sources()

        uv = np.stack(
            self.baselines[(self.baselines.name == name) | (self.baselines.conj_name == name)]["UVW"].values
        )[0, :, np.newaxis, :2]

        return calculate_vis(flux, l, m, uv[:, :, 0], uv[:, :, 1], 1)

    def plot_vis_vs_hour_angle(self):
        print("\nPlotting baseline visibilities vs hour angle")
        baselines = self.img_conf["baselines"]

        for baseline in baselines:
            vis = self._get_baseline_vis(baseline)
            if len(vis) == 0:
                print(f"Baseline {baseline} not found!")
                continue

            hour_angle_range = to_deg(self.hour_angle_range) / 15

            plt.plot(hour_angle_range, vis.real)
            plt.xlabel("Hour Angle (H)")
            plt.ylabel("Magnitude")
            plt.title(f'Baseline {baseline} Visibilities (Real)')
            plt.tight_layout()
            plt.savefig(self.results_dir / f"vis_b_{baseline}_real.png")
            plt.close()

            plt.plot(hour_angle_range, vis.imag)
            plt.xlabel("Hour Angle (H)")
            plt.ylabel("Magnitude")
            plt.title(F'Baseline {baseline} Visibilities (Imaginary)')
            plt.tight_layout()
            plt.savefig(self.results_dir / f"vis_b_{baseline}_imag.png")
            plt.close()
        print("Done.")


if __name__ == "__main__":
    simvis = SimVis()

    simvis.print_info()

    simvis.plot_antennas_2D()
    simvis.plot_sky_model()
    simvis.plot_uv()
    simvis.plot_vis()
    simvis.plot_vis_vs_hour_angle()
    simvis.plot_psf()
    simvis.plot_gridded_vis()
    simvis.plot_obs_img()
