from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path, PosixPath
from typing import Dict, Any

import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
import pandas as pd

from read import read_config, read_enu_coords, read_sky_model_df, read_img_config, input_dir
from utils import to_deg, scale, log_scale


@dataclass
class SimVis:
    results_dir: PosixPath = Path(__file__).parent / 'results'
    config: Dict[str, Any] = field(default_factory=lambda: read_config())
    img_conf: Dict[str, Any] = field(default_factory=lambda: read_img_config())
    enu_coords: np.ndarray = field(default_factory=lambda: read_enu_coords())
    skymodel_df: pd.DataFrame = field(default_factory=lambda: read_sky_model_df())

    def __post_init__(self):
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.obs_wavelength = 299792458 / (self.config['obs_freq'] * 10 ** 9)
        self.hour_angle_range = np.linspace(self.config["hour_angle_range"][0], self.config["hour_angle_range"][1],
                                            self.config["num_steps"])

        cell_size = self.img_conf["cell_size"]
        self.Nx = self.Ny = self.img_conf["num_pixels"]
        self.u_min = -0.5 * self.Nx * cell_size
        self.v_min = -0.5 * self.Ny * cell_size
        self.u_max = -self.u_min
        self.v_max = -self.v_min

        self.skymodel = self._get_skymodel()
        self.baselines = self._get_baselines()
        uv = np.vstack(self.baselines["UVW"].values)[:, :2]
        self.uv = np.concatenate([uv, -uv])
        self.scaled_uv = self._scale_uv()

        self.gridded_uv = np.zeros((self.Nx, self.Ny), dtype=float)
        self.gridded_vis = np.zeros((self.Nx, self.Ny), dtype=complex)

        self._grid()

        self.psf = np.fft.fftshift(np.fft.fft2(self.gridded_uv))

        obs_img = np.abs(np.fft.fftshift(np.fft.fft2(self.gridded_vis)))
        self.obs_img = scale(obs_img, self.skymodel_df["flux"].max())

    def _get_skymodel(self, sigma=0.1):
        plane_size = self.img_conf["plane_size"]
        lm_range = np.linspace(-plane_size / 2, plane_size / 2, self.img_conf["num_pixels"])
        l, m = np.meshgrid(lm_range, lm_range[::-1])

        l_deg = self.skymodel_df["l"].map(to_deg).values
        m_deg = self.skymodel_df["m"].map(to_deg).values

        flux = self.skymodel_df['flux'].values[:, np.newaxis, np.newaxis]
        l0 = l_deg[:, np.newaxis, np.newaxis]
        m0 = m_deg[:, np.newaxis, np.newaxis]
        return np.sum(flux * np.exp(-((l - l0) ** 2 + (m - m0) ** 2) / (2 * sigma ** 2)), axis=0)

    def _get_visibilities(self, u_range, v_range):
        u, v = np.meshgrid(u_range, v_range[::-1])
        flux = self.skymodel_df['flux'].values[:, np.newaxis, np.newaxis]
        l0 = self.skymodel_df['l'].values[:, np.newaxis, np.newaxis]
        m0 = self.skymodel_df['m'].values[:, np.newaxis, np.newaxis]
        return np.sum(flux * np.exp(-2 * np.pi * (l0 * u + m0 * v) * 1j), axis=0)

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

    def _scale_uv(self):
        scaled_uv = np.copy(self.uv)
        scaled_uv[:, 0] /= (2 * self.u_max / self.Nx)
        scaled_uv[:, 1] /= (2 * self.v_max / self.Ny)
        scaled_uv[:, 0] += self.Nx / 2
        scaled_uv[:, 1] += self.Ny / 2
        return np.round(scaled_uv).astype(int)

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

    def _grid(self):
        sources = self.skymodel_df[["flux", "l", "m"]].values
        for i, uv_point in enumerate(self.scaled_uv):
            u = uv_point[0]
            v = uv_point[1]
            if 0 <= u < self.Nx and 0 <= v < self.Ny:
                self.gridded_uv[v, -u] = 1
                flux, l, m = sources[:, 0], sources[:, 1], sources[:, 2]
                self.gridded_vis[v, -u] += np.sum(
                    flux * np.exp(-2 * np.pi * 1j * (l * self.uv[i][0] + m * self.uv[i][1]))
                )

    def _get_baseline_uv(self, name):
        baselines = np.concatenate([self.baselines.name.values, self.baselines.conj_name.values])
        if name not in baselines:
            return []
        return np.stack(
            self.baselines[(self.baselines.name == name) | (self.baselines.conj_name == name)]["UVW"].values
        )[0, :, :2]

    def plot_sky_model(self):
        size = self.img_conf["plane_size"] / 2
        plt.imshow(self.skymodel, extent=(-size, size, -size, size), cmap='jet')
        plt.colorbar(label='Brightness')
        plt.xlabel(r'l ($^{\circ}$)')
        plt.ylabel(r'm ($^{\circ}$)')
        plt.title('Skymodel (in brightness)')
        plt.savefig(self.results_dir / "skymodel.png")
        plt.close()

    def plot_antennas_2D(self):
        E = self.enu_coords[:, 0]
        N = self.enu_coords[:, 1]

        max_E = np.max(np.abs(E))
        max_N = np.max(np.abs(N))

        marker = plt.imread(input_dir / 'telescope.png')
        marker_size = 10

        fig = plt.figure()
        ax = fig.add_subplot(111)

        for i in range(len(E)):
            ax.imshow(marker, extent=[E[i] - marker_size / 2, E[i] + marker_size / 2, N[i] - marker_size / 2,
                                      N[i] + marker_size / 2])

        ax.set_xlabel("W-E (m)")
        ax.set_ylabel("S-N (m)")

        ax.set_xlim([-max_E - (1 / 3) * max_E, max_E + (1 / 3) * max_E])
        ax.set_ylim([-max_N - (1 / 3) * max_N, max_N + (1 / 3) * max_N])

        plt.savefig(self.results_dir / "antennae.png")
        plt.close()

    def plot_uv(self):
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

    def plot_psf(self):
        size = self.img_conf["plane_size"] / 2
        plt.imshow(log_scale(self.psf), extent=(-size, size, -size, size))
        plt.xlabel(r"l ($^{\circ})$")
        plt.ylabel(r"m ($^{\circ})$")
        plt.title('PSF')
        plt.colorbar()
        plt.savefig(self.results_dir / "psf.png")
        plt.close()

    def plot_gridded_visibilities(self):
        u_min = np.floor(self.u_min)
        v_min = np.floor(self.v_min)
        u_max = np.ceil(self.u_max)
        v_max = np.ceil(self.v_max)

        plt.imshow(log_scale(self.gridded_vis), extent=(u_min, u_max, v_min, v_max), cmap="jet")
        plt.xlabel(r"u (rad$^{-1})$")
        plt.ylabel(r"v (rad$^{-1})$")
        plt.title('Gridded Visibilities (Amplitude)')
        plt.colorbar(label="Magnitude", orientation='vertical')
        plt.savefig(self.results_dir / "gridded_amp.png")
        plt.close()

        plt.imshow(np.angle(self.gridded_vis), extent=(u_min, u_max, v_min, v_max), cmap="jet")
        plt.xlabel(r"u (rad$^{-1})$")
        plt.ylabel(r"v (rad$^{-1})$")
        plt.title('Gridded Visibilities (Phase)')
        plt.colorbar(label="Magnitude", orientation='vertical')
        plt.savefig(self.results_dir / "gridded_phase.png")
        plt.close()

    def plot_observed_image(self):
        size = self.img_conf["plane_size"] / 2
        plt.imshow(self.obs_img, cmap="jet", extent=(-size, size, -size, size))
        plt.xlabel(r"l ($^{\circ})$")
        plt.ylabel(r"m ($^{\circ})$")
        plt.title('Observed Image')
        plt.colorbar()
        plt.savefig(self.results_dir / "image.png")
        plt.close()

    def plot_visibilities(self):
        baselines = self.img_conf["baselines"]
        for baseline in baselines:
            uv = self._get_baseline_uv(baseline)
            if len(uv) == 0:
                print(f"Baseline {baseline} not found!")
                continue

            u_range = np.sort(uv[:, 0])
            v_range = np.sort(uv[:, 1])
            visibilities = self._get_visibilities(u_range, v_range)

            u_min = np.floor(u_range[0])
            v_min = np.floor(v_range[0])
            u_max = np.ceil(u_range[-1])
            v_max = np.ceil(v_range[-1])

            plt.figure(figsize=(5, 6))
            plt.imshow(log_scale(visibilities), extent=(u_min, u_max, v_min, v_max), cmap="jet")
            plt.xlabel(r"u (rad$^{-1})$")
            plt.ylabel(r"v (rad$^{-1})$")
            plt.title(f'Baseline {baseline} Visibilities (Amplitude)')
            plt.colorbar(label="Magnitude", orientation='vertical')
            plt.tight_layout()
            plt.savefig(self.results_dir / f"vis_b_{baseline}_amp.png")
            plt.close()

            plt.figure(figsize=(5, 6))
            plt.imshow(np.angle(visibilities), extent=(u_min, u_max, v_min, v_max), cmap='jet')
            plt.xlabel(r"u (rad$^{-1})$")
            plt.ylabel(r"v (rad$^{-1})$")
            plt.title(f'Baseline {baseline} Visibilities (Phase)')
            plt.colorbar(label="Magnitude", orientation='vertical')
            plt.tight_layout()
            plt.savefig(self.results_dir / f"vis_b_{baseline}_phase.png")
            plt.close()


if __name__ == "__main__":
    simvis = SimVis()

    print("Baselines:")
    print(simvis.baselines[["name", "b"]])

    print()
    print("Distance:")
    print(simvis.baselines[["name", "D"]])

    print()
    print("Azimuth:")
    print(simvis.baselines[["name", "A"]])

    print()
    print("Elevation:")
    print(simvis.baselines[["name", "E"]])

    print()
    print("XYZ:")
    print(simvis.baselines[["name", "XYZ"]])

    print()
    print("UV:")
    print(simvis.uv)

    simvis.plot_antennas_2D()
    simvis.plot_sky_model()
    simvis.plot_uv()
    simvis.plot_visibilities()
    simvis.plot_psf()
    simvis.plot_gridded_visibilities()
    simvis.plot_observed_image()
