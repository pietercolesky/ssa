import matplotlib.pyplot as plt
import numpy as np


def plot_sky_model(model, plane_size):
    plt.imshow(model, extent=(-plane_size / 2, plane_size / 2, -plane_size / 2, plane_size / 2), cmap='jet')
    plt.colorbar(label='Brightness')
    plt.xlabel(r'l ($^{\circ}$)')
    plt.ylabel(r'm ($^{\circ}$)')
    plt.title('Skymodel (in brightness)')
    plt.show()


def plot_visibilities(visibilities, u_min, u_max, v_min, v_max):
    fig, axes = plt.subplots(1, 2, figsize=(20, 6))

    im1 = axes[0].imshow(visibilities.real, extent=(u_min, u_max, v_min, v_max), cmap="jet")
    axes[0].set_xlabel(r"u (rad$^{-1})$")
    axes[0].set_ylabel(r"v (rad$^{-1})$")
    axes[0].set_title('Real part of Visibilities')
    cbar = fig.colorbar(im1, ax=axes[0], orientation='vertical')
    cbar.set_label('Magnitude')

    im2 = axes[1].imshow(visibilities.imag, extent=(u_min, u_max, v_min, v_max), cmap='jet')
    axes[1].set_xlabel(r"u (rad$^{-1})$")
    axes[1].set_ylabel(r"v (rad$^{-1})$")
    axes[1].set_title('Imaginary part of Visibilities')
    cbar = fig.colorbar(im2, ax=axes[1], orientation='vertical')
    cbar.set_label('Magnitude')

    plt.tight_layout()
    plt.show()


def plot_antennas_2D(enu_coordinates):
    E = enu_coordinates[:, 0]
    N = enu_coordinates[:, 1]

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
    plt.cla()
    plt.clf()


def plot_uv(values):
    for uvw in values:
        plt.scatter(uvw[:, 0], uvw[:, 1], s=2, c="b", label="Baselines")
        plt.scatter(-uvw[:, 0], -uvw[:, 1], s=2, c="r", label="Conjugate Baselines")

    plt.xlabel(r"u (rad$^{-1})$")
    plt.ylabel(r"v (rad$^{-1})$")
    plt.title('uv-tracks')
    plt.legend(["Baselines", "Conjugate Baselines"])
    plt.show()
