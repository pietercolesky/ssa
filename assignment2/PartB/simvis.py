import numpy as np
import matplotlib.pyplot as plt
from read import enu_coords


class simvis():
    def __init__(self) -> None:
        # self.antenna_ENU = np.array([[0,0,0],[2,3,0.5],[5,6,1]])
        self.antenna_ENU = enu_coords
        pass

    def plot_antennas_3D(self):
        E = []
        N = []
        U = []

        for antenna in self.antenna_ENU:
            E.append(antenna[0])
            N.append(antenna[1])
            U.append(antenna[2])

        max_E = max(np.abs(E))
        max_N = max(np.abs(N))
        max_U = max(np.abs(U))
        min_U = min(U)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(E,N,U)

        ax.set_xlim([-max_E - (1/3)*max_E, max_E + (1/3)*max_E])
        ax.set_ylim([-max_N - (1/3)*max_N, max_N + (1/3)*max_N])
        ax.set_zlim([min_U, max_U + (1/3)*max_U])

        ax.set_xlabel("W-E (m)")
        ax.set_ylabel("S-N (m)")
        ax.set_zlabel("Z")

        plt.show()

    def plot_antennas_2D(self):
        E = []
        N = []
        
        for antenna in self.antenna_ENU:
            E.append(antenna[0])
            N.append(antenna[1])

        max_E = max(np.abs(E))
        max_N = max(np.abs(N))

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(E,N)

        ax.set_xlabel("W-E (m)")
        ax.set_ylabel("S-N (m)")

        ax.set_xlim([-max_E - (1/3)*max_E, max_E + (1/3)*max_E])
        ax.set_ylim([-max_N - (1/3)*max_N, max_N + (1/3)*max_N])

        plt.show()

s = simvis()
s.plot_antennas_3D()