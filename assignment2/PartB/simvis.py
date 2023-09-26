import numpy as np
import matplotlib.pyplot as plt


class simvis():
    def __init__(self) -> None:
        self.antenna_ENU = np.array([[0,0,0],[2,3,0.5],[5,6,1]])
        pass

    def read_antennas(self):
        pass

    def plot_antennas_3D(self):
        E = []
        N = []
        U = []

        for antenna in self.antenna_ENU:
            E.append(antenna[0])
            N.append(antenna[1])
            U.append(antenna[2])

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(E,N,U)

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

        fig = plt.figure()
        ax = fig.add_subplot(111)

        ax.scatter(E,N)

        ax.set_xlabel("W-E (m)")
        ax.set_ylabel("S-N (m)")

        plt.show()

s = simvis()
s.plot_antennas_2D()