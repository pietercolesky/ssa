import numpy as np
import matplotlib.pyplot as plt

N_a = 7
D = 12
b_max = 185
delta_f = 92.9121 * 1e6
freq_min = 1.4 * 1e9
freq_max = 1.95 * 1e9
c = 3 * 1e8

delta_l_c = 5/3600 * np.pi/180
delta_m_c = 5/3600 * np.pi/180

delta_cell = delta_l_c * delta_m_c

print(delta_l_c)

N_x = 10

l_max = 0.5 * N_x * delta_l_c
l_min = -0.5 * N_x * delta_l_c
m_max = 0.5 * N_x * delta_m_c
m_min = -0.5 * N_x * delta_m_c

def deg_to_rad(degree):
        return degree*np.pi/180

def rad_to_deg(radiant):
        return radiant*180/np.pi

def rad_to_arc_sec(radiant):
    return radiant*180/np.pi * 3600

def rad_to_arc_min(radiant):
    return radiant*180/np.pi * 60


delta_sphere = 4 * np.pi #sterradians
delta_image = N_x**2 * delta_cell

B = (N_a**2 - N_a)/2
print(B)

lambda_0 = c/freq_min

theta_p = lambda_0 / D # approximate size of primary beam
theta_s = lambda_0 / b_max # angular resolution of the interferometer

delta_theta = (theta_s * freq_min)/ delta_f # maximum angular radius
P = 86164.0905

delta_t = (theta_s * P)/(2*np.pi*delta_theta)

print(delta_t)

