import numpy as np

center = [[5,30,0],[0,0,0]]
betelgeuse = [[5 , 5, 10.3053],[7,24,25.426]]
rigel = [[5,14,32.272],[-8,12,5.898]]

def ra_to_rad(h, m, s):
    ra_h = h + (m / 60) + (s/3600)
    ra_rad = (ra_h / 24) * 2 * np.pi

    return ra_rad

def dec_to_rad(d, arc_m, arc_s):
    total_degrees = d + arc_m/60 + arc_s/3600
    dec_rad = total_degrees * (np.pi/180)

    return dec_rad

def lm_coordinates(ra_0, dec_0, ra, dec):
    delta_ra = ra - ra_0
    l = np.cos(dec)*np.sin(delta_ra)
    m = np.sin(dec)*np.cos(dec_0) - np.cos(dec)*np.sin(dec_0)*np.cos(delta_ra)

    return [l,m]

center_ra = center[0]
center_dec = center[1]

betelgeuse_ra = betelgeuse[0]
betelgeuse_dec = betelgeuse[1]

rigel_ra =rigel[0]
rigel_dec = rigel[1]

#1
print("Question 1.1")
print("1)")

print()
print("Center LM")
center_lm = lm_coordinates(ra_to_rad(*center_ra), dec_to_rad(*center_dec), ra_to_rad(*center_ra), dec_to_rad(*center_dec))
print(center_lm)

print()
print("Betelgeuse LM")
betelgeuse_lm = lm_coordinates(ra_to_rad(*center_ra), dec_to_rad(*center_dec), ra_to_rad(*betelgeuse_ra), dec_to_rad(*betelgeuse_dec))
print(betelgeuse_lm)

print()
print("Rigel LM")
rigel_lm = lm_coordinates(ra_to_rad(*center_ra), dec_to_rad(*center_dec), ra_to_rad(*rigel_ra), dec_to_rad(*rigel_dec))
print(rigel_lm)

#2
print()
print("2)")
print("Distance for Betelgeuse")
betelgeuse_distance = np.sqrt(betelgeuse_lm[0]**2 + betelgeuse_lm[1]**2)
print(betelgeuse_distance)

#3
print()
print("3)")
print("Angular distance Betelgeuse")

betelgeuse_angular_distance = np.sinh(betelgeuse_distance)
print(betelgeuse_angular_distance)

#4
print()
print("4)")
print(betelgeuse_distance**2)
print(np.sin(betelgeuse_angular_distance)**2)

print("Question 1.2")

from read import enu_coords
print(enu_coords)



