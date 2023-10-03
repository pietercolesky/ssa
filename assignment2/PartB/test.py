import numpy as np
import matplotlib.pyplot as plt

center = [[5,30,0],[0,0,0]]
betelgeuse = [[5 , 55, 10.3053],[7,24,25.426]]
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

def sky_model(point_sources):
    sigma = 0.1
    lm_plane_size = 10
    pixel_count = 500

    l_range = np.linspace(-lm_plane_size/2, lm_plane_size/2, pixel_count)
    m_range = np.linspace(-lm_plane_size/2, lm_plane_size/2, pixel_count)

    sky_model = np.zeros((pixel_count, pixel_count))

    for m in range(len(m_range)):
        for l in range(len(l_range)):
            for point in point_sources:
                sky_model[len(m_range)-m-1,l] += point[0] * np.exp(-((l_range[l] - point[1])**2 + (m_range[m] - point[2])**2) / (2 * sigma**2))

    plt.imshow(sky_model, extent=(-lm_plane_size/2, lm_plane_size/2, -lm_plane_size/2, lm_plane_size/2), cmap = "jet")
    plt.colorbar(label='Brightness')
    plt.xlabel('l (degrees)')
    plt.ylabel('m (degrees)')
    plt.title('Skymodel (in brightness)')
    plt.show()

def real_visibility(point_sources):
    u_range = np.linspace(-4000, 4000, 500)
    v_range = np.linspace(-3000, 3000, 500)

    real_visibilities = np.zeros((len(u_range), len(v_range)))

    for v in range(len(v_range)):
        for u in range(len(u_range)):
            for point in point_sources:
                real_visibilities[len(v_range)-v-1,u] += point[0] * np.cos(2 * np.pi * (point[1] * u_range[u] + point[2] * v_range[v]))

    plt.figure(figsize=(10, 6))
    plt.imshow(real_visibilities, extent=(u_range.min(), u_range.max(), v_range.min(), v_range.max()), cmap = "jet")
    plt.colorbar(label='Magnitude of Visibilities')
    plt.xlabel('u (rad^-1)')
    plt.ylabel('v (rad^-1)')
    plt.title('Real part of Visibility')
    plt.show()

def imaginary_visibility(point_sources):
    u_range = np.linspace(-4000, 4000, 500)
    v_range = np.linspace(-3000, 3000, 500)

    imaginary_visibilities = np.zeros((len(u_range), len(v_range)))

    for v in range(len(v_range)):
        for u in range(len(u_range)):
            for point in point_sources:
                imaginary_visibilities[len(v_range)-v-1,u] += -point[0] * np.sin(2 * np.pi * (point[1] * u_range[u] + point[2] * v_range[v]))

    plt.figure(figsize=(10, 6))
    plt.imshow(imaginary_visibilities, extent=(u_range.min(), u_range.max(), v_range.min(), v_range.max()), cmap='jet')
    plt.colorbar(label='Magnitude of Visibilities')
    plt.xlabel('u (rad^-1)')
    plt.ylabel('v (rad^-1)')
    plt.title('Imaginary part of Visibility')
    plt.show()
    pass

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

print()
print("Question 1.2")

from read import enu_coords, config, skymodel_df


center_ra = config['right_asc']
center_dec = config['declination']

papino_df = skymodel_df[skymodel_df["name"] == "Papino"]
print(papino_df)
paperino_df = skymodel_df[skymodel_df["name"] == "Paperino"]

papino_ra = list(map(float,papino_df["right_asc"][0]))
papino_dec = list(map(float,papino_df["declination"][0]))
paperino_ra = list(map(float,paperino_df["right_asc"][1]))
paperino_dec = list(map(float,paperino_df["declination"][1]))

print("Papino LM")
papino_lm = lm_coordinates(ra_to_rad(*center_ra), dec_to_rad(*center_dec), ra_to_rad(*papino_ra), dec_to_rad(*papino_dec))
print(papino_lm)
print("Paperino LM")
paperino_lm = lm_coordinates(ra_to_rad(*center_ra), dec_to_rad(*center_dec), ra_to_rad(*paperino_ra), dec_to_rad(*paperino_dec))
print(paperino_lm)

papino_flux = papino_df['flux'][0]
paperino_flux = paperino_df['flux'][1]


point_sources = [[papino_flux, papino_lm[0], papino_lm[1]],[paperino_flux, paperino_lm[0], paperino_lm[1]]]
print(point_sources)
sky_model(point_sources)

real_visibility(point_sources)
imaginary_visibility(point_sources)





