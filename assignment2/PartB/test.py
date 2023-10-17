import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

center = [[5,30,0],[0,0,0]]
betelgeuse = [[5 , 55, 10.3053],[7,24,25.426]]
rigel = [[5,14,32.272],[-8,12,5.898]]

def plot_antennas_2D(enu_coordinates):
    E = []
    N = []
    
    for antenna in enu_coordinates:
        E.append(antenna[0])
        N.append(antenna[1])

    max_E = max(np.abs(E))
    max_N = max(np.abs(N))

    marker = plt.imread('telescope.png')
    marker_size = 10
    

    fig = plt.figure()
    ax = fig.add_subplot(111)

    # ax.scatter(E,N, marker="^")

    for i in range(len(E)):
        ax.imshow(marker, extent=[E[i] - marker_size / 2, E[i] + marker_size / 2, N[i] - marker_size / 2, N[i] + marker_size / 2])

    ax.set_xlabel("W-E (m)")
    ax.set_ylabel("S-N (m)")

    ax.set_xlim([-max_E - (1/3)*max_E, max_E + (1/3)*max_E])
    ax.set_ylim([-max_N - (1/3)*max_N, max_N + (1/3)*max_N])

    plt.show()
    plt.cla()
    plt.clf()

def ra_to_rad(h, m, s):
    ra_h = h + (m / 60) + (s/3600)
    ra_rad = (ra_h / 24) * 2 * np.pi

    return ra_rad

def dec_to_rad(d, arc_m, arc_s):
    total_degrees = d + arc_m/60 + arc_s/3600
    dec_rad = total_degrees * (np.pi/180)

    return dec_rad

def deg_to_rad(degree):
    return degree*np.pi/180

def rad_to_deg(radiant):
    return radiant*180/np.pi

def lat_to_rad(d, arc_m, arc_s):
    total_degrees = d + arc_m/60 + arc_s/3600
    lat_rad = total_degrees * (np.pi/180)

    return lat_rad

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
                sky_model[len(m_range)-m-1,l] += point[0] * np.exp(-((l_range[l] - rad_to_deg(point[1]))**2 + (m_range[m] - rad_to_deg(point[2]))**2) / (2 * sigma**2))

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

def visibilities(point_sources):
    u_range = np.linspace(-4000, 4000, 500)
    v_range = np.linspace(-3000, 3000, 500)

    real_visibilities = np.zeros((len(u_range), len(v_range)))
    imaginary_visibilities = np.zeros((len(u_range), len(v_range)))

    for v in range(len(v_range)):
        for u in range(len(u_range)):
            for point in point_sources:
                imaginary_visibilities[len(v_range)-v-1,u] += -point[0] * np.sin(2 * np.pi * (point[1] * u_range[u] + point[2] * v_range[v]))
                real_visibilities[len(v_range)-v-1,u] += point[0] * np.cos(2 * np.pi * (point[1] * u_range[u] + point[2] * v_range[v]))


    fig, (ax1, ax2) = plt.subplots(1,2,figsize=(10,8))


    im1 = ax1.imshow(real_visibilities, extent=(u_range.min(), u_range.max(), v_range.min(), v_range.max()), cmap = "jet", aspect='auto')
    # ax1.colorbar(label='Magnitude of Visibilities')
    ax1.set_title('Real part of Visibility')
    ax1.set_xlabel('u (rad^-1)')
    ax1.set_ylabel('v (rad^-1)')
    cbar = fig.colorbar(im1, ax=ax1, label='Magnitude of Visibilities')

    im2 = ax2.imshow(imaginary_visibilities, extent=(u_range.min(), u_range.max(), v_range.min(), v_range.max()), cmap = "jet", aspect='auto')
    # ax1.colorbar(label='Magnitude of Visibilities')
    ax2.set_title('Imaginary part of Visibility')
    ax2.set_xlabel('u (rad^-1)')
    ax2.set_ylabel('v (rad^-1)')
    cbar = fig.colorbar(im2, ax=ax2, label='Magnitude of Visibilities')

    plt.tight_layout()
    plt.show()

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


point_sources = [[1, 0, 0],[0.2, 0, 0.01832]]
# point_sources = [[1, 0, 0],[0.2, deg_to_rad(0.3), deg_to_rad(0.3)]]

sky_model(point_sources)

visibilities(point_sources)


#Calculating Baseline length, 

def ENU_to_XYZ(azimith, elevation, distance, latitude):
    XYZ_coordinate = distance*np.array([np.cos(latitude)*np.sin(elevation) - np.sin(latitude)*np.cos(elevation)*np.cos(azimith),
                               np.cos(elevation)*np.sin(azimith),
                               np.sin(latitude)*np.sin(elevation) + np.cos(latitude)*np.cos(elevation)*np.cos(azimith)])
    
    return XYZ_coordinate

def XYZ_to_UVW(hour_angle, declination, wavelength, xyz_coordinate):
    matrix = np.array([[np.sin(hour_angle), np.cos(hour_angle), 0],
                             [-np.sin(declination)*np.cos(hour_angle), np.sin(declination)*np.sin(hour_angle), np.cos(declination)],
                             [np.cos(declination), -np.cos(declination)*np.sin(hour_angle), np.sin(declination)]])
    
    uvw_coordinate = matrix.dot(xyz_coordinate/ wavelength)

    return uvw_coordinate



num_steps = config["num_steps"]
hour_angle_range = np.linspace(config['hour_angle_range'][0], config['hour_angle_range'][1], num_steps) * np.pi/12

speed_of_light = 299792458 
wavelength = speed_of_light/ (150*10**6)

plot_antennas_2D(enu_coords)

N = len(enu_coords)
B = (int)((N**2 - N)/2)

baseline_lengths = [0.0]*B
azimiths = [0.0]*B
elevations = [0.0]*B

xyz_coordinates = []

u_m = np.zeros((N,N,num_steps),dtype=float) 
v_m = np.zeros((N,N,num_steps),dtype=float)
w_m = np.zeros((N,N,num_steps),dtype=float)  

count = 0
for i in range(N):
    for j in range(i+1, N):
        baseline_lengths[count] = np.sqrt(np.sum((enu_coords[i] - enu_coords[j])**2))
        azimiths[count] = np.arctan2(enu_coords[j,0] - enu_coords[i,0], enu_coords[j,1] - enu_coords[i,1])
        elevations[count] = np.arcsin((enu_coords[j,2] - enu_coords[i,2])/baseline_lengths[count]) 

        #Calculating the XYZ coordinates
        xyz_coordinate = ENU_to_XYZ(azimiths[count], elevations[count], baseline_lengths[count], lat_to_rad(*config["lat"]))
        #Calculating UVW coordinates for all baselines for all hourangles
        for n in range(num_steps):
            uvw_coordinate = XYZ_to_UVW(hour_angle_range[n],dec_to_rad(*config['declination']),wavelength,xyz_coordinate)
            u_m[i,j,n] = uvw_coordinate[0]
            u_m[j,i,n] = -uvw_coordinate[0]
            v_m[i,j,n] = uvw_coordinate[1]
            v_m[j,i,n] = -uvw_coordinate[1]
            w_m[i,j,n] = uvw_coordinate[2]
            w_m[j,i,n] = -uvw_coordinate[2]
            
            pass
        count+=1


plt.plot(u_m[1,2,:], v_m[1,2,:], "b")
plt.plot(u_m[2,1,:], v_m[2,1,:], "r")
plt.show()
plt.cla()
plt.clf()

for i in range(N):
    for j in range(i+1,N):
        u = u_m[i,j,:]
        v = v_m[i,j,:]

def plot_uv_track(u_m, v_m):

    u_max = np.max(np.abs(u_m))
    v_max = np.max(np.abs(v_m))
    N = len(u_m)

    for i in range(N):
        for j in range(i+1,N):
            u = u_m[i,j,:]
            v = v_m[i,j,:]

            plt.plot(u,v,"b")
            plt.plot(-u,-v,"r")

    plt.xlabel('u (rad^-1)')
    plt.ylabel('v (rad^-1)')
    plt.show() 

plot_uv_track(u_m, v_m)






