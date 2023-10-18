import numpy as np
import matplotlib.pyplot as plt
from read import enu_coords, config, skymodel_df


class simvis():
    def __init__(self, h_min, h_max, dec, lat, freq, antenna_ENU, point_sources, nsteps) -> None:
        # self.antenna_ENU = np.array([[0,0,0],[2,3,0.5],[5,6,1]])
        self.h_min = h_min
        self.h_max = h_max
        self.dec = dec
        self.lat = lat
        self.freq = freq
        self.antenna_ENU = antenna_ENU
        self.point_sources = point_sources
        self.nsteps = nsteps
        self.N = len(enu_coords)
        self.B = (int)((self.N**2 - self.N)/2)
        self.speed_of_light = 299792458 
        self.wavelength = self.speed_of_light/ (self.freq * 1e9)
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
        plt.cla()
        plt.clf()

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
        # plt.cla()
        # plt.clf()

    def ra_to_rad(self, h, m, s):
        ra_h = h + (m / 60) + (s/3600)
        ra_rad = (ra_h / 24) * 2 * np.pi

        return ra_rad
    
    def dec_to_rad(self, d, arc_m, arc_s):
        total_degrees = d + arc_m/60 + arc_s/3600
        dec_rad = total_degrees * (np.pi/180)

        return dec_rad
    
    def deg_to_rad(self, degree):
        return degree*np.pi/180
    
    def rad_to_deg(self, radiant):
        return radiant*180/np.pi
    
    def lat_to_rad(self, d, arc_m, arc_s):
        total_degrees = d + arc_m/60 + arc_s/3600
        lat_rad = total_degrees * (np.pi/180)

        return lat_rad
    
    def lm_coordinates(self, ra_0, dec_0, ra, dec):

        delta_ra = ra - ra_0
        l = np.cos(dec)*np.sin(delta_ra)
        m = np.sin(dec)*np.cos(dec_0) - np.cos(dec)*np.sin(dec_0)*np.cos(delta_ra)

        return [l,m]
    
    def sky_model(self):
        sigma = 0.1
        lm_plane_size = 10
        pixel_count = 500

        l_range = np.linspace(-lm_plane_size/2, lm_plane_size/2, pixel_count)
        m_range = np.linspace(-lm_plane_size/2, lm_plane_size/2, pixel_count)

        sky_model = np.zeros((pixel_count, pixel_count))

        for m in range(len(m_range)):
            for l in range(len(l_range)):
                for point in self.point_sources:
                    sky_model[len(m_range)-m-1,l] += point[0] * np.exp(-((l_range[l] - self.rad_to_deg(point[1]))**2 + (m_range[m] - self.rad_to_deg(point[2]))**2) / (2 * sigma**2))

        plt.imshow(sky_model, extent=(-lm_plane_size/2, lm_plane_size/2, -lm_plane_size/2, lm_plane_size/2), cmap = "jet")
        plt.colorbar(label='Brightness')
        plt.xlabel('l (degrees)')
        plt.ylabel('m (degrees)')
        plt.title('Skymodel (in brightness)')
        plt.show()
        plt.cla()
        plt.clf()

    def plot_real_visibility(self):
        u_range = np.linspace(-4000, 4000, 500)
        v_range = np.linspace(-3000, 3000, 500)

        real_visibilities = np.zeros((len(u_range), len(v_range)))

        for v in range(len(v_range)):
            for u in range(len(u_range)):
                for point in self.point_sources:
                    real_visibilities[len(v_range)-v-1,u] += point[0] * np.cos(2 * np.pi * (point[1] * u_range[u] + point[2] * v_range[v]))

        plt.figure(figsize=(10, 6))
        plt.imshow(real_visibilities, extent=(u_range.min(), u_range.max(), v_range.min(), v_range.max()), cmap = "jet")
        plt.colorbar(label='Magnitude of Visibilities')
        plt.xlabel('u (rad^-1)')
        plt.ylabel('v (rad^-1)')
        plt.title('Real part of Visibility')
        plt.show()
        plt.cla()
        plt.clf()

    def plot_imaginary_visibility(self):
        u_range = np.linspace(-4000, 4000, 500)
        v_range = np.linspace(-3000, 3000, 500)

        imaginary_visibilities = np.zeros((len(u_range), len(v_range)))

        for v in range(len(v_range)):
            for u in range(len(u_range)):
                for point in self.point_sources:
                    imaginary_visibilities[len(v_range)-v-1,u] += -point[0] * np.sin(2 * np.pi * (point[1] * u_range[u] + point[2] * v_range[v]))

        plt.figure(figsize=(10, 6))
        plt.imshow(imaginary_visibilities, extent=(u_range.min(), u_range.max(), v_range.min(), v_range.max()), cmap='jet')
        plt.colorbar(label='Magnitude of Visibilities')
        plt.xlabel('u (rad^-1)')
        plt.ylabel('v (rad^-1)')
        plt.title('Imaginary part of Visibility')
        plt.show()
        plt.cla()
        plt.clf()
        pass

    def plot_visibilities(self):
        u_range = np.linspace(-4000, 4000, 500)
        v_range = np.linspace(-3000, 3000, 500)

        real_visibilities = np.zeros((len(u_range), len(v_range)))
        imaginary_visibilities = np.zeros((len(u_range), len(v_range)))

        for v in range(len(v_range)):
            for u in range(len(u_range)):
                for point in self.point_sources:
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
        plt.cla()
        plt.clf()

    def ENU_to_XYZ(self, azimith, elevation, distance, latitude):
        XYZ_coordinate = distance*np.array([np.cos(latitude)*np.sin(elevation) - np.sin(latitude)*np.cos(elevation)*np.cos(azimith),
                                np.cos(elevation)*np.sin(azimith),
                                np.sin(latitude)*np.sin(elevation) + np.cos(latitude)*np.cos(elevation)*np.cos(azimith)])
        
        return XYZ_coordinate

    def XYZ_to_UVW(self, hour_angle, declination, wavelength, xyz_coordinate):
        matrix = np.array([[np.sin(hour_angle), np.cos(hour_angle), 0],
                                [-np.sin(declination)*np.cos(hour_angle), np.sin(declination)*np.sin(hour_angle), np.cos(declination)],
                                [np.cos(declination), -np.cos(declination)*np.sin(hour_angle), np.sin(declination)]])
        
        uvw_coordinate = matrix.dot(xyz_coordinate/ wavelength)

        return uvw_coordinate

    def uv_tracks(self):
        baseline_lengths = [0.0]*self.B
        azimiths = [0.0]*self.B
        elevations = [0.0]*self.B

        hour_angle_range = np.linspace(self.h_min, self.h_max, self.nsteps) * np.pi/12

        self.u_m = np.zeros((self.N,self.N,self.nsteps),dtype=float) 
        self.v_m = np.zeros((self.N,self.N,self.nsteps),dtype=float)
        self.w_m = np.zeros((self.N,self.N,self.nsteps),dtype=float)  

        count = 0
        for i in range(self.N):
            for j in range(i+1, self.N):
                baseline_lengths[count] = np.sqrt(np.sum((self.antenna_ENU[i] - self.antenna_ENU[j])**2))
                azimiths[count] = np.arctan2(self.antenna_ENU[j,0] - self.antenna_ENU[i,0], self.antenna_ENU[j,1] - self.antenna_ENU[i,1])
                elevations[count] = np.arcsin((self.antenna_ENU[j,2] - self.antenna_ENU[i,2])/baseline_lengths[count]) 

                #Calculating the XYZ coordinates
                xyz_coordinate = self.ENU_to_XYZ(azimiths[count], elevations[count], baseline_lengths[count], lat_to_rad(*self.lat))
                #Calculating UVW coordinates for all baselines for all hourangles
                for n in range(self.nsteps):
                    uvw_coordinate = self.XYZ_to_UVW(hour_angle_range[n],dec_to_rad(*self.dec),self.wavelength,xyz_coordinate)
                    self.u_m[i,j,n] = uvw_coordinate[0]
                    self.u_m[j,i,n] = -uvw_coordinate[0]
                    self.v_m[i,j,n] = uvw_coordinate[1]
                    self.v_m[j,i,n] = -uvw_coordinate[1]
                    self.w_m[i,j,n] = uvw_coordinate[2]
                    self.w_m[j,i,n] = -uvw_coordinate[2]

                count+=1
        pass

    def plot_uv_track(self):
        for i in range(self.N):
            for j in range(i+1,self.N):
                u = self.u_m[i,j,:]
                v = self.v_m[i,j,:]

                plt.plot(u,v,"b")
                plt.plot(-u,-v,"r")

        plt.xlabel('u (rad^-1)')
        plt.ylabel('v (rad^-1)')
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

def main():
    center_ra = config['right_asc']
    center_dec = config['declination']

    papino_df = skymodel_df[skymodel_df["name"] == "Papino"]
    paperino_df = skymodel_df[skymodel_df["name"] == "Paperino"]

    papino_ra = list(map(float,papino_df["right_asc"][0]))
    papino_dec = list(map(float,papino_df["declination"][0]))
    paperino_ra = list(map(float,paperino_df["right_asc"][1]))
    paperino_dec = list(map(float,paperino_df["declination"][1]))

    papino_lm = lm_coordinates(ra_to_rad(*center_ra), dec_to_rad(*center_dec), ra_to_rad(*papino_ra), dec_to_rad(*papino_dec))
    paperino_lm = lm_coordinates(ra_to_rad(*center_ra), dec_to_rad(*center_dec), ra_to_rad(*paperino_ra), dec_to_rad(*paperino_dec))

    papino_flux = papino_df['flux'][0]
    paperino_flux = paperino_df['flux'][1]


    point_sources = [[papino_flux, papino_lm[0], papino_lm[1]],[paperino_flux, paperino_lm[0], paperino_lm[1]]]
    
    num_steps = config["num_steps"]
    h_range = config['hour_angle_range']
    h_min = h_range[0]
    h_max = h_range[1]
    declination = config['declination']
    latitude = config["lat"]
    freq = config["obs_freq"]

    s = simvis(h_min=h_min, h_max=h_max, dec=declination, lat=latitude, freq=freq, antenna_ENU=enu_coords, point_sources=point_sources, nsteps=num_steps)
    s.plot_antennas_2D()
    s.plot_visibilities()
    s.uv_tracks()
    s.plot_uv_track()
    pass

if __name__ == "__main__":
    main()
    pass