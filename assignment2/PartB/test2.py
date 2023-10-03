import numpy as np
import matplotlib.pyplot as plt

# Define parameters for multiple sources
amplitudes = [1.0, 0.7, 0.5]  # Amplitudes of sources
l_values = [0.1, -0.2, 0.3]  # Direction cosines l for sources
m_values = [0.2, 0.1, -0.1]  # Direction cosines m for sources

# Define the range of u and v values
u_range = np.linspace(-100, 100, 200)
v_range = np.linspace(-100, 100, 200)

# Define the frequency (in Hertz)
frequency = 1.0e9  # For example, 1 GHz

# Speed of light
c = 299792458.0  # Speed of light in meters per second

# Calculate u and v values based on frequency and hour angle
hour_angle = 0.0  # Replace with the desired hour angle
u = (frequency / c) * np.cos(hour_angle)
v = (frequency / c) * np.sin(hour_angle)

# Calculate the complex visibilities V(u, v) for multiple sources
complex_visibilities = np.zeros((len(u_range), len(v_range)), dtype=complex)

for i, u in enumerate(u_range):
    for j, v in enumerate(v_range):
        for source_index in range(len(amplitudes)):
            complex_visibilities[i, j] += amplitudes[source_index] * np.exp(-2 * np.pi * 1j * (l_values[source_index] * u + m_values[source_index] * v))

# Plot the magnitude of the complex visibilities
plt.figure(figsize=(10, 6))
plt.imshow(np.abs(complex_visibilities), extent=(u_range.min(), u_range.max(), v_range.min(), v_range.max()), aspect='auto', origin='lower', cmap='jet')
plt.colorbar(label='Magnitude of Visibilities')
plt.xlabel('u (rad^-1)')
plt.ylabel('v (rad^-1)')
plt.title('2D Plot of Complex Visibilities (Magnitude)')
plt.show()