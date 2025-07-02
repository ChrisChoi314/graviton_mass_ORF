import numpy as np
from scipy.signal import argrelmin, argrelmax

theta = np.array([10, 23.1, 36.7, 48.7,63, 76, 89.3, 102.8, 116.3, 130.2, 142, 156.2, 168.2])
correlation_coefficient = np.array([.14, .32 ,0.206, 0.059, -0.047, 0.004, -.013, 0.08, 0.004, 0.054, -0.009, -0.427, -0.195])

### performing parabolic interpolation to find extrema
omega_1 = theta[6]
omega_2 = theta[7]
omega_3 = theta[8]
I_1 = correlation_coefficient[6]
I_2 = correlation_coefficient[7]
I_3 = correlation_coefficient[8]
delta_omega = omega_3- omega_2
omega_par = omega_3 - delta_omega*((I_3**2 - I_2**2)/(I_1**2 - 2*I_2**2+I_3**2) + 1/2)
print(f'maximum for CPTA: omega_par = {omega_par}')

omega_1 = theta[3]
omega_2 = theta[4]
omega_3 = theta[5]
I_1 = correlation_coefficient[3]
I_2 = correlation_coefficient[4]
I_3 = correlation_coefficient[5]
delta_omega = omega_3- omega_2
omega_par = omega_3 - delta_omega*((I_3**2 - I_2**2)/(I_1**2 - 2*I_2**2+I_3**2) + 1/2)

omega1, omega2, omega3 = theta[3], theta[4], theta[5] # equally‑spaced points
f1, f2, f3 = correlation_coefficient[3], correlation_coefficient[4], correlation_coefficient[5]

h = omega3 - omega2 # grid step (omega2 − omega1 = h)
omega_min   = omega2 + h * (f1 - f3) / (2 * (f1 - 2*f2 + f3))

print(f'minimum for CPTA: omega_min = {omega_min}')

normalizing_amp = 4.501905414729434e-30

xii_mean = np.load('xii_mean.npy')
xii_mean = xii_mean*180/np.pi

rho_avg = np.load('rho_avg.npy')
rho_avg = np.array(rho_avg)/normalizing_amp

sig_avg = np.load('sig_avg.npy')
sig_avg = np.array(sig_avg)/normalizing_amp

omega1, omega2, omega3 = xii_mean[6], xii_mean[7], xii_mean[8]  # equally‑spaced points
f1, f2, f3 = rho_avg[6], rho_avg[7], rho_avg[8]

h = omega3 - omega2 # grid step (omega2 − omega1 = h)
omega_min   = omega2 + h * (f1 - f3) / (2 * (f1 - 2*f2 + f3))

print(f'minimum for NANOGrav15: omega_min = {omega_min}')

### calculating extrema from CPTA: 

gamma_min = np.load('monte_carlo_arrays/gamma_01.npy')
gamma_min = np.real(gamma_min)
gamma_max = np.load('monte_carlo_arrays/gamma_61.npy')
gamma_max = np.real(gamma_max)
xi_values = np.load('xi_values.npy')
xi_values = xi_values*180/np.pi

minima_indices = argrelmin(gamma_min)
print(f"Indices of local minima for CPTA: {minima_indices}")
print(f"Local minima values CPTA: {gamma_min[minima_indices]}") 
print(f"angle = {xi_values[minima_indices]}")

maxima_indices = argrelmax(gamma_min)
print(f"Indices of local maxima for CPTA: {maxima_indices}")
print(f"Local maxima values CPTA: {gamma_min[maxima_indices]}") 
print(f"angle = {xi_values[maxima_indices]}")

### calculating extrema from NANOGrav15:
minima_indices = argrelmin(gamma_max)
print(f"Indices of local minima for NANOGrav15: {minima_indices}")
print(f"Local minima values NANOGrav15: {gamma_max[minima_indices]}")
print(f"angle = {xi_values[minima_indices]}")