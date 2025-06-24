import matplotlib.pyplot as plt
import numpy as np
from polarization_func import *

# Estimated data points and error bar values from the image
theta = np.array([10, 23.1, 36.7, 48.7,63, 76, 89.3, 102.8, 116.3, 130.2, 142, 156.2, 168.2])
correlation_coefficient = np.array([.14, .32 ,0.206, 0.059, -0.047, 0.004, -.013, 0.08, 0.004, 0.054, -0.009, -0.427, -0.195])
y_error = np.array([0.114, 0.068,  0.068, 0.08, 0.101, 0.093, 0.105, 0.101, 0.101, 0.102, 0.114, 0.156, 0.199]) # This is half the total error bar length


plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "cm"

fig, axs = plt.subplots(1, 1, sharex=True, rasterized=True,constrained_layout = True)

# Define the values of |k|/k_0
m = 1.31e-24
f_min = 3.17e-10
f_max = 8.27e-7
conversion_factor = 6.5823e-16
lightyear = 100*24*3600*365.25

ratio_min = np.sqrt(1 -(m/ (2*np.pi*f_min*conversion_factor))**2)
ratio_min = 1e-10
ratio_max = np.sqrt(1 -(m/ (2*np.pi*f_max*conversion_factor))**2)

#assuming each mode is contributing equally
omega_T = 1
omega_V = 1
omega_S = .5
beta_V = 1
beta_S = 1

steps = 1000000

# this is the normalization calculated for when |k|/k_0 = 0
beta_T_min = 1/(2.965023154788142e-17)

# this is the normalization calculated for the other freq
beta_T_max = 1/11034649034.984184

limit =np.pi
size = 200
xi_values = np.linspace(0.001, limit, size)

# Compute using Monte Carlo
ratio = .1
f_min = 1/np.sqrt(1 - ratio**2)*m/(2*np.pi) / conversion_factor
fL1 = f_min*lightyear
fL2 = f_min*lightyear

norm = .5

beta_T = 1/1.8849527647214301
gamma_min = np.array([Gamma_T_monte_carlo(beta_T, ratio, xi, fL1, fL2, steps) for xi in xi_values]) + omega_V / omega_T * beta_T / beta_V *np.array([Gamma_V_monte_carlo(beta_V, ratio, xi, fL1, fL2, steps) for xi in xi_values]) + omega_S / omega_T * beta_T / beta_S * np.array([Gamma_S_monte_carlo(beta_S, ratio, xi, fL1, fL2,steps) for xi in xi_values])

normalization = gamma_min[0]
gamma_min = gamma_min/normalization*norm

beta_T = 1/1.8849527647214301
gamma_min_0 = Gamma_effective(ratio, xi_values, omega_S, omega_V, omega_T, beta_S,beta_V,beta_T)
normalization = gamma_min_0[0]
gamma_min_0 = gamma_min_0/normalization*norm

plt.plot(xi_values*180.0/np.pi, gamma_min, color='blue', linestyle='solid', linewidth=2,label=r'$\tilde{\Gamma}_{T}(f_0), \frac{|k|}{k_0} = 0.1$')

beta_T = 3/(4*np.pi) 

plt.plot(xi_values*180.0/np.pi, Gamma_0T_relativistic(1, xi_values, beta_T), color='black', linestyle='dashed', linewidth=2, label='Hellings-Downs')

axs.axhline(0, color='k', lw=0.8)

# Plot the blue line with error bars
plt.errorbar(theta, correlation_coefficient, yerr=y_error, fmt='.', color='C0', capsize=4, label='CPTA Data, f = $f_0$')

# Set labels and title
plt.xlabel(r'$\xi$', fontsize=14)
plt.ylabel('Correlation coefficient', fontsize=14)

plt.xlim(-5, 185)
plt.ylim(-1.1, 1.1)

# Add ticks for better comparison
plt.xticks([0, 30, 60, 90, 120, 150, 180])
plt.yticks(np.arange(-1.0, 1.2, 0.2))


plt.legend(loc='lower left', fontsize=10, frameon=False)

# Display the plot
plt.grid(False)
plt.savefig('figs/fig3.pdf')
plt.show()

