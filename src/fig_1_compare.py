import matplotlib.ticker as mticker
import math

import glob, json, sys
import matplotlib.pyplot as plt
import numpy as np
from polarization_func import *
import matplotlib
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["font.family"] = "serif"


fig, ax = plt.subplots(1, 1, sharex=True, rasterized=True, constrained_layout = True)

# Define the values of |k|/k_0
m = 1.31e-24
f_min = 3.17e-10
f_max = 8.27e-7
conversion_factor = 6.5823e-16
lightyear = 100*24*3600*365.25

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

norm = .5

ratio = .5
f_max = 1/np.sqrt(1 - ratio**2)*m/(2*np.pi) / conversion_factor
fL1 = f_max*lightyear
fL2 = f_max*lightyear
beta_T = 1/7132591.489677116

gamma_max = np.array([Gamma_T_monte_carlo(beta_T, ratio, xi, fL1, fL2, steps) for xi in xi_values]) + omega_V / omega_T * beta_T / beta_V *np.array([Gamma_V_monte_carlo(beta_V, ratio, xi, fL1, fL2, steps) for xi in xi_values]) + omega_S / omega_T * beta_T / beta_S * np.array([Gamma_S_monte_carlo(beta_S, ratio, xi, fL1, fL2,steps) for xi in xi_values])
normalization = gamma_max[0]
gamma_max = gamma_max/normalization*norm

gamma_max_0 = Gamma_effective(ratio, xi_values, omega_S, omega_V, omega_T, beta_S,beta_V,beta_T)
normalization = gamma_max_0[0]
gamma_max_0 = gamma_max_0/normalization*norm

plt.plot(xi_values*180.0/np.pi, gamma_min, color='blue', linestyle='solid', linewidth=2)
plt.plot(xi_values*180.0/np.pi, gamma_max, color='red', linestyle='solid', linewidth=2)

plt.plot(xi_values*180.0/np.pi, gamma_min_0, color='blue', linestyle='dashed', linewidth=2)
plt.plot(xi_values*180.0/np.pi, gamma_max_0, color='red', linestyle='dashed', linewidth=2)

beta_T = 3/(4*np.pi)

plt.plot(xi_values*180.0/np.pi, Gamma_0T_relativistic(1, xi_values, beta_T), color='black', linestyle='dashed', linewidth=2)

ax.set_ylabel(r'$\Gamma(\xi_{ab})$')

plt.xticks([0, 30, 60, 90, 120, 150, 180])
plt.yticks(np.arange(-1.0, 1.2, 0.2))

red_patch = mpatches.Patch(color='blue', 
                             label=r'$|{\bf k}|/k_0 = 0.1$') 
blue_patch = mpatches.Patch(color='red', 
                             label=r'$|{\bf k}|/k_0 = 0.5$')
black_patch = mpatches.Patch(color='black', 
                             label='Hellings-Downs')  
legend_elements = [
    Line2D([0], [0], color='black', linestyle='solid', label=r'Using $\mathcal{E}(f,\hat{\bf p})$'),
    Line2D([0], [0], color='black', linestyle='dashed', label=r'Ignoring $\mathcal{E}(f,\hat{\bf p})$'),
]

first_legend = ax.legend(handles=[red_patch, blue_patch, black_patch],loc='lower left',
           frameon=False,prop={'size': 12})
second_legend = plt.legend(handles = legend_elements, loc='lower right',
           frameon=False,prop={'size': 12})
ax.add_artist(first_legend)

# Set labels and title
plt.xlabel(r'$\xi$', fontsize=14)
plt.ylabel(r'$\Gamma$', fontsize=14)

# Set x and y limits to match the original plot
plt.xlim(-5, 185)
plt.ylim(-1.1, 1.1)

# Display the plot
plt.grid(False) 
plt.savefig('figs/fig1.pdf')
plt.show()