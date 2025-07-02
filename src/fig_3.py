from important_functions import *
import matplotlib.ticker as mticker
import math

import glob, json, sys
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["font.family"] = "serif"

# loading arrays from file
gamma_CPTA = np.load('data/gamma_CPTA.npy')
gamma_NG15 = np.load('data/gamma_NG15.npy')
NG15_chi_squared = np.load('data/NG15_chi_squared.npy')
CPTA_chi_squared = np.load('data/CPTA_chi_squared.npy')

fig, ax = plt.subplots(1, 1, sharex=True, rasterized=True, constrained_layout = True)
ratios = np.linspace(0.01, .99, 99)

plt.plot(ratios, NG15_chi_squared / 12, label = 'MG fit - NG15', color = 'red')
plt.plot(ratios, CPTA_chi_squared / 12, label = 'MG fit - CPTA', color = 'blue')

NG_HD = np.full((99), 22.20)
CP_HD = np.full((99), 38.95)

red_patch = mpatches.Patch(color='red', 
                             label=r'NANOGrav15')
blue_patch = mpatches.Patch(color='blue', 
                             label=r'CPTA DR1') 
legend_elements = [
    Line2D([0], [0], color='black', linestyle='solid', label=r'MG fit'),
    Line2D([0], [0], color='black', linestyle='dashed', label=r'HD fit'),
]

first_legend = ax.legend(handles=[red_patch, blue_patch]+ legend_elements,loc='upper left',
           frameon=False,prop={'size': 12})

ax.add_artist(first_legend)

plt.plot(ratios, NG_HD/13, label = 'HD fit - NANOGrav15', linestyle = 'dashed', color = 'red')
plt.plot(ratios, CP_HD/13, label = 'HD fit - CPTA', linestyle = 'dashed', color = 'blue')

plt.xlabel(r'$|{\bf k}|/k_0$',fontsize=14)
plt.ylabel(r'$\chi^2$/d.o.f.',fontsize=14)

plt.ylim(0, 6)
plt.xlim(0, 1)

# Display the plot
plt.grid(False)
plt.savefig('figs/fig3.pdf')
plt.show()