from important_functions import *
import matplotlib.ticker as mticker
import math
import glob, json, sys
import matplotlib.pyplot as plt
import numpy as np


theta = np.array([10, 23.1, 36.7, 48.7,63, 76, 89.3, 102.8, 116.3, 130.2, 142, 156.2, 168.2])
correlation_coefficient = np.array([.14, .32 ,0.206, 0.059, -0.047, 0.004, -.013, 0.08, 0.004, 0.054, -0.009, -0.427, -0.195])
y_error = np.array([0.114, 0.068,  0.068, 0.08, 0.101, 0.093, 0.105, 0.101, 0.101, 0.102, 0.114, 0.156, 0.199]) # This is half the total error bar length

normalizing_amp =  4.501905414729434e-30

xii_mean = np.load('xii_mean.npy')

rho_avg = np.load('rho_avg.npy')
rho_avg = np.array(rho_avg)/normalizing_amp

sig_avg = np.load('sig_avg.npy')
sig_avg = np.array(sig_avg)/normalizing_amp

m = 1.31e-24
conversion_factor = 6.5823e-16
lightyear = 100*24*3600*365.25
norm = .5

#assuming each mode is contributing equally
omega_T = 1
omega_V = 1
omega_S = .5
beta_V = 1
beta_S = 1
beta_T = 1

steps = 1000000


limit =np.pi
size = 200


M = len(theta)
ratios = np.linspace(0.01, .99, 99)
gamma_NG15 = np.empty((99, M)) 
gamma_CPTA = np.empty((99, M))

NG15_chi_squared = np.empty(99)
CPTA_chi_squared = np.empty(99)

index = 0
for ratio in ratios:
    f_max = 1/np.sqrt(1 - ratio**2)*m/(2*np.pi) / conversion_factor
    fL1 = f_max*lightyear
    fL2 = f_max*lightyear

    xi_values = theta * np.pi / 180.0

    gamma = np.array([Gamma_T_monte_carlo(beta_T, ratio, xi, fL1, fL2, steps) for xi in xi_values]) + omega_V / omega_T * beta_T / beta_V *np.array([Gamma_V_monte_carlo(beta_V, ratio, xi, fL1, fL2, steps) for xi in xi_values]) + omega_S / omega_T * beta_T / beta_S * np.array([Gamma_S_monte_carlo(beta_S, ratio, xi, fL1, fL2,steps) for xi in xi_values])
    normalization = gamma[0]
    gamma = gamma/normalization*norm

    gamma_NG15[index] = gamma

    NG15_chi_squared[index] = np.sum(((rho_avg - np.real(gamma)) / (sig_avg))**2)


    xi_values = xii_mean

    gamma = np.array([Gamma_T_monte_carlo(beta_T, ratio, xi, fL1, fL2, steps) for xi in xi_values]) + omega_V / omega_T * beta_T / beta_V *np.array([Gamma_V_monte_carlo(beta_V, ratio, xi, fL1, fL2, steps) for xi in xi_values]) + omega_S / omega_T * beta_T / beta_S * np.array([Gamma_S_monte_carlo(beta_S, ratio, xi, fL1, fL2,steps) for xi in xi_values])
    normalization = gamma[0]
    gamma = gamma/normalization*norm

    gamma_CPTA[index] = gamma

    CPTA_chi_squared[index] = np.sum(((correlation_coefficient - np.real(gamma)) / (y_error))**2)

    index = index + 1

    print(f'Finished {ratio}')

np.save(f'data/gamma_NG15.npy', gamma_NG15)
np.save(f'data/gamma_CPTA.npy', gamma_CPTA)
np.save(f'data/NG_chi_squared.npy', NG15_chi_squared)
np.save(f'data/CPTA_chi_squared.npy', CPTA_chi_squared)

print(f'NG15 min chi^2: {np.min(NG15_chi_squared)}')
print(f'NG15 min ratio: {ratios[np.argmin(NG15_chi_squared)]}')
print(f'CPTA min chi^2: {np.min(CPTA_chi_squared)}')
print(f'CPTA min ratio: {ratios[np.argmin(CPTA_chi_squared)]}')