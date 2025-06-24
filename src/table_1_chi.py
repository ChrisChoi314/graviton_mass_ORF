# Re-import necessary libraries after code execution environment reset
import matplotlib.ticker as mticker
import math

import glob, json, sys
import matplotlib.pyplot as plt
import numpy as np
from polarization_func import *

from enterprise.signals import parameter
from enterprise.signals import utils
from enterprise.signals import signal_base
from enterprise.signals import selections
from enterprise.signals import white_signals
from enterprise.signals import gp_signals

from enterprise_extensions.models import model_general
from enterprise_extensions import hypermodel
from enterprise_extensions import sampler as ee_sampler
from extra_functions import EmpiricalDistribution2D
from PTMCMCSampler.PTMCMCSampler import PTSampler as ptmcmc

from la_forge import core, diagnostics

from h5pulsar.pulsar import FilePulsar

import scipy.stats as scistats

from optimal_statistic_covariances import OS, full_lstsq_recovery
from enterprise_extensions.models import model_2a
import matplotlib

plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["font.family"] = "serif"

theta = np.array([10, 23.1, 36.7, 48.7,63, 76, 89.3, 102.8, 116.3, 130.2, 142, 156.2, 168.2])
correlation_coefficient = np.array([.14, .32 ,0.206, 0.059, -0.047, 0.004, -.013, 0.08, 0.004, 0.054, -0.009, -0.427, -0.195])
y_error = np.array([0.114, 0.068,  0.068, 0.08, 0.101, 0.093, 0.105, 0.101, 0.101, 0.102, 0.114, 0.156, 0.199]) # This is half the total error bar length

normalizing_amp =  4.501905414729434e-30

def hd(angseps):
    xx = 0.5 * (1-np.cos(angseps))
    return 1.5*xx*np.log(xx) - 0.25*xx + 0.5

xii_mean = np.load('data/xii_mean.npy')

rho_avg = np.load('data/rho_avg.npy')
rho_avg = np.array(rho_avg)/normalizing_amp

sig_avg = np.load('data/sig_avg.npy')
sig_avg = np.array(sig_avg)/normalizing_amp
print(sig_avg)

chi_NG_HD = np.sum(((rho_avg - hd(xii_mean)) / (sig_avg) )**2)                            

chi_CP_HD = np.sum(((correlation_coefficient - hd(theta*np.pi/180.0)) / (y_error))**2 )

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

# Compute using Monte Carlo
ratio = .1
f_min = 1/np.sqrt(1 - ratio**2)*m/(2*np.pi) / conversion_factor
fL1 = f_min*lightyear
fL2 = f_min*lightyear
beta_T = beta_T_min
beta_T = 1/0.45
beta_T = 1/1.8849527647214301*.5
beta_T = 1/5.759871885918659e+36
beta_T = 1/1.8849527647214301

norm = .5

xi_values = theta * np.pi / 180.0

beta_T = 1/1.8849527647214301
gamma_min = np.array([Gamma_T_monte_carlo(beta_T, ratio, xi, fL1, fL2, steps) for xi in xi_values]) + omega_V / omega_T * beta_T / beta_V *np.array([Gamma_V_monte_carlo(beta_V, ratio, xi, fL1, fL2, steps) for xi in xi_values]) + omega_S / omega_T * beta_T / beta_S * np.array([Gamma_S_monte_carlo(beta_S, ratio, xi, fL1, fL2,steps) for xi in xi_values])

normalization = gamma_min[0]
gamma_min = gamma_min/normalization*norm

chi_CP_MG = np.sum(((correlation_coefficient - np.real(gamma_min)) / (y_error))**2)

beta_T = 1/1.8849527647214301
gamma_min_0 = Gamma_effective(ratio, xi_values, omega_S, omega_V, omega_T, beta_S,beta_V,beta_T)
normalization = gamma_min_0[0]
gamma_min_0 = gamma_min_0/normalization*norm

ratio = .5
f_max = 1/np.sqrt(1 - ratio**2)*m/(2*np.pi) / conversion_factor
fL1 = f_max*lightyear
fL2 = f_max*lightyear
beta_T = 1/7132591.489677116

xi_values = xii_mean

gamma_max = np.array([Gamma_T_monte_carlo(beta_T, ratio, xi, fL1, fL2, steps) for xi in xi_values]) + omega_V / omega_T * beta_T / beta_V *np.array([Gamma_V_monte_carlo(beta_V, ratio, xi, fL1, fL2, steps) for xi in xi_values]) + omega_S / omega_T * beta_T / beta_S * np.array([Gamma_S_monte_carlo(beta_S, ratio, xi, fL1, fL2,steps) for xi in xi_values])
normalization = gamma_max[0]
gamma_max = gamma_max/normalization*norm

chi_NG_MG = np.sum(((rho_avg - np.real(gamma_max)) / (sig_avg))**2)

print(f'NANOGrav: chi^2_HD = {chi_NG_HD}')
print(f'NANOGrav: chi^2_MG = {chi_NG_MG}')
print(f'CPTA: chi^2_HD = {chi_CP_HD}')
print(f'CPTA: chi^2_MG = {chi_CP_MG}')

dof_HD = 13 - 0
dof_MG = 13 - 2

print(f'NANOGrav: chi^2_HD/dof = {chi_NG_HD/dof_HD}')
print(f'NANOGrav: chi^2_MG/dof = {chi_NG_MG/dof_MG}')
print(f'CPTA: chi^2_HD/dof = {chi_CP_HD/dof_HD}')
print(f'CPTA: chi^2_MG/dof = {chi_CP_MG/dof_MG}')