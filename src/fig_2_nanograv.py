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


fig, axs = plt.subplots(1, 1, sharex=True, rasterized=True, constrained_layout = True)

# Estimated data points and error bar values from the image
theta = np.array([10, 23.1, 36.7, 48.7,63, 76, 89.3, 102.8, 116.3, 130.2, 142, 156.2, 168.2])

# Define the values of |k|/k_0
m = 1.31e-24
f_min = 3.17e-10
f_max = 8.27e-7
#f_max = np.sqrt(f_min*f_max)
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
beta_T = 1/1.8849527647214301

norm = .5

ratio = .5
f_max = 1/np.sqrt(1 - ratio**2)*m/(2*np.pi) / conversion_factor
fL1 = f_max*lightyear
fL2 = f_max*lightyear
beta_T = 1/7132591.489677116

gamma_max = np.array([Gamma_T_monte_carlo(beta_T, ratio, xi, fL1, fL2, steps) for xi in xi_values]) + omega_V / omega_T * beta_T / beta_V *np.array([Gamma_V_monte_carlo(beta_V, ratio, xi, fL1, fL2, steps) for xi in xi_values]) + omega_S / omega_T * beta_T / beta_S * np.array([Gamma_S_monte_carlo(beta_S, ratio, xi, fL1, fL2,steps) for xi in xi_values])
normalization = gamma_max[0]
gamma_max = gamma_max/normalization*norm

plt.plot(xi_values*180.0/np.pi, gamma_max, color='red', linestyle='solid', linewidth=2, label=r'$\tilde{\Gamma}_{T}(f_1), \frac{|k|}{k_0} = 0.5$')

beta_T = 3/(4*np.pi) 

theta = np.array([10, 23.1, 36.7, 48.7,63, 76, 89.3, 102.8, 116.3, 130.2, 142, 156.2, 168.2])

datapath = './figure1_data/'
pulsar_datapath = './data/'

# Hellings-Downs curve
def hd(angseps):
    xx = 0.5 * (1-np.cos(angseps))
    return 1.5*xx*np.log(xx) - 0.25*xx + 0.5

psrs = []
for hdf5_file in glob.glob(pulsar_datapath + '/hdf5/*.hdf5'):
    psrs.append(FilePulsar(hdf5_file))
print('Loaded {0} pulsars from hdf5 files'.format(len(psrs)))

noise_dict_file = datapath + 'v1p1_wn_dict.json'
noise_dictionary = json.load(open(noise_dict_file, 'r'))

pta = model_2a(psrs, noisedict=noise_dictionary, tm_marg=False, 
               psd='powerlaw', n_rnfreqs=30, n_gwbfreqs=14, gamma_common=13.0/3.0)

ml_os_vector = json.load(open(datapath + 'optstat_ml_gamma4p33.json', 'r'))

os_obj = OS(psrs, pta, ml_os_vector)

print(f"A^2 estimate: {os_obj.os()} +/- {os_obj.os_sigma()} with SNR: {os_obj.snr()}") # for some reason, this line has to be here or else it doesnt work

covariance_matix_between_rhos = np.load(datapath + 'os_covariance_matix_between_rhos.npy')

a_hat, a_covariance = full_lstsq_recovery(os_obj, covariance_matix_between_rhos)

Crho = covariance_matix_between_rhos.copy()

# compute weighted average of correlation data
def weightedavg(rho, sig):
    weights, avg = 0., 0.
    for r,s in zip(rho,sig):
        weights += 1./(s*s)
        avg += r/(s*s)
        
    return avg/weights, np.sqrt(1./weights)

A2_curn = 10**(2*ml_os_vector['gw_log10_A'])
normalizing_amp = A2_curn

xii_bins = np.array([.95, 16.55,29.9,42.7, 55.85, 69.6,82.5,96, 109.5, 123.25,136.1, 149.1, 162.1, 178.8])*np.pi/180.0
nbins_cdf = len(xii_bins) 

import matplotlib.patches as mpatches
    
### 
xii, rho, sig, hd_coeffs = os_obj.angles, os_obj.rhos, os_obj.sigmas, os_obj.orfs

bins = xii_bins 
bin_inds = np.digitize(xii,bins)
bin_inds = np.digitize(xii,bins)-1

xii_mean = []
xii_err = []

## uncorrelated pairs
rho_avg = []
sig_avg = []

## correlated pairs
rho_avg_corr = []
sig_avg_corr = []

npairs = []
for ii in range(len(xii_bins) - 1):
    
    mask = bin_inds == ii
    npairs.append(np.sum(mask))

    xii_mean.append(np.mean(xii[mask]))
    xii_err.append(np.std(xii[mask]))

    r, s = weightedavg(rho[mask], sig[mask])
    rho_avg.append(r)
    sig_avg.append(s)
    
    ubin = os_obj.orfs[mask] 
    rho_tmp = rho[mask] 
    hd_fac = hd(xii_mean[ii]) 
    
    Cbin = Crho[mask,:][:,mask] 
    Xmat = (ubin.T @ np.linalg.inv(Cbin) @ ubin)**(-1.0)
    rho_avg_corr.append( hd_fac * Xmat * (ubin.T @ np.linalg.inv(Cbin) @ 
                                 rho_tmp) )
    sig_avg_corr.append( np.abs(hd_fac) * Xmat**0.5 )


    
xii_mean = np.array(xii_mean)
xii_err = np.array(xii_err)
np.save('data/xii_mean.npy',xii_mean)
np.save('data/rho_avg.npy',rho_avg)
np.save('data/sig_avg.npy',sig_avg)
                            

(_, caps, _) = axs.errorbar(180/np.pi*xii_mean,
                            np.array(rho_avg)/normalizing_amp, 
                            yerr=np.array(sig_avg)/normalizing_amp, 
                            ls='', color='C0',
                            fmt='.', capsize=4,
                            alpha=1.0, label='NANOGrav 15-yr')

axs.axhline(0, color='k', lw=0.8)

# over-plot the HD curve 
idx = np.argsort(xii)
axs.plot(180/np.pi*xii[idx], hd_coeffs[idx], 
         lw=2, color='k', ls='dashed', label='Hellings-Downs')


axs.set_ylabel(r'$\Gamma(\xi_{ab})$')

# Add ticks for better comparison
plt.xticks([0, 30, 60, 90, 120, 150, 180])
plt.yticks(np.arange(-1.0, 1.2, 0.2))

# Set labels and title
plt.xlabel(r'$\xi$', fontsize=14)
plt.ylabel('Correlation coefficient', fontsize=14)

plt.xlim(-5, 185)
plt.ylim(-1.1, 1.1)

plt.legend(loc='lower left', fontsize=10, frameon=False)

# Display the plot
plt.grid(False)
plt.savefig('figs/fig2.pdf')
plt.show()