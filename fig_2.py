import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

#if you want to compute the NANOGrav15 data points manually from the raw data, uncomment the following packages
'''
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
'''

#if you want to compute the effective ORF from MG numerically using Monte-Carlo integration, uncomment the following packages
'''
import glob, json, sys
from polarization_func import *
'''

plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["font.family"] = "serif"


fig, (ax_nga, ax_cpta) = plt.subplots(
    1, 2,
    sharey=True,
    figsize=(10, 4),
    gridspec_kw={"wspace": 0}
)
fig.subplots_adjust(wspace=0)

hd_proxy = Line2D([], [], ls="--", color="k", lw=2)

leg = fig.legend(
    [hd_proxy], ["Hellings – Downs"],
    loc="upper center",
    bbox_to_anchor=(0.5, 0.85),       
    bbox_transform=fig.transFigure,
    frameon=True,                        
    handlelength=3,
    fontsize=10,
    borderaxespad=0.0
)

# makes frame invisible but keeps opaque background
leg.get_frame().set_facecolor("white")   
leg.get_frame().set_edgecolor("none")   
leg.get_frame().set_alpha(1)      
leg.set_zorder(10)        

# Hellings-Downs curve
def hd(angseps):
    xx = 0.5 * (1-np.cos(angseps))
    return 1.5*xx*np.log(xx) - 0.25*xx + 0.5

#if you want to compute the NANOGrav15 data points manually from the raw data, uncomment the following lines. It will automatically save the data, so you only need to run it once.
'''
datapath = './data/figure2_data/'
pulsar_datapath = './data/data_pulsar/'

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
'''

xi_values = np.load("data/xi_values.npy")
normalizing_amp =  4.501905414729434e-30

xii_mean = np.load('data/xii_mean.npy')

rho_avg = np.load('data/rho_avg.npy')
rho_avg = np.array(rho_avg)/normalizing_amp

sig_avg = np.load('data/sig_avg.npy')
sig_avg = np.array(sig_avg)/normalizing_amp

theta = np.array([10, 23.1, 36.7, 48.7,63, 76, 89.3, 102.8, 116.3, 130.2, 142, 156.2, 168.2])
correlation_coefficient = np.array([.14, .32 ,0.206, 0.059, -0.047, 0.004, -.013, 0.08, 0.004, 0.054, -0.009, -0.427, -0.195])
y_error = np.array([0.114, 0.068,  0.068, 0.08, 0.101, 0.093, 0.105, 0.101, 0.101, 0.102, 0.114, 0.156, 0.199]) # This is half the total error bar length

### fig 2, left side  –  NANOGrav 15‑yr
# if you want to compute the effective ORF from MG numerically using Monte-Carlo integration, uncomment the following lines
'''
# Defining important values
m = 1.31e-24
conversion_factor = 6.5823e-16
lightyear = 100*24*3600*365.25

# assuming each mode is contributing equally
omega_T = 1
omega_V = 1
omega_S = .5
beta_V = 1
beta_S = 1
beta_T = 1
norm = .5
steps = 1000000

limit =np.pi
size = 200
xi_values = np.linspace(0.001, limit, size)

ratio = .61
f_max = 1/np.sqrt(1 - ratio**2)*m/(2*np.pi) / conversion_factor
fL1 = f_max*lightyear
fL2 = f_max*lightyear

#gamma_max = np.array([Gamma_T_monte_carlo(beta_T, ratio, xi, fL1, fL2, steps) for xi in xi_values]) + omega_V / omega_T * beta_T / beta_V *np.array([Gamma_V_monte_carlo(beta_V, ratio, xi, fL1, fL2, steps) for xi in xi_values]) + omega_S / omega_T * beta_T / beta_S * np.array([Gamma_S_monte_carlo(beta_S, ratio, xi, fL1, fL2,steps) for xi in xi_values])
#normalization = gamma_max[0]
#gamma_max = gamma_max/normalization*norm
'''

gamma_max = np.load("data/gamma_61.npy")
ax_nga.plot(
    xi_values * 180.0 / np.pi,
    gamma_max,
    color="red",
    linewidth=2,
    label=r"$\tilde{\Gamma}_{T}, |{\bf k}|/k_0 = 0.61$",
)
ax_nga.errorbar(
    180 / np.pi * xii_mean,
    rho_avg,
    yerr=sig_avg,
    fmt=".",
    color="C0",
    capsize=4,
    label="NANOGrav15",
)
ax_nga.plot(
    xi_values * 180.0 / np.pi,
    hd(xi_values),
    lw=2,
    color="k",
    ls="dashed",
    
)
ax_nga.set_xlabel(r"$\xi$", fontsize=12)
ax_nga.set_ylabel(r"$\Gamma$", fontsize=12)  
ax_nga.set_xlim(-5, 185)
ax_nga.set_ylim(-1.1, 1.1)
ax_nga.set_xticks([0, 30, 60, 90, 120, 150, 180])
ax_nga.set_yticks(np.arange(-1.0, 1.2, 0.2))
ax_nga.axhline(0, color="k", lw=0.8)
ax_nga.legend(loc='lower left', frameon=False, fontsize=10)

### fig 2, right side –  CPTA DR1
# if you want to compute the effective ORF from MG numerically using Monte-Carlo integration, uncomment the following lines
'''
ratio = .01
f_min = 1/np.sqrt(1 - ratio**2)*m/(2*np.pi) / conversion_factor
fL1 = f_min*lightyear
fL2 = f_min*lightyear

gamma_min = np.array([Gamma_T_monte_carlo(beta_T, ratio, xi, fL1, fL2, steps) for xi in xi_values]) + omega_V / omega_T * beta_T / beta_V *np.array([Gamma_V_monte_carlo(beta_V, ratio, xi, fL1, fL2, steps) for xi in xi_values]) + omega_S / omega_T * beta_T / beta_S * np.array([Gamma_S_monte_carlo(beta_S, ratio, xi, fL1, fL2,steps) for xi in xi_values])
normalization = gamma_min[0]
gamma_min = gamma_min/normalization*norm
'''
gamma_min = np.load("monte_carlo_arrays/gamma_01.npy")
ax_cpta.plot(
    xi_values * 180.0 / np.pi,
    gamma_min,
    color="blue",
    linewidth=2,
    label=r"$\tilde{\Gamma}_{T}, |{\bf k}|/k_0 = 0.01$",
)
ax_cpta.errorbar(
    theta,
    correlation_coefficient,
    yerr=y_error,
    fmt=".",
    color="C0",
    capsize=4,
    label="CPTA DR1, $f=f_0$",
)
ax_cpta.plot(
    xi_values * 180.0 / np.pi,
    hd(xi_values),
    lw=2,
    color="k",
    ls="dashed",
)
ax_cpta.set_xlabel(r"$\xi$", fontsize=12)
ax_cpta.set_xlim(-5, 185)
ax_cpta.axhline(0, color="k", lw=0.8)
ax_cpta.set_xticks([0, 30, 60, 90, 120, 150, 180])

# Hide the duplicate y‑axis label on the right
ax_cpta.set_ylabel("")
ax_cpta.tick_params(axis="y", labelleft=False)

ax_cpta.legend(loc='lower left',frameon=False, fontsize=10)

plt.savefig("figs/fig2.pdf")
plt.show()
