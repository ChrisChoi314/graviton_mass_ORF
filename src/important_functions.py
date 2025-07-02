import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import math

# The scalar overlap reduction functions

def Gamma_0S_stationary(ratio, xi, beta_S):  
    return (beta_S / 4) * 4 * np.pi / 15 * (-1 + 3 * np.cos(xi)**2)

def Gamma_0S_full(ratio, xi, beta_S):
    if np.abs(ratio) < 1e-6:
        return Gamma_0S_stationary(ratio, xi,beta_S)
    L_1 = (1 + 2*ratio**2*(1-2*np.cos(xi)) - ratio**4*(1 - 2*np.cos(xi)**2) - 2*ratio*(1-  ratio**2*np.cos(xi))*np.sqrt((1 - np.cos(xi))*(2-ratio**2*(1 + np.cos(xi))) ) )**2/(1-ratio**2)**4
    A = ratio
    factor = (beta_S * np.pi) / (4 * 6 * A**5)
    
    term1 = (4 * A * (9 + 4 * A**4 + 18 * np.cos(xi) - 3 * A**2 * (4 + 5 * np.cos(xi))))
    
    term2 = (-12 * (1 - A**2) * (2 * A**2 - 3 - 3 * np.cos(xi)) * 
             np.log((1 - A) / (1 + A)))
    
    term3 = (-9 * (1 - A**2)**2 * np.log(L_1) / 
             np.sqrt((1 - np.cos(xi)) * (2 - A**2 * (1 + np.cos(xi)))))
    
    return factor * (term1 + term2 + term3)


# The vector overlap reduction functions

def Gamma_0V_stationary(ratio, xi, beta_V):  
    return (beta_V / 4) * 8 * np.pi / 15 * (-1 + 3 * np.cos(xi)**2)

def Gamma_0V_relativistic(ratio, xi, beta_V):
    return beta_V * 8 * np.pi * np.cos(xi)/(4*2*3*(1 - ratio**2))

def Gamma_0V_full(ratio, xi, beta_V):
    if np.abs(ratio) < 1e-2:
        return Gamma_0V_stationary(ratio, xi, beta_V)
    L_1 = (1 + 2*ratio**2*(1-2*np.cos(xi)) - ratio**4*(1 - 2*np.cos(xi)**2) - 2*ratio*(1-  ratio**2*np.cos(xi))*np.sqrt((1 - np.cos(xi))*(2-ratio**2*(1 + np.cos(xi))) ) )**2/(1-ratio**2)**4
    A = ratio
    factor = (beta_V * 8 * np.pi) / (8 * A**5) / (1- A**2)
    
    term1 = (A / 3) * (A**6 * np.cos(xi) - 2 * A**4 * (5 * np.cos(xi) + 3) + 
                       2 * A**2 * (11 * np.cos(xi) + 6) - 6 * (2 * np.cos(xi) + 1))
    
    term2 = ((A**2 - 1)**2 * ((A**2 - 2) * np.cos(xi) - 2) * 
             np.log((1 - A) / (1 + A)))
    
    term3 = ((A**2 - 1)**2 * (1 - A**2 * np.cos(xi)) * np.log(L_1) /
             (2 * np.sqrt((1 - np.cos(xi)) * (2 - A**2 * (1 + np.cos(xi))))))
    
    return factor * (term1 + term2 + term3)


# The tensor overlap reduction functions

def Gamma_0T_stationary(ratio, xi, beta_T):  
    return (beta_T / 4) * (8 * np.pi / 15 * (-1 + 3 * np.cos(xi) ** 2 +
            (8 * np.pi / 105) * ratio ** 2 * (-2 - 3 * np.cos(xi) + 
            6 * np.cos(xi) ** 2 + 5 * np.cos(xi) ** 3)))

def Gamma_0T_relativistic(ratio, xi, beta_T):
    return beta_T/4*2*np.pi/3*(3 + np.cos(xi) + 6*(1 - np.cos(xi))*np.log((1-np.cos(xi))/2))

def Gamma_0T_full(ratio, xi, beta_T):
    if np.abs(ratio) < 1e-6:
        return Gamma_0T_stationary(ratio, xi, beta_T)
    L_1 = (1 + 2*ratio**2*(1-2*np.cos(xi)) - ratio**4*(1 - 2*np.cos(xi)**2) - 2*ratio*(1-  ratio**2*np.cos(xi))*np.sqrt((1 - np.cos(xi))*(2-ratio**2*(1 + np.cos(xi))) ) )**2/(1-ratio**2)**4
    return -np.pi/(6*ratio**5)*beta_T/4*(4*ratio*(-3 + (-6 + 5*ratio**2)*np.cos(xi)) + 12*(1 + np.cos(xi) + ratio**2*(1-3*np.cos(xi)))*np.log((1+ratio)/(1-ratio)) + (3*(1 + 2*ratio**2 *(1 - 2*np.cos(xi))- ratio**4*(1 - 2*np.cos(xi)**2) )*np.log(L_1) )/(np.sqrt((1 - np.cos(xi))*(2 -ratio**2*(1 + np.cos(xi))))) )

# The effective overlap reduction function for the three modes

def Gamma_effective(ratio, xi, Omega_S, Omega_V, Omega_T, beta_S, beta_V, beta_T):
    return Gamma_0T_full(ratio, xi, beta_T) + Gamma_0V_full(ratio, xi, beta_V)*(Omega_V / Omega_T)*(beta_T/beta_V) + Gamma_0S_full(ratio, xi, beta_S)*(Omega_S / Omega_T)*(beta_T/beta_S)


## The monte carlo version of the overlap reduction function

def Gamma_T_monte_carlo(beta_T, k_abs_k_0, xi, fL1, fL2, num_samples=150000):
    cos_xi = np.cos(xi)
    sin_xi = np.sin(xi)

    theta = np.random.uniform(0, np.pi, num_samples)
    phi = np.random.uniform(0, 2 * np.pi, num_samples)

    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)

    denom1 = 2 * (1 + (k_abs_k_0) * cos_theta)
    denom2 = 1 + (k_abs_k_0) * (cos_theta * cos_xi + cos_phi * sin_theta * sin_xi)

    numerator = (
        cos_xi**2 * sin_theta**2
        - 2 * cos_theta * sin_theta * cos_xi * sin_xi * cos_phi
        + sin_xi**2 * (cos_theta**2 * cos_phi**2 - sin_phi**2)
    )

    exp1 = np.exp(1j * 2 * np.pi * fL1 * (1 + (k_abs_k_0) * cos_theta)) - 1
    exp2 = np.exp(-1j * 2 * np.pi * fL2 * (1 + (k_abs_k_0) * (cos_theta * cos_xi + cos_phi * sin_theta * sin_xi))) - 1

    #exp1 = 1
    #exp2 = 1

    integrand = (sin_theta**2 / denom1) * (numerator / denom2) * exp1 * exp2 * sin_theta

    avg_value = np.mean(integrand)
    area = 4 * np.pi**2

    return (beta_T / 4) * avg_value * area


def integrand_expression_V(theta, phi, A, xi):
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    cos_xi = np.cos(xi)
    sin_xi = np.sin(xi)

    denom = (1 + A * cos_theta) * (1 + A * (cos_theta * cos_xi + cos_phi * sin_theta * sin_xi))

    term1 = ((A + (2 - A) * cos_theta) *
             (A + (2 - A) * cos_theta * cos_xi) * cos_xi * sin_theta**2)

    term2 = (2 - A**2) * (A + (2 - A) * cos_theta) * cos_theta * sin_theta**2 * sin_xi**2 * cos_phi**2

    term3 = ((A + (2 - A) * cos_theta) *
             (A * cos_theta + (2 - A**2) * np.cos(2 * theta) * cos_xi) *
             sin_theta * sin_xi + 1) * cos_phi

    result = (term1 - term2 - term3) / denom
    return result


def Gamma_V_monte_carlo(beta_V, k_abs_k_0, xi, fL1, fL2, num_samples=150000):
    cos_xi = np.cos(xi)
    sin_xi = np.sin(xi)
    
    theta = np.random.uniform(0, np.pi, num_samples)
    phi = np.random.uniform(0, 2 * np.pi, num_samples)

    A = k_abs_k_0
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)

    exp1 = np.exp(1j * 2 * np.pi * fL1 * (1 + (k_abs_k_0) * cos_theta)) - 1
    exp2 = np.exp(-1j * 2 * np.pi * fL2 * (1 + (k_abs_k_0) * (cos_theta * cos_xi + cos_phi * sin_theta * sin_xi))) - 1

    integrand_vals = integrand_expression_V(theta, phi, A, xi) * np.sin(theta) * exp1 * exp2


    integral_avg = np.mean(integrand_vals)
    area = 4 * np.pi **2  # area of integration domain

    prefactor = (beta_V ) / (4 * 2)/ (1- A**2)
    return prefactor * integral_avg * area


def integrand_expression_S(theta, phi, A, xi):
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    cos_2theta = np.cos(2 * theta)
    cos_2xi = np.cos(2 * xi)
    cos_xi = np.cos(xi)
    sin_xi = np.sin(xi)

    # Numerator and denominator
    numerator = (1 + 4 * A * cos_theta + 3 * cos_2theta)
    denom1 = 1 + A * cos_theta
    denom2 = 1 + A * (cos_theta * cos_xi + cos_phi * sin_theta * sin_xi)
    denom = denom1 * denom2

    # Expression in large parentheses
    poly_term = (
        (-2 + 6 * cos_2theta + 6 * cos_2xi + 6 * cos_2theta * cos_2xi + 16 * A * cos_theta * cos_xi) / 8
        + 2 * (A + 3 * cos_theta * cos_xi) * sin_theta * sin_xi * cos_phi
        + 3 * sin_theta**2 * sin_xi**2 * cos_phi**2
    )

    return numerator * poly_term / denom

# Monte Carlo integration function for Γ₀,s
def Gamma_S_monte_carlo(beta_S, k_abs_k_0, xi, fL1, fL2, num_samples=150000):
    theta = np.random.uniform(0, np.pi, num_samples)
    phi = np.random.uniform(0, 2 * np.pi, num_samples)

    A = k_abs_k_0

    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    cos_xi = np.cos(xi)
    sin_xi = np.sin(xi)
    
    exp1 = np.exp(1j * 2 * np.pi * fL1 * (1 + (k_abs_k_0) * cos_theta)) - 1
    exp2 = np.exp(-1j * 2 * np.pi * fL2 * (1 + (k_abs_k_0) * (cos_theta * cos_xi + cos_phi * sin_theta * sin_xi))) - 1

    #exp1 = 1
    #exp2 = 1

    integrand_vals = integrand_expression_S(theta, phi, A, xi) * np.sin(theta) * exp1 * exp2
    #integrand_vals = integrand_vals.astype(np.complex128)

    integral_avg = np.mean(integrand_vals)
    area = 4 * np.pi **2  # integration domain area

    prefactor = beta_S * (1 / 4) * (1 / 12)
    return prefactor * integral_avg * area

# Hellings-Downs curve
def hd(angseps):
    xx = 0.5 * (1-np.cos(angseps))
    return 1.5*xx*np.log(xx) - 0.25*xx + 0.5