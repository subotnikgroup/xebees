
from constants import *
import numpy as np

## README ##
# Functions in this file should be defined to take the *lab-frame*
# inter-particle distances in units of Bohr, the charges on the
# massive particles (as a tuple), and then any named parameters *with
# their defaults*

## Example declaration ##
#
# def somename(R, r1e, r2e, charges, params=...):
#     Q1, Q2 = charges
#     ...


# Soft Coulomb potential; dv control softness. G is a term of
# dimension mass that effectively scales the competition between
# kinetic and potential energy
def soft_coulomb(R, r1e, r2e, charges, dv=0.5, G=1, p=1):
    Q1, Q2 = charges

    V1  = -Q1      / (r1e**2 + dv**2)**(p/2)
    V2  = -Q2      / (r2e**2 + dv**2)**(p/2)
    VN  =  Q1 * Q2 / (R**2   + dv**2)**(p/2)
    return G*(V1 + V2 + VN)

def soft_coulomb_barrier(R, r1e, r2e, charges, dv=0.5, G=1, p=2, A=1):
    Q1, Q2 = charges

    V1 =  -Q1      / (r1e**p + dv**p)**(1/p)
    V2 =  -Q2      / (r2e**p + dv**p)**(1/p)
    VN =   Q1 * Q2 / (R**p   + dv**p)**(1/p)
    Vbar = Q1 * Q2 / 4 / R**2
    return G*(V1 + V2 + VN + A*Vbar)

def soft_coulomb_exp(R, r1e, r2e, charges, dv=0.5, G=1, p=2, alpha=0.15, A=2):
    Q1, Q2 = charges

    V1 =  -Q1      / (r1e**p + dv**p)**(1/p)
    V2 =  -Q2      / (r2e**p + dv**p)**(1/p)
    VN =   Q1 * Q2 / (R**p   + dv**p)**(1/p)
    Vexp = Q1 * Q2 * A * np.exp(-R/alpha) / R**2
    return G*(V1 + V2 + VN + Vexp)



def harmonic(R, r1e, r2e, charges, w=1, R0=0):
    V = 0.5 * w**2 * (R - R0)**2
    return V

# Original potential used by Xuezhi with the addition of
# charges. Adapted from Borgis, CPL 423 (2006). In their paper,
# asymmetry_param is \gamma and taken to be either 1 or 0.707

# Chemical Physics Letters 423 (2006) 390–394
# doi:10.1016/j.cplett.2006.04.007
def borgis(R_au, r1e_au, r2e_au, charges, asymmetry_param=1):
    Q1, Q2 = charges

    R   = R_au   / ANGSTROM_TO_BOHR
    r1e = r1e_au / ANGSTROM_TO_BOHR
    r2e = r2e_au / ANGSTROM_TO_BOHR

    D, d, a, c = 60, 0.95, 2.52, asymmetry_param
    A, B, C = 2.32e5, 3.15, 2.31e4

    D2 = Q2 * D * (     np.exp(-2*a * (r2e-d))
                    - 2*np.exp(  -a * (r2e-d))
                    + 1)
    D1 = Q1 * D * c**2 * (     np.exp(-(2*a/c) * (r1e-d))
                           - 2*np.exp(-(  a/c) * (r1e-d)))

    VN = Q1 * Q2 * (A*np.exp(-B*R) - C/R**6)

    return KCALMOLE_TO_HARTREE * (D1 + D2 + VN)


# Same potential as the one we used in 1-D; charges added, but only
# "seen" by electron. Adapted from Borgis
def original(R_au, r1e_au, r2e_au, charges, asymmetry_param=1):
    Q1, Q2 = charges

    R   = R_au   / ANGSTROM_TO_BOHR
    r1e = r1e_au / ANGSTROM_TO_BOHR
    r2e = r2e_au / ANGSTROM_TO_BOHR

    D, d, a, c = 60, 0.95, 2.52, asymmetry_param
    A, B, C = 2.32e5, 3.15, 2.31e4

    D2 = Q2 * D * (    np.exp(-2*a * (r2e-d))
                   - 2*np.exp(  -a * (r2e-d))
                   + 1)
    D1 = Q1 * D * c**2 * (    np.exp(-(2*a/c) * (r1e-d))
                          - 2*np.exp(-(  a/c) * (r1e-d)))

    return KCALMOLE_TO_HARTREE * (D1 + D2 + (A*np.exp(-B*R) - C/R**6))
