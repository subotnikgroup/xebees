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
def soft_coulomb(R, r1e, r2e, charges, dv=1, G=0.02):
    Q1, Q2 = charges

    V1 = -G * Q1      / np.sqrt(r1e**2 + dv**2)
    V2 = -G * Q2      / np.sqrt(r2e**2 + dv**2)
    VN =  G * Q1 * Q2 / np.sqrt(R**2   + dv**2)
    return V1 + V2 + VN


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

