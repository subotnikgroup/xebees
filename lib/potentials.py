import xp
from scipy.interpolate import PchipInterpolator
from constants import *

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

# You can use _extents_log_factory to build a function defining the
# appropriate extents in R for your potential as well. simply pass 3
# empirically determined array-likes containing 1) the reduced mass,
# 2) the lower bound, and 3) the upper bound. The resulting function
# will monotonically log-interpolate between them. See
# extents_soft_coulumb below.

def _extents_log_factory(mu12ref, lower, upper, decimals=3):
    if any(xp.diff(mu12ref) < 0):
        raise RuntimeError("mu12ref must be monotonic")

    def extents(mu12):
        if mu12 < mu12ref[0] or mu12 > mu12ref[-1]:
            print(f"WARNING: extents may be invalid for mu12 outside of [{mu12ref[0]},{mu12ref[-1]}]")

        logm = xp.log(mu12)
        logmref = xp.log(mu12ref)

        up_int = PchipInterpolator(logmref, upper)
        lo_int = PchipInterpolator(logmref, lower)

        lo = lo_int(logm)
        up = up_int(logm)

        return xp.round([lo, up, up], decimals)
    return extents


# Soft Coulomb potential; dv controls softness
def soft_coulomb(R, r1e, r2e, charges, dv=0.5):
    Q1, Q2 = charges

    V1  = -Q1      / xp.sqrt(r1e**2 + dv**2)
    V2  = -Q2      / xp.sqrt(r2e**2 + dv**2)
    VN  =  Q1 * Q2 / xp.sqrt(R**2   + dv**2)
    return V1 + V2 + VN

extents_soft_coulomb = _extents_log_factory(
    [1e1, 1e2, 1e3, 1e4, 1e5],
    [0.3, 0.3, 0.5, 1,   1],
    [8,   5,   4,   3.5, 3.5]
)


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
    Vexp = Q1 * Q2 * A * xp.exp(-R/alpha) / R**2
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

    D2 = Q2 * D * (     xp.exp(-2*a * (r2e-d))
                    - 2*xp.exp(  -a * (r2e-d))
                    + 1)
    D1 = Q1 * D * c**2 * (     xp.exp(-(2*a/c) * (r1e-d))
                           - 2*xp.exp(-(  a/c) * (r1e-d)))

    VN = Q1 * Q2 * (A*xp.exp(-B*R) - C/R**6)

    return KCALMOLE_TO_HARTREE * (D1 + D2 + VN)

extents_borgis = _extents_log_factory(
    xp.array([1,   2,   10,  50,  1e2, 1e3, 1e4, 1e5])*AMU_TO_AU,
             [3.5, 3.5, 3.9, 4.3, 4.0, 4.1, 4.3, 4.3],
             [7,   6,   5.6, 5.5, 5.4, 5.2, 5.0, 5.0]
)


# Same potential as the one we used in 1-D; charges added, but only
# "seen" by electron. Adapted from Borgis
def original(R_au, r1e_au, r2e_au, charges, asymmetry_param=1):
    Q1, Q2 = charges

    R   = R_au   / ANGSTROM_TO_BOHR
    r1e = r1e_au / ANGSTROM_TO_BOHR
    r2e = r2e_au / ANGSTROM_TO_BOHR

    D, d, a, c = 60, 0.95, 2.52, asymmetry_param
    A, B, C = 2.32e5, 3.15, 2.31e4

    D2 = Q2 * D * (    xp.exp(-2*a * (r2e-d))
                   - 2*xp.exp(  -a * (r2e-d))
                   + 1)
    D1 = Q1 * D * c**2 * (    xp.exp(-(2*a/c) * (r1e-d))
                          - 2*xp.exp(-(  a/c) * (r1e-d)))

    return KCALMOLE_TO_HARTREE * (D1 + D2 + (A*xp.exp(-B*R) - C/R**6))
