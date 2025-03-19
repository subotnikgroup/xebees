#!/usr/bin/env python
import numpy as np
from sys import stderr
import argparse as ap
from pathlib import Path
from pyscf import lib as pyscflib

from constants import *
from hamiltonian import  KE, KE_FFT
from davidson import solve_davidson, solve_exact, get_davidson_guess
from debug import prms, timer

# FIXME: this is busted!
def V2D_fcm(R_amu, r_amu, g, g_1, g_2, M_1, M_2, m_e):
    R = R_amu / ANGSTROM_TO_BOHR
    r = r_amu / ANGSTROM_TO_BOHR
    
    D, d, a, c = 60, 0.95, 2.52, 1
    A, B, C = 2.32e5, 3.15, 2.31e4
    
    mu12 = M_1*M_2/(M_1+M_2)
    mu = np.sqrt(M_1*M_2*m_e/(M_1+M_2+m_e))
    aa = np.sqrt(mu/mu12) # factor of a for lab and scaled coordinates
    
    kappa2 = r*R*np.cos(g)
    r1e = np.sqrt((aa*r)**2 + (R/aa)**2*(mu12/M_1)**2 - 2*kappa2*mu12/M_1)
    re2 = np.sqrt((aa*r)**2 + (R/aa)**2*(mu12/M_2)**2 + 2*kappa2*mu12/M_2)

    D2 = g_2 * D * (    np.exp(-2*a * (re2-d))
                    - 2*np.exp(  -a * (re2-d))
                    + 1)
    D1 = g_1 * D * c**2 * (    np.exp(-(2*a/c) * (r1e-d))
                           - 2*np.exp(-(  a/c) * (r1e-d)))

    return KCALMOLE_TO_HARTREE * (D1 + D2 + A*np.exp(-B*R/aa) - C/(R/aa)**6)


def parse_args():
    parser = ap.ArgumentParser(
        prog='3body-2D',
        description="computes the lowest k eigenvalues of a 3-body potential in 2D")
    
    parser.add_argument('-k', metavar='num_eigenvalues', default=5, type=int)
    parser.add_argument('-t', metavar="num_threads", default=16, type=int)
    parser.add_argument('-g_1', metavar='g_1', required=True, type=float)
    parser.add_argument('-g_2', metavar='g_2', required=True, type=float)
    parser.add_argument('-M_1', required=True, type=float)
    parser.add_argument('-M_2', required=True, type=float)
    parser.add_argument('-J', default=0, type=float)
    parser.add_argument('-R', dest="NR", metavar="NR", default=101, type=int)
    parser.add_argument('-r', dest="Nr", metavar="Nr", default=400, type=int)
    parser.add_argument('-g', dest="Ng", metavar="Ng", default=400, type=int)
    parser.add_argument('--exact_diagonalization', action='store_true')
    parser.add_argument('--verbosity', default=2, type=int)
    parser.add_argument('--iterations', metavar='max_iterations', default=10000, type=int)
    parser.add_argument('--subspace', metavar='max_subspace', default=1000, type=int)
    parser.add_argument('--guess', metavar="guess.npz", type=Path, default=None)
    parser.add_argument('--evecs', metavar="guess.npz", type=Path, default=None)
    parser.add_argument('--save', metavar="filename")

    return parser.parse_args()


def build_terms(args):
    m_e = AMU_TO_AU * 1
    M_1 = AMU_TO_AU * args.M_1
    M_2 = AMU_TO_AU * args.M_2

    # Grid setup
    # FIXME: need to pick ranges based on masses
    range_R = (1e-5, 4.0)
    range_r = (0, 4)
    range_g = (0, np.pi/2)
    
    # as for imshow, (left, right, bottom, top)
    if hasattr(args, "extent") and args.extent is not None:
        range_R = args.extent[:2]
        range_r = args.extent[2:4]

    R = np.linspace(*range_R, args.NR) * ANGSTROM_TO_BOHR
    r = np.linspace(*range_r, args.Nr) * ANGSTROM_TO_BOHR
    g = np.linspace(*range_g, args.Ng)

    dR, dr, dg = R[1] - R[0], r[1] - r[0], g[1] - g[0]
    Vgrid = V2D_fcm(*np.meshgrid(R, r, g, indexing='ij'), args.g_1, args.g_2, M_1, M_2, m_e)

    #FIXME: nothing below here is correct
    
    P  = np.fft.fftshift(np.fft.fftfreq(args.NR, dR)) * 2 * np.pi
    p  = np.fft.fftshift(np.fft.fftfreq(args.Nr, dr)) * 2 * np.pi
    pg = np.fft.fftshift(np.fft.fftfreq(args.Ng, dg)) * 2 * np.pi

    mu = M_1*M_2/(M_1+M_2)
    Tr = KE(args.Nr, dr, m_e)
    Tmp = KE(args.Nr, dr, (M_1+M_2))
    TR = np.real(KE_FFT(args.NR, P, R, mu))
    Tg=Tmp
    
    return TR, Tr, Tg, Vgrid, (R,P), (r,p), (g, pg)


if __name__ == '__main__':
    args = parse_args()
    print(args)

    # set number of threads for Davidson etc.
    pyscflib.num_threads(args.t)

    TR, Tr, Tg, Vgrid, *_ = build_terms(args)

    # load a guess if there is one
    davidson_guess = get_davidson_guess(args.guess, (args.NR, args.Nr))
    conv, e_approx, evecs = solve_davidson2d(TR, Tr, Tg, Vgrid,
                                             num_state=args.k,
                                             verbosity=args.verbosity,
                                             iterations=args.iterations,
                                             max_subspace=args.subspace,
                                             guess=davidson_guess,
    )
    print("Davidson:", e_approx)
    print(conv)

    if args.evecs:
        np.savez(args.evecs, guess=evecs, V=Vgrid)
        print("Wrote eigenvectors to", args.evecs)


    if args.save is not None:
        if all(conv):
            with open(args.save, "a") as f:
                print(args.M_1, args.M_2, " ".join(map(str, e_approx)), file=f)
            print(f"Computed fixed center-of-mass eigenvalues",
                  f"for M_1={args.M_1}, M_2={args.M_2} amu",
                  f"with charges g_1={args.g_1}, g_1={args.g_1}",
                  f"and appended to {args.save}")
        else:
            print("Skipping saving unconverged results.")
    
    if args.exact_diagonalization:
        e_exact = solve_exact(TR, Tr, Vgrid, num_state=args.k)
        print("Exact:", e_exact)
        prms(e_approx, e_exact, "RMS deviation between Davidson and Exact")

    if not all(conv):
        print("WARNING: Not all eigenvalues converged")
        exit(1)
    else:
        print("All eigenvalues converged")
