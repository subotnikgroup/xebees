#!/usr/bin/env python
import numpy as np
from sys import stderr
import argparse as ap
from pathlib import Path
from pyscf import lib as pyscflib

import os, sys
sys.path.append(os.path.abspath("lib"))

from constants import *
from hamiltonian import  KE, KE_FFT
from davidson import solve_davidson, solve_exact, get_davidson_guess
from debug import prms, timer

def VO(R_amu, r_amu, g_1, g_2, M_1, M_2):
    R, r = R_amu / ANGSTROM_TO_BOHR, r_amu / ANGSTROM_TO_BOHR
    D, d, a, c = 60, 0.95, 2.52, 1
    A, B, C = 2.32e5, 3.15, 2.31e4
    mu = M_1*M_2/(M_1+M_2)

    D1 = g_2 * D * (    np.exp(-2*a * (r + mu/M_2*R - d))
                    - 2*np.exp(  -a * (r + mu/M_2*R - d))
                    + 1)
    D2 = g_1 * D * c**2 * (    np.exp(-(2*a / c) * (mu/M_1*R - r - d))
                           - 2*np.exp(-(  a / c) * (mu/M_1*R - r - d)))

    return KCALMOLE_TO_HARTREE * (D1 + D2 + A * np.exp(-B * R) - C / R**6)


def parse_args():
    parser = ap.ArgumentParser(
        prog='davidson-ps-1d',
        description="computes the lowest k eigenvalues of phase space model in Xuezhi's paper")

    parser.add_argument('-k', metavar='num_eigenvalues', default=5, type=int)
    parser.add_argument('-t', metavar="num_threads", default=16, type=int)
    parser.add_argument('-g_1', metavar='g_1', required=True, type=float)
    parser.add_argument('-g_2', metavar='g_2', required=True, type=float)
    parser.add_argument('-M_1', required=True, type=float)
    parser.add_argument('-M_2', required=True, type=float)
    parser.add_argument('-R', dest="NR", metavar="NR", default=101, type=int)
    parser.add_argument('-r', dest="Nr", metavar="Nr", default=400, type=int)
    parser.add_argument('--exact_diagonalization', action='store_true')
    parser.add_argument('--verbosity', default=2, type=int)
    parser.add_argument('--iterations', metavar='max_iterations', default=10000, type=int)
    parser.add_argument('--subspace', metavar='max_subspace', default=1000, type=int)
    parser.add_argument('--guess', metavar="guess.npz", type=Path, default=None)
    parser.add_argument('--evecs', metavar="guess.npz", type=Path, default=None)
    parser.add_argument('--save', metavar="filename")

    return parser.parse_args()


def build_terms(args):
    m = AMU_TO_AU * 1
    M_1 = AMU_TO_AU * args.M_1
    M_2 = AMU_TO_AU * args.M_2

    # Grid setup
    range_R = (2, 4)
    range_r = (-4, 4)

    # as for imshow, (left, right, bottom, top)
    if hasattr(args, "extent") and args.extent is not None:
        range_R = args.extent[:2]
        range_r = args.extent[-2:]

    R = np.linspace(*range_R, args.NR) * ANGSTROM_TO_BOHR
    r = np.linspace(*range_r, args.Nr) * ANGSTROM_TO_BOHR

    dR, dr = R[1] - R[0], r[1] - r[0]
    Vgrid = VO(*np.meshgrid(R, r, indexing='ij'), args.g_1, args.g_2, M_1, M_2)

    P = np.fft.fftshift(np.fft.fftfreq(args.NR, dR)) * 2 * np.pi
    p = np.fft.fftshift(np.fft.fftfreq(args.Nr, dr)) * 2 * np.pi

    mu = M_1*M_2/(M_1+M_2)
    Tr = KE(args.Nr, dr, m)
    Tmp = KE(args.Nr, dr, (M_1+M_2))
    TR = np.real(KE_FFT(args.NR, P, R, mu))

    return TR, Tr, Tmp, Vgrid, (R,P), (r,p)


if __name__ == '__main__':
    args = parse_args()
    print(args)

    # set number of threads for Davidson etc.
    pyscflib.num_threads(args.t)

    TR, Tr, Tmp, Vgrid, *_ = build_terms(args)

    # load a guess if there is one
    davidson_guess = get_davidson_guess(args.guess, (args.NR, args.Nr))
    conv, e_approx, evecs = solve_davidson(TR, Tr + Tmp, Vgrid,
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
        e_exact = solve_exact(TR, Tr + Tmp, Vgrid, num_state=args.k)
        print("Exact:", e_exact)
        prms(e_approx, e_exact, "RMS deviation between Davidson and Exact")

    if not all(conv):
        print("WARNING: Not all eigenvalues converged")
        exit(1)
    else:
        print("All eigenvalues converged")
