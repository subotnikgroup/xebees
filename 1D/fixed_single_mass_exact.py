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


def VO(R_amu, r_amu, g_1, g_2):
    R, r = R_amu / ANGSTROM_TO_BOHR , r_amu / ANGSTROM_TO_BOHR
    D, d, a, c = 60, 0.95, 2.52, 1
    A, B, C = 2.32e5, 3.15, 2.31e4

    D1 = g_2 * D * (np.exp(-2 * a * (r + R - d)) - 2 * np.exp(-a * (r + R - d)) + 1)
    D2 = g_1 * D * c**2 * (np.exp((2 * a/c) * (r + d)) - 2 * np.exp(a/c * (r + d)))

    return KCALMOLE_TO_HARTREE * (D1 + D2 + A * np.exp(-B * R) - C / R**6)


def build_terms(args):
    M = AMU_TO_AU * args.M
    m = AMU_TO_AU * 1

    # Grid setup
    R = np.linspace(2, 4, args.NR) * ANGSTROM_TO_BOHR
    r = np.linspace(-4, 1, args.Nr) * ANGSTROM_TO_BOHR

    dR, dr = R[1] - R[0], r[1] - r[0]
    Vgrid = VO(*np.meshgrid(R, r, indexing='ij'), args.g_1, args.g_2)

    P = np.fft.fftshift(np.fft.fftfreq(args.NR, dR)) * 2 * np.pi
    p = np.fft.fftshift(np.fft.fftfreq(args.Nr, dr)) * 2 * np.pi

    Tr = KE(args.Nr, dr, m)
    TR = np.real(KE_FFT(args.NR, P, R, M))

    return TR, Tr, Vgrid, (R,P), (r,p)


def parse_args():
    parser = ap.ArgumentParser(
        prog='fixed-single-mass-exact',
        description="Exact 1D solution in the frame of a single fixed mass")

    parser.add_argument('-k', metavar='num_eigenvalues', default=5, type=int)
    parser.add_argument('-t', metavar="num_threads", default=16, type=int)
    parser.add_argument('-g_1', metavar='g_1', required=True, type=float)
    parser.add_argument('-g_2', metavar='g_2', required=True, type=float)
    parser.add_argument('-M', required=True, type=float)
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


if __name__ == '__main__':
    args = parse_args()
    print(args)

    # set number of threads for Davidson etc.
    pyscflib.num_threads(args.t)

    TR, Tr, Vgrid, *_ = build_terms(args)

    # load a guess if there is one
    davidson_guess = get_davidson_guess(args.guess, (args.NR, args.Nr))
    conv, e_approx, evecs = solve_davidson(TR, Tr, Vgrid,
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
                print(args.M, " ".join(map(str, e_approx)), file=f)
            print(f"Computed fixed single-mass eigenvalues",
                  f"for M={args.M} amu",
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
