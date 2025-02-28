#!/usr/bin/env python
import numpy as np
from sys import stderr
from pyscf import lib
import argparse as ap
from pathlib import Path

from constants import *
from hamiltonian import  KE, KE_FFT
from davidson import build_preconditioner, get_davidson_guess, get_davidson_mem
from debug import prms, timer

def VO(R, r, g_1,g_2):
    R, r = R / ANGSTROM_TO_BOHR, r / ANGSTROM_TO_BOHR
    D, d, a, c = 60, 0.95, 2.52, 1
    A, B, C = 2.32e5, 3.15, 2.31e4  

    D1 = g_1 * D * (np.exp(-2 * a * (R/2 + r - d)) - 2 * np.exp(-a * (R/2 + r - d)) + 1)
    D2 = g_2 * D * c**2 * (np.exp(- (2 * a/c) * (R/2 - r - d)) - 2 * np.exp(-a/c * (R/2 - r - d)))
    
    return 0.00159362 * (D1 + D2 + A * np.exp(-B * R) - C / R**6)


@timer
def solve_exact(NR, Nr, R, r, M, m, g_1, g_2, num_state=10):
    dR, dr = R[1] - R[0], r[1] - r[0]
    Vgrid = VO(*np.meshgrid(R, r, indexing='ij'), g_1, g_2)

    P = np.fft.fftshift(np.fft.fftfreq(NR, dR)) * 2 * np.pi
    p = np.fft.fftshift(np.fft.fftfreq(Nr, dr)) * 2 * np.pi

    mu = M_1*M_2/(M_1+M_2)
    Tr = KE(Nr, dr, m)
    Tmp = KE(Nr, dr, (M_1+M_2)* 2)
    TR = np.real(KE_FFT(NR, P, R, mu))

    H = (np.kron(TR, np.eye(Nr)) +
         np.kron(np.eye(NR), Tmp) +
         np.kron(np.eye(NR), Tr) +
         np.diag(Vgrid.ravel())
    )

    eigenvalues, eigenvectors = np.linalg.eigh(H)
    return eigenvalues[:num_state]


@timer
def solve_davidson(NR, Nr, R, r, M_1, M_2, m, num_state=10, g_1=1, g_2=1,
                   verbosity=2,
                   iterations=1000,
                   max_subspace=1000,
                   guess=None,):

    dR, dr = R[1] - R[0], r[1] - r[0]
    Vgrid = VO(*np.meshgrid(R, r, indexing='ij'), g_1, g_2)
    
    P = np.fft.fftshift(np.fft.fftfreq(NR, dR)) * 2 * np.pi
    p = np.fft.fftshift(np.fft.fftfreq(Nr, dr)) * 2 * np.pi

    mu = M_1*M_2/(M_1+M_2)
    Tr = KE(Nr, dr, m)
    Tmp = KE(Nr, dr, (M_1+M_2))
    TR = np.real(KE_FFT(NR, P, R, mu))

    T_r_mp = Tr + Tmp
    def aop_fast(x):
        xa = x.reshape(NR,Nr)
        r  = TR @ xa
        r += xa @ (T_r_mp)
        r += xa * Vgrid
        return r.ravel()

    aop = lambda xs: [ aop_fast(x) for x in xs ]

    if guess is None:
        pc_unitary, guess = build_preconditioner(T_r_mp, TR, Vgrid)
    else:
        pc_unitary, _ = build_preconditioner(T_r_mp, TR, Vgrid)


    conv, eigenvalues, eigenvectors = lib.davidson1(
        aop,
        guess,
        pc_unitary,
        nroots=num_state,
        max_cycle=iterations,
        verbose=verbosity,
        follow_state=False,
        max_space=max_subspace,
        max_memory=get_davidson_mem(0.75),
        tol=1e-12,
    )

    return conv, eigenvalues, eigenvectors


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


if __name__ == '__main__':
    args = parse_args()
    print(args)

    M_1 = AMU_TO_AU * args.M_1
    M_2 = AMU_TO_AU * args.M_2
    m = AMU_TO_AU * 1

    # Grid setup
    R = np.linspace(2, 4, args.NR) * ANGSTROM_TO_BOHR
    r = np.linspace(-2, 2, args.Nr) * ANGSTROM_TO_BOHR

    # load a guess if there is one
    davidson_guess = get_davidson_guess(args.guess, (args.NR, args.Nr))

    # set number of threads for Davidson
    lib.num_threads(args.t)

    conv, e_approx, evecs = solve_davidson(args.NR, args.Nr, R, r, M_1, M_2, m, num_state=args.k, g_1=args.g_1, g_2=args.g_2,
                                          verbosity=args.verbosity,
                                          iterations=args.iterations,
                                          max_subspace=args.subspace,
                                          guess=davidson_guess,
    )
    print("Davidson:", e_approx)
    print(conv)
    
    if not all(conv):
        print("WARNING: Not all eigenvalues converged; results will not be saved!")
    else:
        print("All eigenvalues converged")
        if args.evecs:
            np.savez(args.evecs, guess=evecs)
            print("Wrote eigenvectors to", args.evecs)


    if args.save is not None and all(conv):
        with open(args.save, "a") as f:
            print(M_1, M_2, " ".join(map(str, e_approx)), file=f)
        print(f"Computed eigenvalues for M={M} amu and appended to {args.save}")
    
    if args.exact_diagonalization:
        e_exact = solve_exact(args.NR, args.Nr, R, r, M_1, M_2, m, num_state=args.k, g_1=args.g_1, g_2=args.g_2)
        print("Exact:", e_exact)
        prms(e_approx, e_exact, "RMS deviation between Davidson and Exact")
