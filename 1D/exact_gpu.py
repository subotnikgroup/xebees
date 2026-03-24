#!/usr/bin/env python
import numpy
from sys import stderr
import argparse as ap
from pathlib import Path
from threadpoolctl import ThreadpoolController

import os, sys
sys.path.append(os.path.abspath("lib"))

import xp
from constants import *
from hamiltonian import  KE, KE_FFT
from davidson import solve_davidson, solve_exact, get_davidson_guess
from debug import prms, timer
import potentials


def VO(R_amu, r_amu, g_1, g_2, M_1, M_2):
    R, r = R_amu / ANGSTROM_TO_BOHR, r_amu / ANGSTROM_TO_BOHR
    D, d, a, c = 60, 0.95, 2.52, 1
    A, B, C = 2.32e5, 3.15, 2.31e4
    mu = M_1*M_2/(M_1+M_2)

    D1 = g_2 * D * (    xp.exp(-2*a * (r + mu/M_2*R - d))
                    - 2*xp.exp(  -a * (r + mu/M_2*R - d))
                    + 1)
    D2 = g_1 * D * c**2 * (    xp.exp(-(2*a / c) * (mu/M_1*R - r - d))
                           - 2*xp.exp(-(  a / c) * (mu/M_1*R - r - d)))
    VN = g_1 * g_2 * (A * xp.exp(-B * R) - C / R**6)

    return KCALMOLE_TO_HARTREE * (D1 + D2 + VN)




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
    parser.add_argument('--backend', default='numpy')
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
    mu = M_1*M_2/(M_1+M_2)

    extent = potentials.extents_borgis(mu)
    print("extent",extent)
    R_min = extent[0]
    R_max = extent[1]
    r_min = -extent[2]
    r_max = extent[2]


    R = xp.linspace(R_min, R_max, args.NR)
    r = xp.linspace(r_min, r_max, args.Nr)

    dR, dr = R[1] - R[0], r[1] - r[0]
    Vgrid = VO(*xp.meshgrid(R, r, indexing='ij'), args.g_1, args.g_2, M_1, M_2)

    P = xp.fft.fftshift(xp.fft.fftfreq(args.NR, dR)) * 2 * xp.pi
    p = xp.fft.fftshift(xp.fft.fftfreq(args.Nr, dr)) * 2 * xp.pi

    
    Tr = KE(args.Nr, dr, m)
    Tmp = KE(args.Nr, dr, (M_1+M_2))
    TR = -xp.real(KE_FFT(args.NR, P, R)) / (2 * mu)

    return dr, TR, Tr, Tmp, Vgrid, (R,P), (r,p)


if __name__ == '__main__':
    args = parse_args()
    print(args)

    # you can only select the backend once and it must be before you use any xp functions
    if xp.backend != args.backend:
        xp.backend = args.backend

    # set number of threads for Davidson etc.
    threadctl = ThreadpoolController()
    threadctl.limit(limits=args.t)

    batch_eigvalsh = xp.linalg.eigvalsh
    if xp.backend == 'cupy':
        try:
            print("cupy detected; trying diagonalization with torch backend")
            import torch
            torch.cuda.current_device()
        except ModuleNotFoundError:
            print("torch not found.")
        except AssertionError:
            print("torch not available.")
        else:
            def torch_eigvalsh(H):
                return xp.asarray(torch.linalg.eigvalsh(torch.from_dlpack(H)))
            batch_eigvalsh = torch_eigvalsh 

    dr,TR, Tr, Tmp, Vgrid, *_ = build_terms(args)

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
    p2 = KE(args.Nr, dr,order=2, bare=True)
    p1 = KE(args.Nr, dr,order=1, bare=True)
    print("evecs[0].shape",evecs[0].shape)
    print("p2.shape",p2.shape)
    evecs_0 = evecs[0].reshape(args.NR, args.Nr)
    evecs_1 = evecs[1].reshape(args.NR, args.Nr)
    p2_expec  = -xp.einsum('Rr, rx, Rx -> ', xp.conj(evecs_0), p2, evecs_0)
    p2_expec1 =  xp.einsum('Rr, rx, Rx -> ', xp.conj(evecs_1), p2, evecs_0)
    p1_expec1 = -1j*xp.einsum('Rr, rx, Rx -> ', xp.conj(evecs_1), p1, evecs_0)
    #p2_expec = xp.conj(evecs[0]).T @ (p2 @ evecs[0])
    print("p2_expec_00",p2_expec)
    print("p2_expec_01", p2_expec1)
    print("p1_expec1_01", p1_expec1)
    if args.evecs:
        numpy.savez(args.evecs, guess=evecs, V=Vgrid)
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
