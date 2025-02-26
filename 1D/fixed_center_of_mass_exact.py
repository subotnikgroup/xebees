#!/usr/bin/env python

import numpy as np
import scipy.sparse as sp
from sys import stderr
from scipy.special import factorial
from scipy.signal import convolve
from pyscf import lib
import argparse as ap
import timeit
from os import sysconf
from pathlib import Path
# import opt_einsum

# Constants
AMU_TO_AU = 1822.888486209
ANGSTROM_TO_BOHR = 1.8897259886

from functools import wraps
from time import perf_counter as time

from contextlib import contextmanager


@contextmanager
def timer_ctx(label=""):
   start = time()
   yield
   end = time()
   elapsed = end - start
   print(f"[{label}] Elapsed time: {1e6*elapsed:0.3}us")


def timer(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        start = time()
        result = f(*args, **kwargs)
        end = time()
        elapsed = end - start
        print(f"Elapsed time: {elapsed}s")
        return result
    return wrapper

def VO(R, r, g_1,g_2):
    R, r = R / ANGSTROM_TO_BOHR, r / ANGSTROM_TO_BOHR
    D, d, a, c = 60, 0.95, 2.52, 1
    A, B, C = 2.32e5, 3.15, 2.31e4  

    D1 = g_1 * D * (np.exp(-2 * a * (R/2 + r - d)) - 2 * np.exp(-a * (R/2 + r - d)) + 1)
    D2 = g_2 * D * c**2 * (np.exp(- (2 * a/c) * (R/2 - r - d)) - 2 * np.exp(-a/c * (R/2 - r - d)))
    
    return 0.00159362 * (D1 + D2 + A * np.exp(-B * R) - C / R**6)


def get_stencil_coefficients(stencil_size, derivative_order):
    if stencil_size % 2 == 0:
        raise ValueError("Stencil size must be odd.")
    
    half_size = stencil_size // 2
    A = np.vander(np.arange(-half_size, half_size + 1), increasing=True).T
    b = np.zeros(stencil_size)
    b[derivative_order] = factorial(derivative_order)
    
    return np.linalg.solve(A, b)

def KE(N, dx, mass, sparse=False, stencil_size=11):
    stencil = get_stencil_coefficients(stencil_size, 2) / dx**2
    I = np.eye(N)
    T = -1 / (2 * mass) * np.array([convolve(I[i, :], stencil, mode='same') for i in range(N)])
    
    return sp.csr_matrix(T) if sparse else T

def KE_FFT(N, P, R, mass): 
    Tp = np.diag(P**2 / (2 * mass))
    exp_RP = np.exp(1j * np.outer(P, R))
    
    return (exp_RP.T.conj() @ Tp @ exp_RP) / N

def solve_BO_surface(NR, Nr, R, r, m):
    dr = r[1] - r[0]
    return np.array([np.linalg.eigvalsh(KE(Nr, dr, m) + np.diag(VO(R[i], r)))[0] for i in range(NR)])

def solve_BOv(NR, Nr, R, r, M, m):
    dR = R[1] - R[0]
    return np.linalg.eigvalsh(KE(NR, dR, M) + np.diag(solve_BO_surface(NR, Nr, R, r, m)))

def prms(A,B, label=""):
    print(label, np.sqrt(np.mean((A-B)**2)))

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
def build_preconditioner(Tr, TR, Vgrid):
    NR, Nr = Vgrid.shape

    guess = np.zeros((NR,Nr))

    U_n    = np.zeros((NR,Nr,Nr))
    U_v    = np.zeros((Nr,NR,NR))
    Ad_n   = np.zeros((NR,Nr))
    Ad_vn  = np.zeros((NR,Nr))

    # diagonalize H electronic: r->n
    for i in range(NR):
        Hel = Tr + np.diag(Vgrid[i])
        Ad_n[i], U_n[i] = np.linalg.eigh(Hel)

        # align phases
        if i > 0:
            if np.sum(U_n[i] * U_n[i-1]) < 0:
                U_n[i] *= -1.0

        guess[i] = U_n[i, 0].T

    # diagonalize Born-Oppenheimer Hamiltonian: R->v
    for i in range(Nr):
        Hbo = TR + np.diag(Ad_n[:,i])
        Ad_vn[:,i], U_v[i] = np.linalg.eigh(Hbo)

        # align phases
        if i > 0:
            if np.sum(U_v[i] * U_v[i-1]) < 0:
                U_v[i] *= -1.0

    # stamp down the vib-ground state
    for i in range(Nr):
        guess[:,i] =  U_v[0].T @ guess[:,i]


    def precond_Rn(dx, e, x0):
        dx_Rr = dx.reshape((NR,Nr))
        
        #for i in range(NR):
        #    dx_Rn[i] = U_n[i].T @ dx_Rr[i]

        dx_Rn = np.einsum('Rji,Rj->Ri', U_n, dx_Rr)
        tr_Rn = dx_Rn / (Ad_n - e)
        tr_Rr = np.einsum('Rij,Rj->Ri', U_n, tr_Rn)
        
        #for i in range(NR):
        #    tr_Rr[i] = U_n[i] @ tr_Rn[i]
        
        return tr_Rr.ravel()


    # for our simple case, these contractions were no observable help and harder to read
    # to_vn = opt_einsum.contract_expression('nji,jqn,jq->in', U_v, U_n, (NR,Nr), constants=[0,1], optimize='optimal')
    # to_Rr = opt_einsum.contract_expression('Rij,jRq,qj->Ri', U_n, U_v, (NR,Nr), constants=[0,1], optimize='optimal')

    # Elimination of temporaries by merging the contractions powered by opt_einsum_fx.
    # c.f.: https://opt-einsum-fx.readthedocs.io/en/latest/api.html#opt_einsum_fx.fuse_einsums
    def precond_vn(dx, e, x0):
        dx_Rr = dx.reshape((NR,Nr))

        #dx_Rn = np.einsum('Rji,Rj->Ri', U_n, dx_Rr, optimize=True)
        #dx_vn = np.einsum('nji,jn->in', U_v, dx_Rn, optimize=True)

        dx_vn = np.einsum('nji,jqn,jq->in', U_v, U_n, dx_Rr, optimize=True)

        tr_vn = dx_vn / (Ad_vn - e)

        #tr_Rn = np.einsum('nij,jn->in', U_v, tr_vn, optimize=True)
        #tr_Rr = np.einsum('Rij,Rj->Ri', U_n, tr_Rn, optimize=True)

        tr_Rr = np.einsum('Rij,jRq,qj->Ri', U_n, U_v, tr_vn, optimize=True)

        return tr_Rr.ravel()

    return precond_vn, guess.ravel()


def get_davidson_mem(fraction):
    if fraction > 1 or fraction < 0:
        raise RuntimeError("Fraction of system memory for Davidson must be on [0, 1]")

    try:
        system_memory_mb = (sysconf('SC_PAGE_SIZE') * sysconf('SC_PHYS_PAGES')) / 1024**2
    except ValueError:
        print("Unable to determine system memory!")
        system_memory_mb = 8000
    finally:
        davidson_mem = fraction * system_memory_mb
        print(f"Davidson will consume up to {int(davidson_mem)}MB of memory.")

    return davidson_mem



@timer
def solve_davidson(NR, Nr, R, r, M_1, M_2, m, num_state=10, g_1=1, g_2=1, verbosity=2,
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
        #max_space=200, # FIXME: think about tuning this parameter
        max_memory=get_davidson_mem(0.75),
        tol=1e-12,
    )

    return conv, eigenvalues, eigenvectors


def get_davidson_guess(guessfile, grid_dims):
    if guessfile is None:
        return

    if not guessfile.exists():
        print(f"WARNING: requested guess-file, {guessfile}, does not exist!")
        return

    guess = np.load(guessfile)['guess']
    if guess.shape[1] == np.prod(grid_dims):
        print("Loaded guess from", guessfile)
        return guess
    else:
        print("WARNING: Loaded guess of improper dimension; discarding!")
        return


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
