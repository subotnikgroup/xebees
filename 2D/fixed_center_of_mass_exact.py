#!/usr/bin/env python
import numpy as np
from sys import stderr
import argparse as ap
from pathlib import Path
from pyscf import lib as pyscflib

from constants import *
from hamiltonian import  KE, KE_FFT
from davidson import get_davidson_guess, get_davidson_mem, solve_exact_gen
from debug import prms, timer


def V2D_fcm(R_amu, r_amu, gamma, g_1, g_2, M_1, M_2, m_e):
    R = R_amu / ANGSTROM_TO_BOHR
    r = r_amu / ANGSTROM_TO_BOHR
    
    D, d, a, c = 60, 0.95, 2.52, 1
    A, B, C = 2.32e5, 3.15, 2.31e4
    
    mu12 = M_1*M_2/(M_1+M_2)
    mu = np.sqrt(M_1*M_2*m_e/(M_1+M_2+m_e))
    aa = np.sqrt(mu/mu12) # factor of 'a' for lab and scaled coordinates
    
    kappa2 = r*R*np.cos(gamma)
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
    parser.add_argument('-g', dest="Ng", metavar="Ng", default=250, type=int)
    parser.add_argument('--exact_diagonalization', action='store_true')
    parser.add_argument('--verbosity', default=2, type=int)
    parser.add_argument('--iterations', metavar='max_iterations', default=10000, type=int)
    parser.add_argument('--subspace', metavar='max_subspace', default=1000, type=int)
    parser.add_argument('--guess', metavar="guess.npz", type=Path, default=None)
    parser.add_argument('--evecs', metavar="guess.npz", type=Path, default=None)
    parser.add_argument('--save', metavar="filename")

    return parser.parse_args()


# FIXME: wrap this whole object up as a class, provide @ operartor,
# deal with jax, etc. Think about where to add V.
def build_terms(args):
    m_e = AMU_TO_AU * 1
    M_1 = AMU_TO_AU * args.M_1
    M_2 = AMU_TO_AU * args.M_2

    # Grid setup
    # FIXME: need to pick ranges based on masses, charges
    range_R = (2, 4.0)  # FIXME: why does V get weird when we take R->1 ?
    range_r = (1e-5, 5)
    range_g = (0, 2*np.pi)
    
    # as for imshow, (left, right, bottom, top)
    if hasattr(args, "extent") and args.extent is not None:
        range_R = args.extent[:2]
        range_r = args.extent[2:4]

    R = np.linspace(*range_R, args.NR) * ANGSTROM_TO_BOHR
    r = np.linspace(*range_r, args.Nr) * ANGSTROM_TO_BOHR
    g = np.linspace(*range_g, args.Ng)

    dR, dr, dg = R[1] - R[0], r[1] - r[0], g[1] - g[0]
    Vgrid = V2D_fcm(*np.meshgrid(R, r, g, indexing='ij'),
                    args.g_1, args.g_2, M_1, M_2, m_e)

    P  = np.fft.fftshift(np.fft.fftfreq(args.NR, dR)) * 2 * np.pi
    p  = np.fft.fftshift(np.fft.fftfreq(args.Nr, dr)) * 2 * np.pi
    pg = np.fft.fftshift(np.fft.fftfreq(args.Ng, dg)) * 2 * np.pi

    J = args.J
    mu = np.sqrt(M_1*M_2*m_e/(M_1+M_2+m_e))

    # N.B.: These all lack the factor of -1/(2 * mu)
    # N.B.: The default stencil degree is 11
    ddR2 = KE(args.NR, dR, bare=True)
    ddr2 = KE(args.Nr, dr, bare=True)

    # Part of the reason for using a cyclic *stencil* for gamma rather
    # than KE_FFT is that it wasn't immediately obvious how I would
    # represent ∂/∂γ. (∂²/∂γ² was clear.)
    ddg2 = KE(args.Ng, dg, bare=True, cyclic=True)
    ddg1 = KE(args.Ng, dg, bare=True, cyclic=True, order=1)

    # since we need these below
    R_grid, r_grid, g_grid = np.meshgrid(R, r, g, indexing='ij')
    Rinv2 = 1.0/(R_grid)**2
    rinv2 = 1.0/(r_grid)**2

    # FIXME: likely will want to @jax.jit this; likely will need to
    # make it a class method; c.f.:
    # https://docs.jax.dev/en/latest/faq.html#how-to-use-jit-with-methods
    def Hx(x):
        xa = x.reshape(Vgrid.shape)
        ke = np.zeros(Vgrid.shape)

        # Radial Kinetic Energy terms, easy
        ke += np.einsum('Rrg,RS->Srg', xa, ddR2)  # ∂²/∂R²
        ke += np.einsum('Rrg,rs->Rsg', xa, ddr2)  # ∂²/∂r²

        #  ∂²/∂γ² + 1/4 terms
        keg  = np.einsum('Rrg,gh->Rrh', xa, ddg2)  # ∂²/∂γ²
        keg += xa / 4.0                            # ∂²/∂γ² + 1/4
        ke += (Rinv2 + rinv2)*keg                  # (1/R^2 + 1/r^2) (∂²/∂γ² + 1/4)

        # Angular Kinetic Energy J terms
        # FIXME: deal with complex wf; will break at present for J != 0
        if J != 0:
            keg = 2*J*(1j)*np.einsum('Rrg,gh->Rrh', xa, ddg1)  # 2iJ ∂/∂γ
            keg += xa*J**2                                     # 2iJ ∂/∂γ + J^2
            ke -= Rinv2*keg                                    # (1/R^2)*(2iJ ∂/∂γ + J^2)

        # mass portion of KE
        ke *= -1 / (2*mu)

        # Potential terms
        res = ke + xa * Vgrid
        return res.ravel()


    return ddR2, ddr2, (ddg2,ddg1), Vgrid, Hx, (R,P), (r,p), (g,pg)


def build_preconditioner2D(TR, Tr, Tg, Vgrid, Hx, min_guess=4):
    NR, Nr, Ng = Vgrid.shape

    guess = np.zeros((NR,Nr,Ng))

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
            for j in range(Nr):
                if np.sum(U_n[i,:,j] * U_n[i-1,:,j]) < 0:
                    U_n[i,:,j] *= -1.0

    # diagonalize Born-Oppenheimer Hamiltonian: R->v
    for i in range(Nr):
        Hbo = TR + np.diag(Ad_n[:,i])
        Ad_vn[:,i], U_v[i] = np.linalg.eigh(Hbo)

        # align phases
        if i > 0:
            for j in range(NR):
                if np.sum(U_v[i,:,j] * U_v[i-1,:,j]) < 0:
                    U_v[i,:,j] *= -1.0

    # BO states are like: U_n[:,:,n]
    # vib states are like: U_v[n,:,v]
    # our first guess was the ground state BO wavefuction dressed by the first vibrational state
    # guess = U_n[:,:,0] * U_v[0,:,0,np.newaxis]
    # Now we take something like the first num_guess states
    s = int(np.ceil(np.sqrt(min_guess)))
    guesses = [(U_n[:,:,n] * U_v[n,:,v,np.newaxis]).ravel() for n in range(s) for v in range(s)]


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


    # build the whole H and then extract its diagonal
    Hd = np.diag(np.array([Hx(e) for e in np.eye(np.prod(Vgrid.shape))]))
    def precond_stupid(dx, e, x0):
        return dx/(Hd-e)

    
    #return precond_vn, guesses
    return precond_stupid, np.random.random(np.prod(Vgrid.shape))


if __name__ == '__main__':
    args = parse_args()
    print(args)

    # set number of threads for Davidson etc.
    pyscflib.num_threads(args.t)

    TR, Tr, Tg, Vgrid, Hx, *_ = build_terms(args)

    if (guess:=get_davidson_guess(args.guess, (args.NR, args.Nr, args.Ng))) is None:
        precond, guess = build_preconditioner2D(TR, Tr, Tg, Vgrid, Hx, args.k)
    else:
        precond, _ = build_preconditioner2D(TR, Tr, Tg, Vgrid, Hx, 0)

    conv, e_approx, evecs = pyscflib.davidson1(
        lambda xs: [ Hx(x) for x in xs ],
        guess,
        precond,
        nroots=args.k,
        max_cycle=args.iterations,
        verbose=args.verbosity,
        follow_state=False,
        max_space=args.subspace,
        max_memory=get_davidson_mem(0.75),
        tol=1e-12,
    )

    print("Davidson:", e_approx)
    print(conv)

    if args.evecs:
        np.savez(args.evecs, guess=evecs, V=Vgrid)
        print("Wrote eigenvectors to", args.evecs)

    if args.save is not None:
        if all(conv):
            with open(args.save, "a") as f:
                print(args.M_1, args.M_2, args.g_1, args.g_2, args.J,
                      " ".join(map(str, e_approx)), file=f)
            print(f"Computed fixed center-of-mass eigenvalues",
                  f"for M_1={args.M_1}, M_2={args.M_2} amu",
                  f"with charges g_1={args.g_1}, g_2={args.g_2}",
                  f"and total J={args.J}",
                  f"and appended to {args.save}")
        else:
            print("Skipping saving unconverged results.")
    
    if args.exact_diagonalization:
        e_exact = solve_exact_gen(Hx, np.prod(Vgrid.shape), num_state=args.k)
        print("Exact:", e_exact)
        prms(e_approx, e_exact, "RMS deviation between Davidson and Exact")

    if not all(conv):
        print("WARNING: Not all eigenvalues converged")
        exit(1)
    else:
        print("All eigenvalues converged")
