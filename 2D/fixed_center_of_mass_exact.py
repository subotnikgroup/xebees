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

class Hamiltonian:
    def __init__(self, args):
        self.m_e = AMU_TO_AU * 1
        self.M_1 = AMU_TO_AU * args.M_1
        self.M_2 = AMU_TO_AU * args.M_2
        self.mu  = np.sqrt(self.M_1*self.M_2*self.m_e/(self.M_1+self.M_2+self.m_e))

        self.g_1 = args.g_1
        self.g_2 = args.g_2

        self.J   = args.J

        # Grid setup
        # FIXME: need to pick ranges based on masses, charges
        range_R = (2, 4.0)  # FIXME: why does V get weird when we take R->1 ?
        range_r = (1e-5, 5)
        range_g = (0, 2*np.pi)

        # as for imshow, (left, right, bottom, top)
        if hasattr(args, "extent") and args.extent is not None:
            range_R = args.extent[:2]
            range_r = args.extent[2:4]

        self.R = np.linspace(*range_R, args.NR) * ANGSTROM_TO_BOHR
        self.r = np.linspace(*range_r, args.Nr) * ANGSTROM_TO_BOHR
        self.g = np.linspace(*range_g, args.Ng)

        self.R_grid, self.r_grid, self.g_grid = np.meshgrid(self.R, self.r, self.g, indexing='ij')
        self.Vgrid = self.V_2Dfcm(self.R_grid, self.r_grid, self.g_grid)
        self.shape = self.Vgrid.shape

        dR = self.R[1] - self.R[0]
        dr = self.r[1] - self.r[0]
        dg = self.g[1] - self.g[0]

        self.P  = np.fft.fftshift(np.fft.fftfreq(args.NR, dR)) * 2 * np.pi
        self.p  = np.fft.fftshift(np.fft.fftfreq(args.Nr, dr)) * 2 * np.pi
        self.pg = np.fft.fftshift(np.fft.fftfreq(args.Ng, dg)) * 2 * np.pi


        # N.B.: These all lack the factor of -1/(2 * mu)
        # N.B.: The default stencil degree is 11
        self.ddR2 = KE(args.NR, dR, bare=True)
        self.ddr2 = KE(args.Nr, dr, bare=True)

        # Part of the reason for using a cyclic *stencil* for gamma rather
        # than KE_FFT is that it wasn't immediately obvious how I would
        # represent ∂/∂γ. (∂²/∂γ² was clear.)
        self.ddg2 = KE(args.Ng, dg, bare=True, cyclic=True)
        self.ddg1 = KE(args.Ng, dg, bare=True, cyclic=True, order=1)

        # since we need these in Hx; maybe fine to compute on the fly?
        self.Rinv2 = 1.0/(self.R_grid)**2
        self.rinv2 = 1.0/(self.r_grid)**2


    def V_2Dfcm(self, R_amu, r_amu, gamma):
        R = R_amu / ANGSTROM_TO_BOHR
        r = r_amu / ANGSTROM_TO_BOHR

        D, d, a, c = 60, 0.95, 2.52, 1
        A, B, C = 2.32e5, 3.15, 2.31e4

        mu12 = self.M_1*self.M_2/(self.M_1+self.M_2)
        aa = np.sqrt(self.mu/mu12) # factor of 'a' for lab and scaled coordinates

        kappa2 = r*R*np.cos(gamma)
        r1e = np.sqrt((aa*r)**2 + (R/aa)**2*(mu12/self.M_1)**2 - 2*kappa2*mu12/self.M_1)
        re2 = np.sqrt((aa*r)**2 + (R/aa)**2*(mu12/self.M_2)**2 + 2*kappa2*mu12/self.M_2)

        D2 = self.g_2 * D * (    np.exp(-2*a * (re2-d))
                             - 2*np.exp(  -a * (re2-d))
                             + 1)
        D1 = self.g_1 * D * c**2 * (    np.exp(-(2*a/c) * (r1e-d))
                                     - 2*np.exp(-(  a/c) * (r1e-d)))

        return KCALMOLE_TO_HARTREE * (D1 + D2 + A*np.exp(-B*R/aa) - C/(R/aa)**6)

    # allows H @ x
    def __matmul__(self, other):
        return self.Hx(other.ravel())

    # FIXME: likely will want to @jax.jit this, c.f.:
    # https://docs.jax.dev/en/latest/faq.html#how-to-use-jit-with-methods
    def Hx(self, x):
        xa = x.reshape(self.shape)
        ke = np.zeros(self.shape)

        # Radial Kinetic Energy terms, easy
        ke += np.einsum('Rrg,RS->Srg', xa, self.ddR2)  # ∂²/∂R²
        ke += np.einsum('Rrg,rs->Rsg', xa, self.ddr2)  # ∂²/∂r²

        #  ∂²/∂γ² + 1/4 terms
        keg  = np.einsum('Rrg,gh->Rrh', xa, self.ddg2)  # ∂²/∂γ²
        keg += xa / 4.0                                 # ∂²/∂γ² + 1/4
        ke += (self.Rinv2 + self.rinv2)*keg             # (1/R^2 + 1/r^2) (∂²/∂γ² + 1/4)

        # Angular Kinetic Energy J terms
        # FIXME: deal with complex wf; will break at present for J != 0
        if self.J != 0:
            keg = 2*J*(1j)*np.einsum('Rrg,gh->Rrh', xa, self.ddg1)  # 2iJ ∂/∂γ
            keg += xa*J**2                                          # 2iJ ∂/∂γ + J^2
            ke -= self.Rinv2*keg                                    # (1/R^2)*(2iJ ∂/∂γ + J^2)

        # mass portion of KE
        ke *= -1 / (2*self.mu)

        # Potential terms
        res = ke + xa * self.Vgrid
        return res.ravel()


    # FIXME: need conditioners to be available before being built
    def build_preconditioner2D(self, min_guess=4):
        # build the whole H and then extract its diagonal
        Hd = np.diag(np.array([self.Hx(e) for e in np.eye(np.prod(self.shape))]))
        def precond_stupid(dx, e, x0):
            return dx/(Hd-e)

        return precond_stupid, np.random.random(np.prod(self.shape))
        
        NR, Nr, Ng = self.shape
    
        guess = np.zeros(self.shape)
    
        U_n    = np.zeros((NR,Nr,Nr))
        U_v    = np.zeros((Nr,NR,NR))
        Ad_n   = np.zeros((NR,Nr))
        Ad_vn  = np.zeros((NR,Nr))
    
        # diagonalize H electronic: r->n
        for i in range(NR):
            Hel = Tr + np.diag(self.Vgrid[i])
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

        return precond_vn, guesses


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


if __name__ == '__main__':
    args = parse_args()
    print(args)

    # set number of threads for Davidson etc.
    pyscflib.num_threads(args.t)

    H = Hamiltonian(args)

    if (guess:=get_davidson_guess(args.guess, H.shape)) is None:
        precond, guess = H.build_preconditioner2D(args.k)
    else:
        precond, _ = H.build_preconditioner2D(0)

    conv, e_approx, evecs = pyscflib.davidson1(
        lambda xs: [ H @ x for x in xs ],
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
        np.savez(args.evecs, guess=evecs, V=H.Vgrid)
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
        e_exact = solve_exact_gen(H.Hx, np.prod(H.shape), num_state=args.k)
        print("Exact:", e_exact)
        prms(e_approx, e_exact, "RMS deviation between Davidson and Exact")

    if not all(conv):
        print("WARNING: Not all eigenvalues converged")
        exit(1)
    else:
        print("All eigenvalues converged")
