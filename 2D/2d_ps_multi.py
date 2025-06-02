#!/usr/bin/env python
import numpy as np
from sys import stderr
import argparse as ap
from pathlib import Path
from pyscf import lib as pyscflib
import jax
import jax.numpy as jnp
from functools import partial
import multiprocessing
import os

from constants import *
from hamiltonian import  KE, KE_FFT, KE_Borisov, Gamma_etf, Gamma_erf,inverse_weyl_transform
from davidson import get_davidson_guess, get_davidson_mem, solve_exact_gen, eye_lazy
from debug import prms, timer, timer_ctx

class Hamiltonian:
    __slots__ = ( # any new members must be added here
        'm_e', 'M_1', 'M_2', 'mu', 'g_1', 'g_2', 'J',
        'R', 'P', 'R_grid', 'r', 'p', 'r_grid', 'g', 'pg', 'g_grid',
        'Vgrid', 'ddR2', 'ddr2', 'ddg2', 'ddg1','ddr1','axes',
        'Rinv2', 'rinv2', 'diag', '_preconditioner_data',
        'shape', 'size','guess','k','mu12',
        '_locked'
    )

    # FIXME: this is useful for debugging, but ultimately we would
    # like to freeze this on creaton too.
    _mutable_keys = ('_preconditioner_data',)

    # prevent data from being modified
    def __setattr__(self, key, value):
        if getattr(self, '_locked', False) and key not in self._mutable_keys:
            raise AttributeError(f"Cannot modify '{key}'; all members are frozen on creation")
        super().__setattr__(key, value)

    def __init__(self, args):

        self.m_e = 1
        self.M_1 = args.M_1
        self.M_2 = args.M_2
        self.mu  = np.sqrt(self.M_1*self.M_2*self.m_e/(self.M_1+self.M_2+self.m_e))
        self.mu12 = self.M_1*self.M_2/(self.M_1+self.M_2)
        self.g_1 = args.g_1
        self.g_2 = args.g_2

        self.J = args.J
        self.guess = args.guess
        self.k = args.k


        extent = np.array([1/args.NR, 2, 2])        
        if hasattr(args, "extent") and args.extent is not None:
            extent = args.extent

        R_range = extent[:2]
        r_max   = extent[-1]

        print("unscaled coords:", R_range, r_max)

        if r_max < R_range[-1]/2:
            raise RuntimeError("r_max should be at least R_max/2")

        R_range *= ANGSTROM_TO_BOHR 
        r_max   *= ANGSTROM_TO_BOHR 

        print("  scaled coords:", R_range, r_max)

        self.R = np.linspace(*R_range, args.NR)
        self.r = np.linspace(r_max/args.Nr, r_max, args.Nr)

        # require Ng to be even
        if args.Ng % 2 != 0:
            raise RuntimeError(f"Ng must be even!")

        # N.B.: It is essential that we not include the endpoint in
        # gamma lest our cyclic grid be ill-formed and 2nd derivatives
        # all over the place
        self.g = np.linspace(0, 2*np.pi, args.Ng, endpoint=False)

        self.axes = (self.R, self.r, self.g)

        
        # N.B.: It is essential that we not include the endpoint in
        # gamma lest our cyclic grid be ill-formed and 2nd derivatives
        # all over the place
        self.g = np.linspace(0, 2*np.pi, args.Ng, endpoint=False)

        self.r_grid, self.g_grid = np.meshgrid(self.r, self.g, indexing='ij')
        self.shape = (args.NR, args.Nr, args.Ng)
        self.size = args.NR * args.Nr * args.Ng

        dR = self.R[1] - self.R[0]
        dr = self.r[1] - self.r[0]
        dg = self.g[1] - self.g[0]

        self.P  = np.fft.fftshift(np.fft.fftfreq(args.NR, dR)) * 2 * np.pi
        self.p  = np.fft.fftshift(np.fft.fftfreq(args.Nr, dr)) * 2 * np.pi
        self.pg = np.fft.fftshift(np.fft.fftfreq(args.Ng, dg)) * 2 * np.pi

        # N.B.: These all lack the factor of -1/(2 * mu)
        # We also are throwing away the returned jacobian of R/r
        self.ddR2, _ = KE_Borisov(self.R, bare=True)
        self.ddr2, _ = KE_Borisov(self.r, bare=True)

        # Part of the reason for using a cyclic *stencil* for gamma
        # rather than KE_FFT is that it wasn't immediately obvious how
        # I would represent ∂/∂γ. (∂²/∂γ² was clear.)  N.B.: The
        # default stencil degree is 11
        self.ddg2 = KE(args.Ng, dg, bare=True, cyclic=True)
        self.ddg1 = KE(args.Ng, dg, bare=True, cyclic=True, order=1)
        self.ddr1, _ = KE_Borisov(self.r, bare=True, order=1)

        # since we need these in Hx; maybe fine to compute on the fly?
        self.rinv2 = 1.0/(self.r)**2
        
        self._lock()


    # ensure that arrays cannot be written to
    def _lock(self):
        self._locked = True
        for key in self.__slots__:
            try:
                member = super().__getattribute__(key)
            except AttributeError:
                continue
            if (key not in self._mutable_keys and
                isinstance(member := super().__getattribute__(key), np.ndarray)):
                member.flags.writeable = False


    def soft_coulomb(self, R, r1e, r2e, charges, dv=0.5, G=1, p=1):
        Q1, Q2 = charges
    
        V1  = -Q1      / (r1e**2 + dv**2)**(p/2)
        V2  = -Q2      / (r2e**2 + dv**2)**(p/2)
        VN  =  Q1 * Q2 / (R**2   + dv**2)**(p/2)
        return G*(V1 + V2 + VN)
    
    def V_2Dfcm(self,R, r, gamma):
        mu12 = self.mu12
        M_1 = self.M_1
        M_2 = self.M_2

        kappa2 = r*R*np.cos(gamma)

        r1e2 = (r)**2 + (R)**2*(mu12/M_1)**2 - 2*kappa2*mu12/M_1
        r2e2 = (r)**2 + (R)**2*(mu12/M_2)**2 + 2*kappa2*mu12/M_2
        
        r1e = np.sqrt(np.where(r1e2 < 0, 0, r1e2))
        r2e = np.sqrt(np.where(r2e2 < 0, 0, r2e2))

        return self.soft_coulomb(R, r1e, r2e, (self.g_1, self.g_2))
    
    @timer
    def build_preconditioner(self,Ham,Ri, min_guess=4):
        # FIXME: use buildDiag. (should be super fast).
        #print("hsize",Ham.shape)
        with timer_ctx(f"build H diag {Ham.shape[0]}"):
            #self._preconditioner_data = np.array([e @ (self @ e) for e in eye_lazy(self.size)])
            self._preconditioner_data = np.diagonal(Ham)
            # likely overkill
            self._preconditioner_data.flags.writeable = False
        Vgrid = self.V_2Dfcm(Ri, self.r_grid, self.g_grid)
        return np.exp(-(Vgrid - np.min(Vgrid))*27.211*2).ravel()
        
    def preconditioner(self, dx, e, x0):
        if not hasattr(self, "_preconditioner_data"):
            print("building stupid preconditioner")
            self.build_preconditioner()
        return self._preconditioner_kernel(dx, e, x0)

    # FIXME: will want to @jax.jit this too
    def _preconditioner_kernel(self, dx, e, x0):
        Hd = self._preconditioner_data
        return dx/(Hd-e)



def solve_davidson(H, v_diag,i,k,Htot,
                   num_state=3,
                   verbosity=2,
                   iterations=1000,
                   subspace=1000,
                   guess_c=None,):
        
        def aop_fast(x):
            xa = x.reshape(v_diag.shape)
            xa = xa.astype(complex)
            #Hel=-1/(2*H.m_e)*(np.kron(H.ddr2,np.eye(H.shape[2])) + np.kron(np.diag(H.rinv2), H.ddg2))
            r  = -1/(2*H.m_e)*(H.ddr2 @ xa)
            r += np.diag(H.rinv2)@ xa @ H.ddg2
            gammaetf1R, gammaetf1t, gammaetf2R, gammaetf2t = Gamma_etf_new(H.R[i],H.r_grid,H.g_grid,H.ddr1,H.ddg1,H.M_1,H.M_2,xa)
            gamma1R = gammaetf1R
            gamma2R = gammaetf2R
            
            gammasq1R = gamma1R@gamma1R
            gammasq2R = gamma2R@gamma2R
            
            GammatotR = (H.M_2*gamma1R-H.M_1*gamma2R)/(H.M_1+H.M_2)
              
            GammasqtotR = ((H.M_2**2*gammasq1R)+(H.M_1**2*gammasq2R)-(H.M_1*H.M_2*gamma1R@gamma2R)-(H.M_1*H.M_2*gamma2R@gamma1R))/(H.M_1+H.M_2)**2
            Gammamat = -1j*GammatotR*H.P[k]/H.mu12
            r += Gammamat

            r += xa * v_diag
            
            return (r.ravel())
        
        aop = lambda xs: [ aop_fast(x) for x in xs ]
        
        with timer_ctx("Davidson"):
            conv, e_approx, evecs = pyscflib.davidson1(
                aop,
                guess_c,
                H.preconditioner,
                nroots=num_state,
                max_cycle=iterations,
                verbose=verbosity,
                follow_state=False,
                max_space=subspace,
                max_memory=get_davidson_mem(0.75),
                tol=1e-12,
            )

        print("Davidson:", e_approx[0:2])
        print(conv)
    
        return conv, e_approx

def compute_EPS(info):
    i, k, H, args = info
    print("i,k",i,k)
    gammaetf1R, gammaetf1t, gammaetf2R, gammaetf2t = Gamma_etf(H.R[i],H.r_grid,H.g_grid,H.ddr1,H.ddg1,H.M_1,H.M_2)
    gammaerf1t, gammaerf2t = Gamma_erf(H.R[i],H.r_grid,H.g_grid,H.ddr1,H.ddg1,H.M_1,H.M_2)
            
    gamma1R = gammaetf1R
    gamma2R = gammaetf2R
    gamma1t = gammaetf1t+gammaerf1t
    gamma2t = gammaetf2t+gammaerf2t

    gammasq1R = gamma1R@gamma1R
    gammasq2R = gamma2R@gamma2R
    gammasq1t = gamma1t@gamma1t
    gammasq2t = gamma2t@gamma2t

    GammatotR = (H.M_2*gamma1R-H.M_1*gamma2R)/(H.M_1+H.M_2)
    Gammatotth = (H.M_2*gamma1t-H.M_1*gamma2t)/(H.M_1+H.M_2)  

    GammasqtotR = ((H.M_2**2*gammasq1R)+(H.M_1**2*gammasq2R)-(H.M_1*H.M_2*gamma1R@gamma2R)-(H.M_1*H.M_2*gamma2R@gamma1R))/(H.M_1+H.M_2)**2
    Gammasqtotth = ((H.M_2**2*gammasq1t)+(H.M_1**2*gammasq2t)-(H.M_1*H.M_2*gamma1t@gamma2t)-(H.M_1*H.M_2*gamma2t@gamma1t))/(H.M_1+H.M_2)**2      
    
    v_diag = np.diag(H.V_2Dfcm(H.R[i], H.r_grid, H.g_grid).ravel())
    
    Gammamat = -1j*GammatotR*H.P[k]/H.mu12 -1j*Gammatotth*(H.J/H.R[i])/H.mu12-(GammasqtotR+Gammasqtotth)/(2*H.mu12) 
    Hel=-1/(2*H.m_e)*(np.kron(H.ddr2,np.eye(H.shape[2])) + np.kron(np.diag(H.rinv2), H.ddg2))             
    Htot = Hel+v_diag+Gammamat

    #guess = H.build_preconditioner(v_diag,H.R[i],5)
    v_diag = H.V_2Dfcm(H.R[i], H.r_grid, H.g_grid)

    e_approx,e_wave = np.linalg.eigh(Htot)
        
    eps_val = e_approx[0] + 0.5 * H.P[k]**2 / H.mu12 + 0.5 * (H.J/H.R[i])**2 /H.mu12

    return i,k,eps_val

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
    EPS = np.zeros((H.shape[0], H.shape[0]))
 
    index_pairs = [(i, k, H, args) for i in range(H.shape[0]) for k in range(H.shape[0])]
    with multiprocessing.Pool(processes=5) as pool:
        results = pool.map(compute_EPS, index_pairs)

    for i,k,val in results:
        EPS[i, k] = val
    
    HPS = inverse_weyl_transform(EPS, H.shape[0], H.R, H.P)
    EPSv, psiPSv = np.linalg.eigh(HPS)
    print("EPSv",EPSv)
    if args.evecs:
        np.savez(args.evecs, guess=psiPSv, R=H.R)
        print("Wrote eigenvectors to", args.evecs)

    if args.save is not None:
        #if all(conv):
        with open(args.save, "a") as f:
            print(args.M_1, args.M_2, args.g_1, args.g_2, args.J,
                  " ".join(map(str, EPSv)), file=f)
        print(f"Computed fixed center-of-mass eigenvalues",
              f"for M_1={args.M_1}, M_2={args.M_2} amu",
              f"with charges g_1={args.g_1}, g_2={args.g_2}",
              f"and total J={args.J}",
              f"and appended to {args.save}")
        #else:
        #    print("Skipping saving unconverged results.")
    
#
    #if not all(conv):
    #    print("WARNING: Not all eigenvalues converged")
    #    exit(1)
    #else:
    #    print("All eigenvalues converged")

