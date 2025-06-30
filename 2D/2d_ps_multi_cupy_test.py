#!/usr/bin/env python
import numpy as np

from sys import stderr
import argparse as ap
from pathlib import Path
#import jax
#import jax.numpy as jnp
from functools import partial
import multiprocessing
import os, sys 
sys.path.append(os.path.abspath("lib"))
import xp
import nvtx

from constants import *
from hamiltonian import  KE, KE_FFT, KE_Borisov, Gamma_etf_erf,inverse_weyl_transform,KE_FFT_cutoff,gamma_grad
from davidson import get_davidson_guess, get_davidson_mem, solve_exact_gen, eye_lazy
from debug import prms, timer, timer_ctx
#from threadpoolctl import ThreadpoolController
import concurrent.futures as cf
import potentials
from time import perf_counter
#import cupyx.scipy.sparse.linalg as cusparse_linalg



if __name__ == '__main__':
    from tqdm import tqdm
else:  # mock this out for use in Jupyter Notebooks etc
    def tqdm(iterator, **kwargs):
        print(f"Mock call to tqdm({kwargs})")
        return iterator

class Hamiltonian:
    __slots__ = ( # any new members must be added here
        'm_e', 'M_1', 'M_2', 'mu', 'g_1', 'g_2', 'J','mur',
        'R', 'P', 'R_grid', 'r', 'p', 'r_grid', 'g', 'pg', 'g_grid','RP_grid',
        'ddR2', 'ddr2', 'ddg2', 'ddg1','ddr1','axes','Vgrid','rp_grid','gp_grid',
        'Rinv2', 'rinv2', 'diag', '_preconditioner_data','Pg',
        'shape', 'size','guess','k','mu12','_Vfunc',
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
        self.mu  = xp.sqrt(self.M_1*self.M_2*self.m_e/(self.M_1+self.M_2+self.m_e))
        self.mu12 = self.M_1*self.M_2/(self.M_1+self.M_2)
        self.g_1 = args.g_1
        self.g_2 = args.g_2
        self.J = args.J

        if not hasattr(args, "potential"):
            args.extent = 'soft_coulomb'

        if args.potential == 'borgis' or args.potential == 'original':
            print(f"Waring: All masses scaled to AMU for {args.potential}!")
            self.m_e *= AMU_TO_AU
            self.M_1 *= AMU_TO_AU
            self.M_2 *= AMU_TO_AU

        self.mu   = xp.sqrt(self.M_1*self.M_2*self.m_e/(self.M_1+self.M_2+self.m_e))
        self.mur  = (self.M_1+self.M_2)*self.m_e/(self.M_1+self.M_2+self.m_e)
        self.mu12 = self.M_1*self.M_2/(self.M_1+self.M_2)
        self._Vfunc, extent_func = {
            'soft_coulomb': (potentials.soft_coulomb, potentials.extents_soft_coulomb),
            'borgis': (potentials.borgis, potentials.extents_borgis),
            }[args.potential]

        extent = extent_func(self.mu12)

        print(f"Potential: {args.potential}")

        if hasattr(args, "extent") and args.extent is not None:
            extent = args.extent

        R_range = extent[:2]
        r_max   = extent[-1]
        print("r_max",r_max)
        print("R_range",R_range)
        print("unscaled coords:", R_range, r_max)

        if r_max < R_range[-1]/2:
            raise RuntimeError("r_max should be at least R_max/2")

        print("  scaled coords:", R_range, r_max)

        self.R = xp.linspace(*R_range, args.NR)
        self.r = xp.linspace(r_max/args.Nr, r_max, args.Nr)

        # require Ng to be even
        if args.Ng % 2 != 0:
            raise RuntimeError(f"Ng must be even!")

        # N.B.: It is essential that we not include the endpoint in
        # gamma lest our cyclic grid be ill-formed and 2nd derivatives
        # all over the place
        self.g = xp.linspace(0, 2*xp.pi, args.Ng, endpoint=False)

        self.axes = (self.R, self.r, self.g)

        self.shape = (args.NR, args.Nr, args.Ng)
        self.size = args.NR * args.Nr * args.Ng

        dR = self.R[1] - self.R[0]
        dr = self.r[1] - self.r[0]
        dg = self.g[1] - self.g[0]

        self.P  = xp.fft.fftshift(xp.fft.fftfreq(args.NR, dR)) * 2 * xp.pi
        self.RP_grid = xp.meshgrid(self.R, self.P, indexing='ij')
        # N.B.: These all lack the factor of -1/(2 * mu)
        # We also are throwing away the returned jacobian of R/r
        #self.ddR2, _ = KE_Borisov(self.R, bare=True)
        self.ddR2    = KE(args.NR, dR, bare=True, cyclic=False) + xp.diag(1/4/self.R**2)
        self.ddr2, _ = KE_Borisov(self.r, bare=True)

        # Part of the reason for using a cyclic *stencil* for gamma
        # rather than KE_FFT is that it wasn't immediately obvious how
        # I would represent ∂/∂γ. (∂²/∂γ² was clear.)  N.B.: The
        # default stencil degree is 11
        self.ddg2 = KE(args.Ng, dg, bare=True, cyclic=True)
        #self.ddg1 = KE(args.Ng, dg, bare=True, cyclic=True, order=1)
        self.Pg = xp.fft.fftshift(xp.fft.fftfreq(args.Ng, dg)) * 2 * xp.pi
        self.ddg1 = gamma_grad(args.Ng, self.Pg, self.g)
        self.ddr1, _ = KE_Borisov(self.r, bare=True, order=1)

        self.R_grid, self.rp_grid, self.gp_grid = xp.meshgrid(self.R, self.r, self.g, indexing='ij')
        self.r_grid, self.g_grid = xp.meshgrid(self.r, self.g, indexing='ij')
        self.Vgrid = self.V(self.R_grid, self.rp_grid, self.gp_grid)

        # since we need these in Hx; maybe fine to compute on the fly?
        self.rinv2 = 1.0/(self.r)**2
        
        self._lock()

    def V(self, R, r, gamma, spitvals=False):

        mu12 = self.mu12
        M_1 = self.M_1
        M_2 = self.M_2

        kappa2 = r*R*xp.cos(gamma)

        r1e2 = (r)**2 + (R)**2*(mu12/M_1)**2 - 2*kappa2*mu12/M_1
        r2e2 = (r)**2 + (R)**2*(mu12/M_2)**2 + 2*kappa2*mu12/M_2

        r1e = xp.sqrt(xp.where(r1e2 < 0, 0, r1e2))
        r2e = xp.sqrt(xp.where(r2e2 < 0, 0, r2e2))
        
        if spitvals == True:
            return r1e2,r2e2
        else:
            return self._Vfunc(R, r1e, r2e, (self.g_1, self.g_2))

    # ensure that arrays cannot be written to
    def _lock(self):
        self._locked = True
        for key in self.__slots__:
            try:
                member = super().__getattribute__(key)
            except AttributeError:
                continue
            if (key not in self._mutable_keys and
                isinstance(member := super().__getattribute__(key), xp.ndarray)):
                member.flags.writeable = False

def compute_BO(H, Nelec,Ad_n):
    NR, Nr, Ng = H.shape
    Ad_vn = xp.zeros((NR, Nelec))

    for i in range(Nelec):
        Hbo = -1/(2*H.mu12)*H.ddR2 + xp.diag(Ad_n[:,i])
        #Ad_vn[:,i], U_v[i] = batch_eigh(Hbo)
        Ad_vn[:,i] = batch_eigvalsh(Hbo)
    
    return Ad_vn, Ad_n

def compute_EPS(info):
    i, k, H, args, GammatotR, Gammatotth, GammasqtotR, Gammasqtotth, Hel, v_diag = info
    
    #Gammamat = -1j*GammatotR*H.P[k]/H.mu12 -1j*Gammatotth*(H.J/H.R[i])/H.mu12-(GammasqtotR+Gammasqtotth)/(2*H.mu12) 
    #Gammamat = -1j*GammatotR*H.P[k]/H.mu12 -1j*Gammatotth*(H.J/H.R[i])/H.mu12
               
    #Htot = Hel+v_diag+Gammamat

    e_approx,e_wave = xp.linalg.eigh(Htot)
        
    eps_val = e_approx[0] + 0.5 * H.P[k]**2 / H.mu12 + 0.5 * (H.J/H.R[i])**2 /H.mu12

    return k,eps_val

def parse_args():
    parser = ap.ArgumentParser(
        prog='3body-2D',
        description="computes the lowest k eigenvalues of a 3-body potential in 2D")
    
    class NumpyArrayAction(ap.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            setattr(namespace, self.dest, xp.array(values, dtype=float))

    parser.add_argument('-k', metavar='num_eigenvalues', default=5, type=int)
    parser.add_argument('-t', metavar="num_threads", default=1, type=int)
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
    parser.add_argument('--potential', choices=['soft_coulomb', 'borgis'],
                        default='soft_coulomb')
    parser.add_argument('--extent', metavar="X", action=NumpyArrayAction,
                        nargs=3, help="Rmin Rmax rmax, in Bohr "
                        "(typically set automatically)")
    parser.add_argument('--backend', default='numpy')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    print(args)
    xp.backend = args.backend
    
    H = Hamiltonian(args)
    NR,Nr,Ng = H.shape
    Nelec = Nr*Ng 
    EPS = xp.zeros((NR, NR))
    
    start_script = perf_counter()
    Hel = -1/(2*H.mur)*(xp.kron(H.ddr2,xp.eye(H.shape[2])) + xp.kron(xp.diag(H.rinv2), H.ddg2))
    
    EPS = xp.zeros((H.shape[0], H.shape[0]))
    EPS_bo = xp.zeros((H.shape[0], H.shape[0]))
    ival = xp.zeros([H.shape[0],1])
    Ad_n  = xp.zeros((NR, Nelec))
    #Htotbo = xp.zeros([H.shape[0],Nelec,Nelec])
    
    mem_thresh = 1e5
    memory_constrained = H.size > mem_thresh

    print(f"memory constraint threshold = {mem_thresh}, {memory_constrained}")

    batch_eigvalsh = xp.linalg.eigvalsh
    if xp.backend == 'cupy':
        try:
            print("cupy detected; trying diagonalization with torch backend")
            import torch
        except ModuleNotFoundError:
            print("torch not found; using cupy")
        else:
            def torch_eigvalsh(H):
                return xp.asarray(torch.linalg.eigvalsh(torch.from_dlpack(H)))
            batch_eigvalsh = torch_eigvalsh         

    for i in range(H.shape[0]):
        print("i",i,flush=True)
        v_diag = xp.diag(H.Vgrid[i,:,:].ravel())
        Htotbo = Hel+v_diag
        e_approx_bo = batch_eigvalsh(Htotbo)
        
        Ad_n[i]  = e_approx_bo[:Nelec]
        ival[i] = e_approx_bo[0]

        r1e2, r2e2 = H.V(H.R[i], H.r_grid, H.g_grid, spitvals=True)
        gammaetf1x, gammaetf1y, gammaetf2x, gammaetf2y, gammaerf1y, gammaerf2y = Gamma_etf_erf(H.R[i],H.r_grid,H.g_grid,H.ddr1,H.ddg1,H.M_1,H.M_2,H.mu12,r1e2,r2e2)
        #gammaetf1x, gammaetf1y, gammaetf2x, gammaetf2y, gammaerf1y, gammaerf2y = xp.zeros([Nr*Ng,Nr*Ng]),xp.zeros([Nr*Ng,Nr*Ng]),xp.zeros([Nr*Ng,Nr*Ng]),xp.zeros([Nr*Ng,Nr*Ng]),xp.zeros([Nr*Ng,Nr*Ng]),xp.zeros([Nr*Ng,Nr*Ng])
        gamma1x = gammaetf1x
        gamma2x = gammaetf2x
        gamma1y = gammaetf1y
        gamma2y = gammaetf2y

        Gammatotx = (H.M_2*gamma1x-H.M_1*gamma2x)/(H.M_1+H.M_2)
        Gammatoty = (H.M_2*gamma1y-H.M_1*gamma2y)/(H.M_1+H.M_2)  

        gammasq1x = xp.dot(gamma1x,gamma1x)
        gammasq2x = xp.dot(gamma2x,gamma2x)
        gammasq1y = xp.dot(gamma1y,gamma1y)
        gammasq2y = xp.dot(gamma2y,gamma2y)

        Gammasqtotx = ((H.M_2**2*gammasq1x)+(H.M_1**2*gammasq2x)-(H.M_1*H.M_2*xp.dot(gamma1x,gamma2x))-(H.M_1*H.M_2*xp.dot(gamma2x,gamma1x)))/(H.M_1+H.M_2)**2
        Gammasqtoty = ((H.M_2**2*gammasq1y)+(H.M_1**2*gammasq2y)-(H.M_1*H.M_2*xp.dot(gamma1y,gamma2y))-(H.M_1*H.M_2*xp.dot(gamma2y,gamma1y)))/(H.M_1+H.M_2)**2 
        

        for j in range(H.shape[0]):
            print("i,j",i,j,flush=True)
            
            #Gammamat = -1j*Gammatotx*(H.P[j]-1/(2*H.R[i]))/H.mu12 -1j*Gammatoty*(H.J/H.R[i])/H.mu12-(Gammasqtotx+Gammasqtoty)/(2*H.mu12)
            Gammamat = -1j*Gammatotx*(H.P[j]-1/(2*H.R[i]))/H.mu12 -1j*Gammatoty*(H.J/H.R[i])/H.mu12           
            Htotgamma = Hel+v_diag+Gammamat
            #e_approx_ps,e_wave = batch_eigh(Htotgamma)
            e_approx_ps = batch_eigvalsh(Htotgamma)
            EPS[i,j] = e_approx_ps[0]


    Rval, Pval = H.RP_grid
    
    Helmat = xp.repeat(ival,H.shape[0],axis=1)
    EPS_bo += Helmat   
    EPS_bo += 1/(2*H.mu12)*(Pval**2+(H.J/Rval)**2-(1/4/Rval**2))
    HPS_bo = inverse_weyl_transform(EPS_bo, H.shape[0], H.R, H.P)
    EPSv_bo = batch_eigvalsh(HPS_bo)
    print("PS BO vib gap",EPSv_bo[1]-EPSv_bo[0],flush=True)

    EPS += 1/(2*H.mu12)*(Pval**2+(H.J/Rval)**2-(1/4/Rval**2))
    HPS = inverse_weyl_transform(EPS, H.shape[0], H.R, H.P)
    EPSv = batch_eigvalsh(HPS)
    print("PS vib gap",EPSv[1]-EPSv[0],flush=True)

    Ad_vn, Ad_n = compute_BO(H, Nelec,Ad_n)
    e_bo = xp.sort(Ad_vn.flatten())
    bo = e_bo[1] - e_bo[0]
    print("BO vib gap",bo,flush=True)
    print("Ad_n",Ad_n[:5,0],Ad_n[:5,1],flush=True)
    print("Ad_vn",Ad_vn[:5,0],Ad_vn[:5,1],flush=True)
    
    end_script = perf_counter()  
    print("Cupy time",end_script-start_script,flush=True)
    exit()
      
    Ad_vn, Ad_n = compute_BO(H, Nelec,Ad_n)
    e_bo = xp.sort(Ad_vn.flatten())
    bo = e_bo[1] - e_bo[0]
    print("BO vib gap",bo)
    print("Ad_n",Ad_n[:5,0],Ad_n[:5,1])
    print("Ad_vn",Ad_vn[:5,0],Ad_vn[:5,1])

      
    print("ps, bo, error:", ex, bo, (bo-ex)/ex)

    if args.evecs:
        np.savez(args.evecs, guess=psiPSv, R=H.R)
        print("Wrote eigenvectors to", args.evecs)

    if args.save is not None:
        with open(args.save, "a") as f:
            print(args.M_1, args.M_2, args.g_1, args.g_2, args.J,
                  " ".join(map(str, EPSv)), file=f)
            
            print("\nBO", " ".join(map(str, Ad_vn)), file=f)

        print(f"Computed fixed center-of-mass eigenvalues",
              f"for M_1={args.M_1}, M_2={args.M_2} amu",
              f"with charges g_1={args.g_1}, g_2={args.g_2}",
              f"and total J={args.J}",
              f"and appended to {args.save}")
