#!/usr/bin/env python

# Why is the standard eigen_solver so slow??
#   cupyx.cusolver.syevj?
#   cuDSS
# why do we use so much memory when we have cupynumeric in the conda environment?
# what do we need to do to support the jax.numpy backend?
# memory concerns
# nvtx and timing annotations
#
# Explore reduced precision preconditioner

#import jax
#import jax.numpy as jnp
#jax.config.update('jax_enable_x64', True)

from numpy.fft import fft, fftshift
from scipy.integrate import simpson
from scipy.sparse.linalg import lobpcg
from scipy.interpolate import RegularGridInterpolator

from sys import stderr
import argparse as ap
from pathlib import Path

import concurrent.futures as cf
from itertools import product
from functools import reduce, partial
import operator

import os, sys
sys.path.append(os.path.abspath("lib"))

import xp
import numpy  # only use this for reading and writing objects
import linalg_helper as lib
#from pyscf import lib
import potentials
from constants import *
from hamiltonian import  KE, KE_FFT, KE_Borisov, inverse_weyl_transform
from davidson import phase_match, phase_match_mem_constrained, get_interpolated_guess, get_davidson_mem, solve_exact_gen, eye_lazy
from debug import prms, timer, timer_ctx
from threadpoolctl import ThreadpoolController
from time import perf_counter

if __name__ == '__main__':
    from tqdm import tqdm
else:  # mock this out for use in Jupyter Notebooks etc
    def tqdm(iterator, **kwargs):
        print(f"Mock call to tqdm({kwargs})")
        return iterator


class Hamiltonian:
    __slots__ = ( # any new members must be added here
        'm_e', 'M_1', 'M_2', 'mu', 'g_1', 'g_2', 'J','mur',
        'R', 'P_R', 'R_grid', 'r', 'p', 'r_grid','p_grid', 'g', 'pg', 'g_grid','RP_grid', 'ddp2', 'ddp1',
        'ddR2', 'ddr2', 'ddg2', 'ddg1', 'ddr1', 'ddr2','axes','Vgrid','rb_grid','gb_grid','pb_grid',
        'Rinv2', 'rinv2', 'diag', '_preconditioner_data','Pg',
        'shape', 'size','guess','k','mu12','_Vfunc',
        '_locked','max_threads'
    )

    def __init__(self, args):
        # save number of threads for preconditioner
        self.max_threads = getattr(args, "t", 1)

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
        #if args.Ng % 2 != 0:
        #    raise RuntimeError(f"Ng must be even!")

        # N.B.: It is essential that we not include the endpoint in
        # gamma lest our cyclic grid be ill-formed and 2nd derivatives
        # all over the place
        self.g = xp.linspace(0, 2*xp.pi, args.Ng, endpoint=False)
        self.p = xp.linspace(0, 2*xp.pi, args.Np, endpoint=True)##XXXXX  check this

        self.axes = (self.R, self.r, self.g, self.p)

        self.shape = (args.NR, args.Nr, args.Ng, args.Np)
        self.size = args.NR * args.Nr * args.Ng * args.Np

        dR = self.R[1] - self.R[0]
        dr = self.r[1] - self.r[0]
        dg = self.g[1] - self.g[0]
        dp = self.p[1] - self.p[0] 

        self.P_R  = xp.fft.fftshift(xp.fft.fftfreq(args.NR, dR)) * 2 * xp.pi
        self.RP_grid = xp.meshgrid(self.R, self.P_R, indexing='ij')
        # N.B.: These all lack the factor of -1/(2 * mu)
        # We also are throwing away the returned jacobian of R/r
        #self.ddR2, _ = KE_Borisov(self.R, bare=True)
        self.ddR2    = KE(args.NR, dR, bare=True, cyclic=False) + xp.diag(1/4/self.R**2)
        self.ddr2, _ = KE_Borisov(self.r, bare=True)
        self.ddr1, _ = KE_Borisov(self.r, bare=True, order=1)

        # Part of the reason for using a cyclic *stencil* for gamma
        # rather than KE_FFT is that it wasn't immediately obvious how
        # I would represent ∂/∂γ. (∂²/∂γ² was clear.)  N.B.: The
        # default stencil degree is 11
        self.ddg2 = KE(args.Ng, dg, bare=True, cyclic=True)
        self.ddg1 = KE(args.Ng, dg, bare=True, cyclic=True, order=1)

        self.ddp2 = KE(args.Np, dp, bare=True, cyclic=True)
        self.ddp1 = KE(args.Np, dp, bare=True, cyclic=True, order=1)
    
        self.R_grid, self.rb_grid, self.pb_grid, self.gb_grid = xp.meshgrid(self.R, self.r, self.p, self.g, indexing='ij')
        self.r_grid, self.p_grid, self.g_grid,  = xp.meshgrid(self.r, self.p, self.g, indexing='ij')
        self.Vgrid = self.V(self.R_grid, self.rb_grid, self.pb_grid, self.gb_grid)

        # since we need these in Hx; maybe fine to compute on the fly?
        self.rinv2 = 1.0/(self.r)**2
        

        # Lock the object and protect arrays from writing
        if xp.backend != 'torch':
            def recursive_lock(obj):
                if isinstance(obj, xp.ndarray):
                    obj.flags.writeable=False
                elif isinstance(obj, tuple):
                    (recursive_lock(x) for x in obj)

            for key in self.__slots__:
                if hasattr(self, key):
                    recursive_lock(super().__getattribute__(key))

        
        self._locked = True

    def V(self, R, r, gamma, psi, spitvals=False):

        mu12 = self.mu12
        M_1 = self.M_1
        M_2 = self.M_2

        kappa2 = r*R*xp.cos(gamma)*xp.cos(psi)

        r1e2 = (r)**2 + (R)**2*(mu12/M_1)**2 - 2*kappa2*mu12/M_1
        r2e2 = (r)**2 + (R)**2*(mu12/M_2)**2 + 2*kappa2*mu12/M_2

        r1e = xp.sqrt(xp.where(r1e2 < 0, 0, r1e2))
        r2e = xp.sqrt(xp.where(r2e2 < 0, 0, r2e2))
        
        if spitvals == True:
            return r1e2,r2e2
        else:
            return self._Vfunc(R, r1e, r2e, (self.g_1, self.g_2))
    
    def build_Hel(self, Ridx=None):
        NR, Nx, Ny = self.shape
        Nelec = Nx*Ny
        Hel = xp.empty((NR, Nelec, Nelec), dtype=self.dtype)
        Hel[:] = -1/(2*self.mur)*(xp.kron(self.ddx2,xp.eye(Ny)) + xp.kron(xp.eye(Nx), self.ddy2))

        if Ridx is None:
            Ridx = xp.arange(NR)
        else:
            Ridx = xp.atleast_1d(Ridx)
            NR,  = Ridx.shape

        Hel[:, xp.arange(Nelec), xp.arange(Nelec)] +=(  # extract diagonal at every R
            xp.reshape(self.Vgrid[Ridx], (NR, Nelec))   # + V
        )

        return xp.squeeze(Hel)

    def _output_info(self):

        NR,Nx,Ny = self.shape
        Nelec = Nx*Ny
        Ad_n  = xp.zeros((NR, Nelec))
        He = -1/(2*self.mur)*(xp.kron(self.ddx2,xp.eye(Ny)) + xp.kron(xp.eye(Nx), self.ddy2))
    
        for i in range(self.shape[0]):
            print(i,"i")
            v_diag = xp.diag(self.Vgrid[i,:,:].ravel())
            Hel = He+v_diag
            e_approx_bo = xp.linalg.eigvalsh(Hel)
            Ad_n[i] = e_approx_bo[:Nelec]
        
        Ad_vn = xp.zeros((NR, Nelec))
        U_v = xp.zeros((Nelec,NR,NR))
        for i in range(5):
            print(i,"j")
            Hbo = -1/(2*self.mu12)*self.ddR2 + xp.diag(Ad_n[:,i])
            Ad_vn[:,i], U_v[i] = xp.linalg.eigh(Hbo)

        #Hbo = xp.empty((Nelec, NR, NR))                # Hbo = -1/2/μ(∂²/∂R² + 1/4/R²) + V_n
        #Hbo[:] = -1/(2*self.mu12)*self.ddR2         #       -1/2/μ(∂²/∂R² + 1/4/R²)
        #Hbo[:, xp.arange(NR), xp.arange(NR)] += Ad_n.T
        #Ad_vn, U_v = batch_eigh(Hbo)
        pc = (Ad_vn,U_v,Ad_n)
        #print("Ad_n",Ad_n)

        return pc

def Gamma_etf_polar(R,r,p,g,ddr,ddp,ddg,M_1,M_2,mu12,r1e2,r2e2):

    Ng = len(ddg)
    Nr = len(ddr)
    Np = len(ddp)
    
    theta1 = xp.exp(-r1e2)
    theta2 = xp.exp(-r2e2)
    partition = theta1 + theta2

    cosgamma = xp.cos(g+xp.pi/2)
    singamma = xp.sin(g+xp.pi/2)
    cospsi = xp.cos(p)
    sinpsi = xp.sin(p)

    invr = 1/(r[:,0])

    t1 = xp.diag((theta1/partition).ravel())
    t2 = xp.diag((theta2/partition).ravel())

    #xp.fill_diagonal(spg, xp.diag(spg) * singamma[0,:])
    #xp.fill_diagonal(cpg, xp.diag(cpg) * cosgamma[0,:])

    pr =  xp.kron(xp.kron(ddr, xp.diag(singamma[0,:,0])), xp.diag(cospsi[0,0,:])) + xp.kron(xp.kron(xp.diag(invr), xp.diag(cosgamma[0,:,0])*ddg), cospsi[0,0,:]) + xp.kron(xp.kron(xp.diag(invr), xp.diag(singamma[0,:,0])), xp.diag(sinpsi)*ddp)
    pp =  xp.kron(xp.kron(ddr, xp.diag(singamma[0,:,0])), xp.diag(sinpsi[0,0,:])) + xp.kron(xp.kron(xp.diag(invr), xp.diag(singamma[0,:,0])*ddg), sinpsi[0,0,:]) + xp.kron(xp.kron(xp.diag(invr), xp.diag(singamma[0,:,0])), xp.diag(cospsi)*ddp)
    pt =  xp.kron(xp.kron(ddr, xp.diag(cosgamma[0,:,0])), xp.eye(Np)) - xp.kron(xp.kron(xp.diag(invr), singamma[0,:,0])*ddg1, xp.eye(Np)) 

    t1pr = xp.dot(t1,pr)
    prt1 = xp.dot(pr,t1)
    t2pr = xp.dot(t2,pr)
    prt2 = xp.dot(pr,t2)

    t1pp = xp.dot(t1,pp)
    ppt1 = xp.dot(pp,t1)
    t2pp = xp.dot(t2,pp)
    ppt2 = xp.dot(pp,t2)

    t1pt = xp.dot(t1,pt)
    ptt1 = xp.dot(pt,t1)
    t2pt = xp.dot(t2,pt)
    ptt2 = xp.dot(pt,t2)

    gammaetf1r = -0.5*(t1pr + prt1)
    gammaetf1t = -0.5*(t1pt + ptt1)
    gammaetf1p = -0.5*(t1pp + ppt1)

    gammaetf2r = -0.5*(t2pr + prt2)
    gammaetf2t = -0.5*(t2pt + ptt2)
    gammaetf2p = -0.5*(t2pp + ppt2)
    
    return gammaetf1r, gammaetf1p, gammaetf1t, gammaetf2r, gammaetf2p, gammaetf2t    

def compute_EPS(info):

    Rval, Pval, Htot_bo, gammacoeff_R, gammacoeff_phi, gammacoeff_theta, \
    Gammatotr, Gammatotp, Gammatott, Gammasqtotr, Gammasqtotp,Gammasqtott, mu12 = info
    
    #print("i,j",Rval,Pval,gammacoeff_R[Rval,Pval],flush=True)           
    
    Htot = Htot_bo[Rval]+(gammacoeff_R[Rval]*gammatotr)+(gammacoeff_phi[Rval]*gammatotphi)+(gammacoeff_theta[Rval]*gammatottheta)
    Htotsq = Htot - (Gammasqtotr +Gammasqtotp+ Gammasqtott)/(2*mu12)
    
    e_approx = xp.linalg.eigvalsh(Htot)
    e_approxsq = xp.linalg.eigvalsh(Htotsq)
    
    return Rval,Pval,e_approx[0],e_approxsq[0]


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
    parser.add_argument('-Pphi', default=0, type=float)
    parser.add_argument('-Ptheta', default=0, type=float)
    parser.add_argument('-R', dest="NR", metavar="NR", default=101, type=int)
    parser.add_argument('-r', dest="Nr", metavar="Nr", default=400, type=int)
    parser.add_argument('-gamma', dest="Ng", metavar="Ng", default=250, type=int)
    parser.add_argument('-psi', dest="Np", metavar="Np", default=250, type=int)
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

    # you can only select the backend once and it must be before you use any xp functions
    if xp.backend != args.backend:
        xp.backend = args.backend

    if xp.backend == 'jax.numpy':
        import jax
        jax.config.update('jax_enable_x64', True)
    elif xp.backend == 'torch':
        xp.set_default_dtype(xp.float64)

    print("threads",args.t)
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

    H = Hamiltonian(args)
    start_script = perf_counter()
    
    NR,Nr,Ng,Np = H.shape
    Nelec = Nr*Ng*Np 
    
    Hel = -1/(2*H.mur)*(
        xp.kron(xp.kron(H.ddr2, xp.eye(H.shape[Ng])), xp.eye(H.shape[Np])) +\
        xp.kron(xp.kron(xp.diag(H.rinv2), xp.diag(xp.cot(H.g+xp.pi/2))*H.ddg1), xp.eye(H.shape[Np]))+\
        xp.kron(xp.kron(xp.diag(H.rinv2), H.ddg2), H.shape[Np])+\
        xp.kron(xp.kron(xp.diag(H.rinv2), xp.sin(H.g+xp.pi/2)**2),H.ddp2)
        )

    Htot_bo = xp.zeros([NR,Nelec,Nelec,Nelec])
    Htot_bo[:] = Hel
    Htot_bo[:,xp.arange(Nelec),xp.arange(Nelec),xp.arange(Nelec)] += xp.reshape(H.Vgrid[:],(NR,Nelec))#XXXXXcheck this 
    
    ival = xp.zeros([NR,1])
    def Htot_R(i):
        return Htot_bo[i]
   
    threadctl = ThreadpoolController()
    with threadctl.limit(limits=1), cf.ThreadPoolExecutor(max_workers=args.t) as ex:
        result = list(tqdm(ex.map(lambda i: (i, xp.linalg.eigvalsh(Htot_R(i))), range(NR)), total=NR))
        Ad_n = xp.zeros(NR)
        for i, a in result:
            Ad_n[i] = a[0]
            ival[i,0] = a[0]
    
    EPS = xp.zeros((H.shape[0], H.shape[0]))
    EPSsq = xp.zeros((H.shape[0], H.shape[0]))

    gammacoeff_R = -1j*H.P/H.mu12 #XXXXXX check this if factor -1/4R present
    gammacoeff_phi = -1j*(H.Pphi/H.R)/H.mu12
    gammacoeff_theta = +1j*(H.Ptheta/H.R)/H.mu12

    Gammasqtotr = xp.zeros([Nelec,Nelec,Nelec],dtype=complex)
    Gammasqtott = xp.zeros([Nelec,Nelec,Nelec],dtype=complex)
    Gammatotr = xp.zeros([Nelec,Nelec,Nelec],dtype=complex)
    Gammatott = xp.zeros([Nelec,Nelec,Nelec],dtype=complex)

    for i in range(H.shape[0]):
        print("i",i,flush=True)

        r1e2, r2e2 = H.V(H.R[i], H.r_grid, H.g_grid, spitvals=True)
        with timer_ctx("build gamma"):
            gammaetf1r, gammaetf1p, gammaetf1t, gammaetf2r, gammaetf2p, gammaetf2t = Gamma_etf_polar(H.R[i],H.r_grid,H.p_grid,H.g_grid,H.ddr1,H.ddp1,H.ddg1,H.M_1,H.M_2,H.mu12,r1e2,r2e2)
            gammaetr1r, gammaerf1p, gammaerf1t, gammaerf2r, gammaerf2p, gammaerf2t = Gamma_erf_polar(H.R[i],H.r_grid,H.p_grid,H.g_grid,H.ddr1,H.ddp1,H.ddg1,H.M_1,H.M_2,H.mu12,r1e2,r2e2)

        gamma1r = gammaetf1r+gammaerf1r
        gamma2r = gammaetf2r+gammaerf2r
        gamma1t = gammaetf1t+gammaerf1t
        gamma2t = gammaetf2t+gammaerf2t
        gamma1p = gammaetf1p+gammaerf1p
        gamma2p = gammaetf2p+gammaerf2p

        Gammatotr = (H.M_2*gamma1r-H.M_1*gamma2r)/(H.M_1+H.M_2)
        Gammatotp = (H.M_2*gamma1p-H.M_1*gamma2p)/(H.M_1+H.M_2)
        Gammatott = (H.M_2*gamma1t-H.M_1*gamma2t)/(H.M_1+H.M_2)
        
        gammasq1r = xp.dot(gamma1r,gamma1r)
        gammasq2r = xp.dot(gamma2r,gamma2r)
        gamma1r2r = xp.dot(gamma1r,gamma2r)
        gamma2r1r = xp.dot(gamma2r,gamma1r)

        gammasq1p = xp.dot(gamma1p,gamma1p)
        gammasq2p = xp.dot(gamma2p,gamma2p)       
        gamma1p2p = xp.dot(gamma1p,gamma2p)
        gamma2p1p = xp.dot(gamma2p,gamma1p)

        gammasq1t = xp.dot(gamma1t,gamma1t)
        gammasq2t = xp.dot(gamma2t,gamma2t)       
        gamma1t2t = xp.dot(gamma1t,gamma2t)
        gamma2t1t = xp.dot(gamma2t,gamma1t)

        Gammasqtotr = ((H.M_2**2*gammasq1r)+(H.M_1**2*gammasq2r)-(H.M_1*H.M_2*gamma1r2r)-(H.M_1*H.M_2*gamma2r1r))/(H.M_1+H.M_2)**2
        Gammasqtotp = ((H.M_2**2*gammasq1p)+(H.M_1**2*gammasq2p)-(H.M_1*H.M_2*gamma1p2p)-(H.M_1*H.M_2*gamma2p1p))/(H.M_1+H.M_2)**2
        Gammasqtott = ((H.M_2**2*gammasq1t)+(H.M_1**2*gammasq2t)-(H.M_1*H.M_2*gamma1t2t)-(H.M_1*H.M_2*gamma2t1t))/(H.M_1+H.M_2)**2 

        index_pairs = [(i, k, Htot_bo, gammacoeff_R, gammacoeff_phi, gammacoeff_theta, Gammatotr, Gammatotp, Gammatott, Gammasqtotr, Gammasqtotp,Gammasqtott, H.mu12) for k in range(NR)]
               
        threadctl = ThreadpoolController()
        h_workers = min(args.t, H.shape[0])    
        blasthreads = max(args.t//h_workers, 1)
 
        #blasthreads x max_workers =< args.t =< 48
        with cf.ThreadPoolExecutor(max_workers=h_workers) as ex, threadctl.limit(limits=blasthreads):
            results = list(tqdm(
                ex.map(compute_EPS, index_pairs),
                total=H.shape[0], desc="Building EPS"))
        for i,k,val,valsq in results:
            EPS[i, k] = val
            EPSsq[i, k] = valsq

    Hbo_new = -1/(2*H.mu12)*(H.ddR2 - xp.diag(H.Pphi**2/H.R**2)- xp.diag(H.Ptheta**2/H.R**2)) +xp.diag(Ad_n)
    Ad_vn_new = batch_eigvalsh(Hbo_new)
    e_bo_new = xp.sort(Ad_vn_new.flatten())
    bo_new = e_bo_new[1] - e_bo_new[0]
    print("BO new vib gap",bo_new,flush=True)
        
    EPS += 1/(2*H.mu12)*(Pval**2+H.Pphi**2/Rval**2+H.Ptheta**2/Rval**2)
    HPS = inverse_weyl_transform(EPS, H.shape[0], H.R, H.P)
    EPSv = batch_eigvalsh(HPS)
    print("PS vib gap",EPSv[1]-EPSv[0],flush=True)

    EPSsq += 1/(2*H.mu12)*(Pval**2+H.Pphi**2/Rval**2+H.Ptheta**2/Rval**2)
    HPSsq = inverse_weyl_transform(EPSsq, H.shape[0], H.R, H.P)
    EPSvsq = batch_eigvalsh(HPSsq)
    print("PS vib gap sq",EPSvsq[1]-EPSvsq[0],flush=True)

    EPS_bo = xp.zeros((H.shape[0], H.shape[0]))
    Helmat = xp.repeat(ival,H.shape[0],axis=1)
    EPS_bo += Helmat   
    EPS_bo += 1/(2*H.mu12)*(Pval**2+H.Pphi**2/Rval**2+H.Ptheta**2/Rval**2)
    HPS_bo = inverse_weyl_transform(EPS_bo, H.shape[0], H.R, H.P)
    EPSv_bo = batch_eigvalsh(HPS_bo)
    print("Weyl BO vib gap",EPSv_bo[1]-EPSv_bo[0],flush=True)

    end_script = perf_counter()  
    print("Numpy time",end_script-start_script,flush=True)

    
