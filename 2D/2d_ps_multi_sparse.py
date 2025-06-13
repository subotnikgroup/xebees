#!/usr/bin/env python
import numpy as np

from sys import stderr
import argparse as ap
from pathlib import Path
#from pyscf import lib as pyscflib
import jax
import jax.numpy as jnp
from functools import partial
import multiprocessing
import os, sys 
sys.path.append(os.path.abspath("lib"))
import xp
import nvtx

from constants import *
from hamiltonian import  KE, KE_FFT, KE_Borisov, inverse_weyl_transform
from davidson import get_davidson_guess, get_davidson_mem, solve_exact_gen, eye_lazy
from debug import prms, timer, timer_ctx
from threadpoolctl import ThreadpoolController
import concurrent.futures as cf
import potentials
from time import perf_counter
import cupyx.scipy.sparse.linalg as cpx_linalg
import cupyx.scipy.sparse as cpx_sp


if __name__ == '__main__':
    from tqdm import tqdm
else:  # mock this out for use in Jupyter Notebooks etc
    def tqdm(iterator, **kwargs):
        print(f"Mock call to tqdm({kwargs})")
        return iterator

class Hamiltonian:
    __slots__ = ( # any new members must be added here
        'm_e', 'M_1', 'M_2', 'mu', 'g_1', 'g_2', 'J','mur',
        'R', 'P', 'R_grid', 'r', 'p', 'r_grid', 'g', 'pg', 'g_grid',
        'ddR2', 'ddr2', 'ddg2', 'ddg1','ddr1','axes','Vgrid','rp_grid','gp_grid',
        'Rinv2', 'rinv2', 'diag', '_preconditioner_data',
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

        # N.B.: These all lack the factor of -1/(2 * mu)
        # We also are throwing away the returned jacobian of R/r
        #self.ddR2, _ = KE_Borisov(self.R, bare=True)
        self.ddR2    = KE(args.NR, dR, bare=True, cyclic=False)
        self.ddr2, _ = KE_Borisov(self.r, bare=True)

        # Part of the reason for using a cyclic *stencil* for gamma
        # rather than KE_FFT is that it wasn't immediately obvious how
        # I would represent ∂/∂γ. (∂²/∂γ² was clear.)  N.B.: The
        # default stencil degree is 11
        self.ddg2 = KE(args.Ng, dg, bare=True, cyclic=True)
        self.ddg1 = KE(args.Ng, dg, bare=True, cyclic=True, order=1)
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
            return r1e,r2e
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

def compute_BO(H, Nelec):
    NR, Nr, Ng = H.shape
    U_n   = xp.zeros((NR, Nr*Ng, Nelec))
    U_v   = xp.zeros((Nelec, NR, NR))
    Ad_n  = xp.zeros((NR, Nelec))
    Ad_vn = xp.zeros((NR, Nelec))

    def diag_Hel(i, Ad_n=Ad_n):
        v_diag = cpx_sp.diags(H.Vgrid[i,:,:].ravel())
        Hel = -1/(2*H.mur)*(cpx_sp.kron(H.ddr2,cpx_sp.eye(H.shape[2]))+cpx_sp.kron(cpx_sp.diags(H.rinv2), H.ddg2))
        val, vec = cpx_linalg.eigsh(Hel+v_diag,return_eigenvectors=False)
        Ad_n[i] = val[:Nelec]

    threadctl = ThreadpoolController()
    with cf.ThreadPoolExecutor(max_workers=16) as ex, threadctl.limit(limits=1):
        list(tqdm(
            ex.map(diag_Hel, range(NR)),
            total=NR, desc="Building electronic surfaces"))

    def diag_Hbo(i, Ad_n=Ad_n, Ad_vn=Ad_vn):
        Hbo = -1/(2*H.mu12)*H.ddR2 + cpx_sp.diags(Ad_n[:,i])
        Ad_vn[:,i] = cpx_linalg.eigsh(Hb,return_eigenvectors=Falseo)
        #Ad_vn[:,i], U_v[i] = xp.linalg.eigh(Hbo)

    with cf.ThreadPoolExecutor(max_workers=16) as ex, threadctl.limit(limits=1):
        list(tqdm(
            ex.map(diag_Hbo, range(Nelec)),
            total=Nelec, desc="Building vibrational states"))
    
    return Ad_vn, U_n, U_v, Ad_n


@nvtx.annotate("gamma_build", color="red")
def Gamma_etf_erf(R,r,g,pr,pg,M_1,M_2,mu12,r1e,r2e):

    Ng = len(pg)
    Nr = len(pr)

    theta1 = xp.exp(-r1e)
    theta2 = xp.exp(-r2e)
    partition = theta1 + theta2

    cosgamma = xp.cos(g)
    singamma = xp.sin(g)
    invr = 1/r[:,0]

    t1 = cpx_sp.diags((theta1/partition).ravel())
    t2 = cpx_sp.diags((theta2/partition).ravel())

    px =  cpx_sp.kron(pr,cpx_sp.diags(cosgamma[0,:])) - cpx_sp.kron(cpx_sp.diags(invr),cpx_sp.diags(singamma[0,:]).dot(pg))
    py =  cpx_sp.kron(pr,cpx_sp.diags(singamma[0,:])) + cpx_sp.kron(cpx_sp.diags(invr),cpx_sp.diags(cosgamma[0,:]).dot(pg))
    
    t1px = t1.dot(px)
    pxt1 = px.dot(t1)
    t2px = t2.dot(px)
    pxt2 = px.dot(t2)
    t1py = t1.dot(py)
    pyt1 = py.dot(t1)
    t2py = t2.dot(py)
    pyt2 = py.dot(t2)

    gammaetf1x = -0.5*(t1px + pxt1)
    gammaetf1y = -0.5*(t1py + pyt1)
    gammaetf2x = -0.5*(t2px + pxt2)
    gammaetf2y = -0.5*(t2py + pyt2)

    rcosg = cpx_sp.diags((r*cosgamma).ravel())
    rsing = cpx_sp.diags((r*singamma).ravel())

    J1 = -0.5*((rcosg-(cpx_sp.eye(Nr*Ng)*R*mu12/M_1)).dot(t1py+pyt1)-rsing.dot(t1px+pxt1))
    J2 = -0.5*((rcosg+(cpx_sp.eye(Nr*Ng)*R*mu12/M_2)).dot(t2py+pyt2)-rsing.dot(t2px+pxt2))

    #check signs
    #flip signs because of the cross product
    gammaerf1y = 1/R*(J1+J2)
    gammaerf2y = -1/R*(J1+J2)

    return gammaetf1x, gammaetf1y, gammaetf2x, gammaetf2y, gammaerf1y, gammaerf2y

def compute_EPS(info):
    i, k, H, args, GammatotR, Gammatotth, GammasqtotR, Gammasqtotth, Hel, v_diag = info
    
    #Gammamat = -1j*GammatotR*H.P[k]/H.mu12 -1j*Gammatotth*(H.J/H.R[i])/H.mu12-(GammasqtotR+Gammasqtotth)/(2*H.mu12) 
    Gammamat = -1j*GammatotR*H.P[k]/H.mu12 -1j*Gammatotth*(H.J/H.R[i])/H.mu12
               
    Htot = Hel+v_diag+Gammamat

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
    EPS = xp.zeros((H.shape[0], H.shape[0]))

    threadctl = ThreadpoolController()
    threadctl.limit(limits=args.t)

    start_script = perf_counter()
    with nvtx.annotate("Building Hel no V", color="cyan"):
        t0 = perf_counter()
        Hel = -1/(2*H.mur)*(cpx_sp.kron(H.ddr2,cpx_sp.eye(H.shape[2]))+cpx_sp.kron(cpx_sp.diags(H.rinv2), H.ddg2))
        #print("Building Hel no V time: ", perf_counter() - t0)


    for i in range(H.shape[0]):

        v_diag = cpx_sp.diags(H.Vgrid[i,:,:].ravel())
        
        with nvtx.annotate("Building r1e", color="gray"):
            t0 = perf_counter()
            r1e, r2e = H.V(H.R[i], H.r_grid, H.g_grid, spitvals=True)
            
        gammaetf1x, gammaetf1y, gammaetf2x, gammaetf2y, gammaerf1y, gammaerf2y = Gamma_etf_erf(H.R[i],H.r_grid,H.g_grid,H.ddr1,H.ddg1,H.M_1,H.M_2,H.mu12,r1e,r2e)
        #print(gammaetf1x.toarray())
        
        #print(type(gammaetf1x), type(gammaetf1y), type(gammaetf2x), type(gammaetf2y), type(gammaerf1y), type(gammaerf2y))
        
        with nvtx.annotate("gamma_construct", color="green"):
            M_1 = H.M_1
            M_2 = H.M_2
            t0 = perf_counter()
            gamma1x = gammaetf1x
            gamma2x = gammaetf2x
            gamma1y = gammaetf1y+gammaerf1y
            gamma2y = gammaetf2y+gammaerf2y

            gammasq1x = gamma1x@gamma1x
            gammasq2x = gamma2x@gamma2x
            gammasq1y = gamma1y@gamma1y
            gammasq2y = gamma2y@gamma2y

            Gammatotx = (M_2*gamma1x-M_1*gamma2x)/(M_1+M_2)
            Gammatoty = (M_2*gamma1y-M_1*gamma2y)/(M_1+M_2)  

            Gammasqtotx = ((M_2**2*gammasq1x)+(M_1**2*gammasq2x)-(M_1*M_2*gamma1x@gamma2x)-(M_1*M_2*gamma2x@gamma1x))/(M_1+M_2)**2
            Gammasqtoty = ((M_2**2*gammasq1y)+(M_1**2*gammasq2y)-(M_1*M_2*gamma1y@gamma2y)-(M_1*M_2*gamma2y@gamma1y))/(M_1+M_2)**2      
            
        #print(type(Gammasqtotx),type(Gammasqtoty))   
        index_pairs = [(i, k, H, args, Gammatotx, Gammatoty, Gammasqtotx, Gammasqtoty, Hel, v_diag) for k in range(H.shape[0])]


        for j in range(H.shape[0]):
            print(i,j)
            with nvtx.annotate("Building Hel", color="yellow"):
                t0 = perf_counter()
                Gammamat = -1j*Gammatotx*H.P[j]/H.mu12 -1j*Gammatoty*(H.J/H.R[i])/H.mu12    
                Htot = Hel+v_diag+Gammamat
                #print(type(Htot))

            Htot_new = xp.copy(Htot.toarray())

            with nvtx.annotate("eigh", color="blue"):
                t0 = perf_counter()

                if xp.backend == 'cupy':
                    try:
                        print("cupy detected; trying diagonalization with torch backend")
                        import torch
                        vals, vecs = torch.linalg.eigh(torch.from_dlpack(Htot_new))
                        e_approx, U_n = xp.asarray(vals), xp.asarray(vecs)
                    except ModuleNotFoundError:
                        print("failed; using cupy")
                        e_approx, U_n = xp.linalg.eigh(Htot_new)
                else:
                    #Ad_n, U_n = xp.linalg.eigh(Hel)
                    e_approx = xp.linalg.eigvalsh(Htot_new)
                          
            #with nvtx.annotate("eigh", color="blue"):
            #    t0 = perf_counter()
            #    Htot_new = xp.copy(Htot.toarray())
            #    e_approx = cpx_linalg.eigsh(Htot,return_eigenvectors=False)
            #    e_approx = cpx_linalg.eigsh(Htot,which='SA',return_eigenvectors=False)
            #    #print(e_approx)
            #    #e_approx = xp.linalg.eigvalsh(Htot_new)
            #    print("e_approx",e_approx)
                         
            eps_val = e_approx[0] + 0.5 * H.P[j]**2 / H.mu12 + 0.5 * (H.J/H.R[i])**2 /H.mu12
            
            EPS[i,j] = eps_val
    np.savez("EPS.npz", EPS=EPS, R=H.R, P=H.P)

    end_script = perf_counter()  

    print("Sparse time",end_script-start_script)     
    
    HPS = inverse_weyl_transform(EPS, H.shape[0], H.R, H.P)

    with nvtx.annotate("final eigh", color="purple"):
        t0 = perf_counter()
        EPSv, psiPSv = xp.linalg.eigh(HPS)
        #print("final eigh time: ", perf_counter() - t0)
    
    print("EPSv",EPSv)

    #Ad_vn, U_n, U_v, Ad_n = compute_BO(H, Nelec)
    #e_bo = xp.sort(Ad_vn.flatten())
    #bo = e_bo[1] - e_bo[0]
    #print("BO vib gap",bo)
    print("PS vib gap",EPSv[1]-EPSv[0])
    #ex = EPSv[1] - EPSv[0]   
    #print("ps, bo, error:", ex, bo, (bo-ex)/ex)

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
