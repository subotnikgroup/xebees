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
        'm_e', 'M_1', 'M_2', 'mu', 'mu12', 'mur', 'aa', 'g_1', 'g_2', 'J',
        'R', 'P', 'R_grid', 'x', 'p', 'x_grid', 'y', 'pg', 'j', 'y_grid',
        'axes', 'dtype', 'args',
        'max_threads','xp_grid','yp_grid',
        'preconditioner', 'make_guess', '_Vfunc',
        'Vgrid', 'ddR2', 'ddx2', 'ddy2', 'ddx', 'ddy',
        'Rinv2', 'rinv2', 'diag', '_preconditioner_data','theta',
        'shape', 'size',
        '_locked', '_hash', 'r_lab', 'R_lab', 'ddr_lab2', 'ddR_lab2','RP_grid'
    )

    def __init__(self, args):
        # save number of threads for preconditioner
        self.max_threads = getattr(args, "t", 1)

        self.m_e = 1
        self.M_1 = args.M_1
        self.M_2 = args.M_2

        self.g_1 = args.g_1
        self.g_2 = args.g_2

        self.J   = args.J
        self.dtype = xp.float64 if self.J == 0 else xp.complex128

        #self.theta = xp.pi*args.theta/180

        # Potential function selection
        if not hasattr(args, "potential"):
            args.extent = 'soft_coulomb'

        if args.potential == 'borgis' or args.potential == 'original':
            print(f"Waring: All masses scaled to AMU for {args.potential}!")
            self.m_e *= AMU_TO_AU
            self.M_1 *= AMU_TO_AU
            self.M_2 *= AMU_TO_AU

        self.mu   = numpy.sqrt(self.M_1*self.M_2*self.m_e/(self.M_1+self.M_2+self.m_e))
        self.mur  = (self.M_1+self.M_2)*self.m_e/(self.M_1+self.M_2+self.m_e)
        self.mu12 = self.M_1*self.M_2/(self.M_1+self.M_2)
        self.aa   = numpy.sqrt(self.mu12/self.mu) # factor of 'a' for lab and scaled coordinates
        self._Vfunc, extent_func = {
            'soft_coulomb': (potentials.soft_coulomb, potentials.extents_soft_coulomb),
            'borgis': (potentials.borgis, potentials.extents_borgis),
            }[args.potential]

        extent = extent_func(self.mu12)

        print(f"Potential: {args.potential}")

        if hasattr(args, "extent") and args.extent is not None:
            extent = args.extent

        print("extent",extent)

        R_min = extent[0]
        R_max = extent[1]
        #x_min = -6
        #x_max = 2
        #y_min = -5
        #y_max = 5
        x_min = -extent[2]
        x_max = extent[2]
        y_min = -extent[2]
        y_max = extent[2]

        self.R = xp.linspace(R_min, R_max, args.NR)
        self.x = xp.linspace(x_min, x_max, args.Nx)
        self.y = xp.linspace(y_min, y_max, args.Ny)

        self.axes = (self.R, self.x, self.y)

        self.R_grid, self.xp_grid, self.yp_grid = xp.meshgrid(self.R, self.x, self.y, indexing='ij')
        self.Vgrid = self.V(self.R_grid, self.xp_grid, self.yp_grid)
        self.x_grid, self.y_grid = xp.meshgrid(self.x, self.y, indexing='ij')
        self.shape = self.Vgrid.shape
        self.size = xp.prod(xp.asarray(self.shape))

        dR = self.R[1] - self.R[0]
        dx = self.x[1] - self.x[0]
        dy = self.y[1] - self.y[0]

        self.P  = xp.fft.fftshift(xp.fft.fftfreq(args.NR, dR)) * 2 * xp.pi
        self.RP_grid = xp.meshgrid(self.R, self.P, indexing='ij')


        self.ddR2 = KE(args.NR, dR, bare=True, cyclic=False) + xp.diag(1/4/self.R**2)
        
        self.ddx2 = KE(args.Nx, dx, bare=True, cyclic=False)
        self.ddx = KE(args.Nx, dx, bare=True, cyclic=False, order=1)
        self.ddy2 = KE(args.Ny, dy, bare=True, cyclic=False)
        self.ddy = KE(args.Ny, dy, bare=True, cyclic=False, order=1)

        # since we need these in Hx; maybe fine to compute on the fly?

        #builder, self.preconditioner, self.make_guess = {
        #    'BO':     (self._build_preconditioner_BO,        self._preconditioner_BO,        self._make_guess_BO)
        #    }[args.preconditioner]
#
        #with timer_ctx(f"Build preconditioner {args.preconditioner}"):
        #    self._preconditioner_data = builder()


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

        self._hash = numpy.random.randint(2**63)  # self._make_hash()
        self._locked = True

    def V(self, R, rx, ry,spitvals=False):

        mu12 = self.mu12
        M_1 = self.M_1
        M_2 = self.M_2

        #r1e2 = (rx)**2 + (ry)**2 + (R)**2*(mu12/M_1)**2 - 2*R*mu12/M_1*(rx*xp.cos(self.theta)+ry*xp.sin(self.theta))
        #r2e2 = (rx)**2 + (ry)**2 + (R)**2*(mu12/M_2)**2 + 2*R*mu12/M_2*(rx*xp.cos(self.theta)+ry*xp.sin(self.theta))

        r1e2 = (rx)**2 + (ry)**2 + (R)**2*(mu12/M_1)**2 - 2*R*mu12/M_1*rx
        r2e2 = (rx)**2 + (ry)**2 + (R)**2*(mu12/M_2)**2 + 2*R*mu12/M_2*rx

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

def Gamma_etf_cart(R,x,y,ddx,ddy,M_1,M_2,mu12,r1e2,r2e2):

    Nx = len(ddx)
    Ny = len(ddy)
    px = xp.kron(ddx,xp.eye(Ny))
    py = xp.kron(xp.eye(Nx),ddy)
    
    theta1 = xp.exp(-r1e2)
    theta2 = xp.exp(-r2e2)
    partition = theta1 + theta2

    t1 = xp.diag((theta1/partition).ravel())
    t2 = xp.diag((theta2/partition).ravel())
       
    t1px = xp.dot(t1,px)
    pxt1 = xp.dot(px,t1)
    t2px = xp.dot(t2,px)
    pxt2 = xp.dot(px,t2)
    t1py = xp.dot(t1,py)
    pyt1 = xp.dot(py,t1)
    t2py = xp.dot(t2,py)
    pyt2 = xp.dot(py,t2)

    gammaetf1x = -0.5*(t1px + pxt1)
    gammaetf1y = -0.5*(t1py + pyt1)
   
    gammaetf2x = -0.5*(t2px + pxt2)
    gammaetf2y = -0.5*(t2py + pyt2)

    return gammaetf1x, gammaetf1y, gammaetf2x, gammaetf2y   

def compute_EPS(info):
    
    Rval, Pval, Htot_bo, gammacoeff_R, gammacoeff_theta, gammatotx, gammatoty, gammasqtotx, gammasqtoty = info
    #print("i,j",Rval,Pval,flush=True)           
    
    Htot = Htot_bo[Rval]+(gammacoeff_R[Rval,Pval]*gammatotx)+(gammacoeff_theta[Rval]*gammatoty)
    Htot_sq = Htot - gammasqtotx - gammasqtoty 
    e_approx = xp.linalg.eigvalsh(Htot)
    e_approx_sq = xp.linalg.eigvalsh(Htot_sq)   

    return Rval,Pval,e_approx[0],e_approx_sq[0]


def parse_args():
    parser = ap.ArgumentParser(
        prog='3body-2D',
        description="computes the lowest k eigenvalues of a 3-body potential in 2D")

    class ArrayAction(ap.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            setattr(namespace, self.dest, xp.array(values, dtype=float))

    parser.add_argument('-k', metavar='num_eigenvalues', default=5, type=int)
    parser.add_argument('-t', metavar="num_threads", default=16, type=int)
    parser.add_argument('-g_1', metavar='g_1', required=True, type=float)
    parser.add_argument('-g_2', metavar='g_2', required=True, type=float)
    parser.add_argument('-M_1', required=True, type=float)
    parser.add_argument('-M_2', required=True, type=float)
    parser.add_argument('-J', default=0, type=float)
    parser.add_argument('-R', dest="NR", metavar="NR", default=101, type=int)
    parser.add_argument('-x', dest="Nx", metavar="Nx", default=400, type=int)
    parser.add_argument('-y', dest="Ny", metavar="Ny", default=158, type=int)
    parser.add_argument('-theta', dest="theta", metavar="theta", default=0, type=float)
    parser.add_argument('--potential', choices=['soft_coulomb', 'borgis'],
                        default='soft_coulomb')
    parser.add_argument('--extent', metavar="X", action=ArrayAction,
                        nargs=3, help="Rmin Rmax rmax, in Bohr "
                        "(typically set automatically)")
    parser.add_argument('--exact_diagonalization', action='store_true')
    parser.add_argument('--bo_spectrum', metavar='spec.npz', type=Path, default=None)
    parser.add_argument('--preconditioner', choices=['naive', 'V1', 'BO', 'BO-int', 'jfull'],
                        default="naive", type=str)
    parser.add_argument('--verbosity', default=2, type=int)
    parser.add_argument('--backend', default='numpy')
    parser.add_argument('--iterations', metavar='max_iterations', default=10000, type=int)
    parser.add_argument('--subspace', metavar='max_subspace', default=1000, type=int)
    parser.add_argument('--guess', metavar="guess.npz", type=Path, default=None)
    parser.add_argument('--evecs', metavar="guess.npz", type=Path, default=None)
    parser.add_argument('--save', metavar="filename")

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

    with timer_ctx("Build H"):
        H = Hamiltonian(args)
    start_script = perf_counter()
    NR,Nx,Ny = H.shape
    Nelec = Nx*Ny
    
    Hel = -1/(2*H.mur)*(xp.kron(H.ddx2,xp.eye(H.shape[2])) + xp.kron(xp.eye(H.shape[1]), H.ddy2))

    Htot_bo_test = xp.zeros([NR,Nelec,Nelec])
    Htot_bo_test[:] = Hel
    Htot_bo_test[:,xp.arange(Nelec),xp.arange(Nelec)] += xp.reshape(H.Vgrid[:],(NR,Nelec))
    #Ad_n = np.zeros(NR)
    ival = xp.zeros([NR,1])
    def Htot_R(i):
        return Htot_bo_test[i]

    
    threadctl = ThreadpoolController()
    with threadctl.limit(limits=1), cf.ThreadPoolExecutor(max_workers=args.t) as ex:
        result = list(tqdm(ex.map(lambda i: (i, xp.linalg.eigvalsh(Htot_R(i))), range(NR)), total=NR))
        Ad_n = xp.zeros(NR)
        for i, a in result:
            Ad_n[i] = a[0]
            ival[i,0] = a[0]
    
    #e_approx_bo = batch_eigvalsh(Htot_bo_test)
    #ival_check = e_approx_bo[:,0][None].T
    #Ad_n_check = e_approx_bo[:,0]

    EPS = xp.zeros((H.shape[0], H.shape[0]))
    EPSsq = xp.zeros((H.shape[0], H.shape[0]))
    
    Rval, Pval = H.RP_grid
    gammacoeff_R = -1j*(Pval-1/(2*Rval))/H.mu12
    gammacoeff_theta = -1j*(H.J/H.R)/H.mu12
    Gammasqtotx = xp.zeros([Nelec,Nelec],dtype=complex)
    Gammasqtoty = xp.zeros([Nelec,Nelec],dtype=complex)
    Gammatotx = xp.zeros([Nelec,Nelec],dtype=complex)
    Gammatoty = xp.zeros([Nelec,Nelec],dtype=complex)

    for i in range(H.shape[0]):
        print("i",i,flush=True)

        r1e2, r2e2 = H.V(H.R[i], H.x_grid, H.y_grid, spitvals=True)
        with timer_ctx("build gamma"):
            gammaetf1x, gammaetf1y, gammaetf2x, gammaetf2y = Gamma_etf_cart(H.R[i],H.x_grid,H.y_grid,H.ddx,H.ddy,H.M_1,H.M_2,H.mu12,r1e2,r2e2)
        
        Gammatotx = (H.M_2*gammaetf1x-H.M_1*gammaetf2x)/(H.M_1+H.M_2)
        Gammatoty = (H.M_2*gammaetf1y-H.M_1*gammaetf2y)/(H.M_1+H.M_2)
        gammasq1x = xp.dot(gammaetf1x,gammaetf1x)
        gammasq2x = xp.dot(gammaetf2x,gammaetf2x)
        gammasq1y = xp.dot(gammaetf1y,gammaetf1y)
        gammasq2y = xp.dot(gammaetf2y,gammaetf2y)

        Gammasqtotx = ((H.M_2**2*gammasq1x)+(H.M_1**2*gammasq2x)-(H.M_1*H.M_2*xp.dot(gammaetf1x,gammaetf2x))-(H.M_1*H.M_2*xp.dot(gammaetf2x,gammaetf1x)))/(H.M_1+H.M_2)**2
        Gammasqtoty = ((H.M_2**2*gammasq1y)+(H.M_1**2*gammasq2y)-(H.M_1*H.M_2*xp.dot(gammaetf1y,gammaetf2y))-(H.M_1*H.M_2*xp.dot(gammaetf2y,gammaetf1y)))/(H.M_1+H.M_2)**2 

        index_pairs = [(i, k, Htot_bo_test, gammacoeff_R, gammacoeff_theta,Gammatotx,Gammatoty,Gammasqtotx, Gammasqtoty) for k in range(NR)]


        threadctl = ThreadpoolController()
        h_workers = min(args.t, H.shape[0])    
        blasthreads = 1 # min(h_workers/args.t, 1)
 
        with cf.ThreadPoolExecutor(max_workers=h_workers) as ex, threadctl.limit(limits=blasthreads):
            results = list(tqdm(
               ex.map(compute_EPS, index_pairs),
               total=H.shape[0], desc="Building EPS"))
        for i,k,val,valsq in results:
            EPS[i, k] = val
            EPSsq[i, k] = valsq

    #blasthreads x max_workers =< args.t =< 48
    
    EPS += 1/(2*H.mu12)*(Pval**2-(1/4/Rval**2)+H.J**2/Rval**2)
    HPS = inverse_weyl_transform(EPS, H.shape[0], H.R, H.P)
    EPSv = batch_eigvalsh(HPS)
    print("PS vib gap",EPSv[1]-EPSv[0],flush=True)

    EPSsq += 1/(2*H.mu12)*(Pval**2-(1/4/Rval**2)+H.J**2/Rval**2)
    HPSsq = inverse_weyl_transform(EPSsq, H.shape[0], H.R, H.P)
    EPSvsq = batch_eigvalsh(HPSsq)
    print("PS vib gap sq",EPSvsq[1]-EPSvsq[0],flush=True)

    Hbo_new = -1/(2*H.mu12)*(H.ddR2 - xp.diag(H.J**2/H.R**2)) +xp.diag(Ad_n)
    Ad_vn_new = batch_eigvalsh(Hbo_new)
    e_bo_new = xp.sort(Ad_vn_new.flatten())
    bo_new = e_bo_new[1] - e_bo_new[0]
    print("BO new vib gap",bo_new,flush=True)

    EPS_bo = xp.zeros((H.shape[0], H.shape[0]))
    Helmat = xp.repeat(ival,H.shape[0],axis=1)
    EPS_bo += Helmat   
    EPS_bo += 1/(2*H.mu12)*(Pval**2-(1/4/Rval**2)+H.J**2/Rval**2)
    HPS_bo = inverse_weyl_transform(EPS_bo, H.shape[0], H.R, H.P)
    EPSv_bo = batch_eigvalsh(HPS_bo)
    print("Weyl BO vib gap",EPSv_bo[1]-EPSv_bo[0],flush=True)

    if args.evecs:
        
        EPS += 1/(2*H.mu12)*(Pval**2-(1/4/Rval**2)+H.J**2/Rval**2)
        HPS = inverse_weyl_transform(EPS, H.shape[0], H.R, H.P)
        EPSv,EPSvwfn = xp.linalg.eigh(HPS)
        print("PS vib gap",EPSv[1]-EPSv[0],flush=True)

        EPSsq += 1/(2*H.mu12)*(Pval**2-(1/4/Rval**2)+H.J**2/Rval**2)
        HPSsq = inverse_weyl_transform(EPSsq, H.shape[0], H.R, H.P)
        EPSvsq,EPSvsqwfn = xp.linalg.eigh(HPSsq)
        print("PS vib gap sq",EPSvsq[1]-EPSvsq[0],flush=True)
        
        numpy.savez_compressed(args.evecs, EPS=EPSvwfn, H=H.R)
        numpy.savez_compressed("SQ"+str(args.evecs), EPS=EPSvsqwfn, H=H.R)
        print("Wrote eigenvectors to", args.evecs)

    end_script = perf_counter()  
    print("Numpy time",end_script-start_script,flush=True)

    
