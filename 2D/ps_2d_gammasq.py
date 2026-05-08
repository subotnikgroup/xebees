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
        'R', 'P_R', 'R_grid', 'x', 'p', 'x_grid', 'y', 'pg', 'j', 'y_grid',
        'axes', 'dtype', 'args',
        'max_threads','xp_grid','yp_grid',
        'preconditioner', 'make_guess', '_Vfunc',
        'Vgrid', 'ddR2', 'ddx2', 'ddy2', 'ddx', 'ddy', 'ddr'
        'Rinv2', 'rinv2', 'diag', '_preconditioner_data','theta',
        'shape', 'size','boshape','shape','r',
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
        x_min = -extent[2]
        x_max = extent[2]
        y_min = -extent[2]
        y_max = extent[2]

        self.R = xp.linspace(R_min, R_max, args.NR)
        self.x = xp.linspace(x_min, x_max, args.Nx)
        self.y = xp.linspace(y_min, y_max, args.Ny)
        self.r = xp.sqrt(self.x[:,None]**2+self.y[None,:]**2)


        self.shape = (args.NR, args.Nx, args.Ny)
        self.boshape = (args.Nx, args.Ny)

        self.axes = (self.R, self.x, self.y)

        self.R_grid, self.xp_grid, self.yp_grid = xp.meshgrid(self.R, self.x, self.y, indexing='ij')
        self.Vgrid = self.V(self.R_grid, self.xp_grid, self.yp_grid)
        self.x_grid, self.y_grid = xp.meshgrid(self.x, self.y, indexing='ij')
        self.shape = self.Vgrid.shape
        self.size = xp.prod(xp.asarray(self.shape))

        dR = self.R[1] - self.R[0]
        dx = self.x[1] - self.x[0]
        dy = self.y[1] - self.y[0]

        self.P_R  = xp.fft.fftshift(xp.fft.fftfreq(args.NR, dR)) * 2 * xp.pi
        # self.P_R  = xp.fft.fftfreq(args.NR, dR) * 2 * xp.pi
        self.RP_grid = xp.meshgrid(self.R, self.P_R, indexing='ij')


        self.ddR2 = KE(args.NR, dR, bare=True, cyclic=False) + xp.diag(1/4/self.R**2)
        
        self.ddx2 = KE(args.Nx, dx, bare=True, cyclic=False)
        self.ddx = KE(args.Nx, dx, bare=True, cyclic=False, order=1)
        self.ddy2 = KE(args.Ny, dy, bare=True, cyclic=False)
        self.ddy = KE(args.Ny, dy, bare=True, cyclic=False, order=1)


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
    
def Gamma_etf(H,t1):
    # Building ETF terms
    ddx, ddy = H.ddx, H.ddy
    
    t1px = xp.einsum('xy,xa->xay', t1, ddx, optimize=True)
    pxt1 = xp.einsum('xa,ay->xay', ddx, t1, optimize=True)
    t1py = xp.einsum('xy,yb->xyb', t1, ddy, optimize=True)
    pyt1 = xp.einsum('yb,xb->xyb', ddy, t1, optimize=True)
    gammaetf1x = -0.5*(t1px + pxt1)
    gammaetf1y = -0.5*(t1py + pyt1)
    
    return gammaetf1x, gammaetf1y

def Gamma_erf(H,t1,t2,Ridx):
    # Building ERF terms
    rx, ry = H.x,H.y
    Nx, Ny = H.boshape

    ddx1 = H.ddx[:, :, None]
    ddy1 = H.ddy[None, :, :]

    R = H.R[Ridx]
    mu12 = H.mu12
    M_1 = H.M_1
    M_2 = H.M_2
    coeff = H.R[Ridx]/2*((H.M_2*t1-H.M_1*t2)/(H.M_1+H.M_2))
    
    Jya = -1/H.R[Ridx]*xp.einsum('x,xyb->xyb', rx, ddy1, optimize=True)
    Jyb = -1/H.R[Ridx]*xp.einsum('xy,xyb->xyb', coeff, ddy1, optimize=True)
    Jyc = -1/H.R[Ridx]*xp.einsum('y,xay->xay', ry, ddx1, optimize=True)
    Jyd = -1/H.R[Ridx]*xp.einsum('xyb,xb->xyb', ddy1, coeff,optimize=True)

    return Jya, Jyb, Jyc, Jyd

def Tx(H,xdav):
    # Kinetic energy terms
    xdav = xdav.reshape((-1,) + H.boshape)
    Hel_dav = -1/(2*H.mur)*(
        xp.einsum('xa,Bay->Bxy',H.ddx2,xdav,optimize=True)
        +xp.einsum('yb,Bxb->Bxy',H.ddy2,xdav,optimize=True)
        )
    return Hel_dav.reshape(xdav.shape)


def Gamma_sq_erf(H, t1,t2,Ridx):
    # Building ERF squared terms
    rx, ry = H.x,H.y
    Nx, Ny = H.boshape

    ddx1 = H.ddx[:, :, None]
    ddy1 = H.ddy[None, :, :]

    R = H.R[Ridx]
    coeff = H.R[Ridx]/2*((H.M_2*t1-H.M_1*t2)/(H.M_1+H.M_2))
    rxe = rx[:,None]-coeff
    
    J1 = xp.einsum('xy,xyb,xb,xbY->xyY', rxe, ddy1, rxe, ddy1, optimize=True)
    J2 = -xp.einsum('xy,xyb,b,xab->xayb',rxe, ddy1, ry, ddx1, optimize=True)
    J3 = -xp.einsum('xy,xyb,xbY,xY->xyY', rxe, ddy1, ddy1, coeff, optimize=True)
    
    J4 = -xp.einsum('y,xay,ay,ayb->xayb', ry, ddx1, rxe, ddy1, optimize=True)
    J5 = +xp.einsum('y,xay,y,aXy->xXy', ry, ddx1, ry, ddx1, optimize=True)
    J6 = +xp.einsum('y,xay,ay,ayb->xayb', ry, ddx1, coeff, ddy1, optimize=True)

    J7 = -xp.einsum('xyb, xb, xb, xbY->xyY', ddy1, coeff, rxe, ddy1, optimize=True)
    J8 = +xp.einsum('xyb, xb, b, xab->xayb', ddy1, coeff, ry, ddx1, optimize=True)
    J9 = +xp.einsum('xyb, xb, xbY, xY->xyY', ddy1, coeff, ddy1, coeff, optimize=True)

    dydy = (J1+J3+J7+J9)/R**2
    dxdy = (J2+J4+J6+J8)/R**2
    dxdx = J5/R**2
    
    diagsq = (rxe**2*xp.diag(H.ddy2)[None,:]+ry**2*xp.diag(H.ddx2)[:,None]+coeff**2*xp.diag(H.ddy2)[None,:])/R**2

    return dydy, dxdy, dxdx, diagsq.flatten()

def Gamma_etf_erf(H,gammaetfy,t1,t2,Ridx):
    # Building ETF-ERF terms they are only in the y direction
    rx, ry = H.x,H.y
    ddx1 = H.ddx[:, :, None]
    ddy1 = H.ddy[None, :, :]
    R = H.R[Ridx]
    coeff = R/2*((H.M_2*t1-H.M_1*t2)/(H.M_1+H.M_2))
    rxe = rx[:,None]-coeff
    etfy = gammaetfy
    #etferf1 = +xp.einsum('xyb,xb,xbY->xyY',etfy,rxe,ddy1,optimize=True)
    #etferf2 = -xp.einsum('xyb,b,xab->xayb',etfy,ry,ddx1,optimize=True)
    #etferf3 = -xp.einsum('xyb,xbY,xY->xyY',etfy,ddy1,coeff,optimize=True)
#
    #erfetf1 = +xp.einsum('xy,xyb,xbY->xyY',rxe,ddy1,etfy,optimize=True)
    #erfetf2 = -xp.einsum('y,xay,ayb->xayb',ry,ddx1,etfy,optimize=True)
    #erfetf3 = -xp.einsum('xyb,xb,xbY->xyY',ddy1,coeff,etfy,optimize=True)

    etferf1 = -xp.einsum('xyb,xb,xbY->xyY',etfy,rxe,ddy1,optimize=True)
    etferf2 = +xp.einsum('xyb,b,xab->xayb',etfy,ry,ddx1,optimize=True)
    etferf3 = +xp.einsum('xyb,xbY,xY->xyY',etfy,ddy1,coeff,optimize=True)

    erfetf1 = -xp.einsum('xy,xyb,xbY->xyY',rxe,ddy1,etfy,optimize=True)
    erfetf2 = +xp.einsum('y,xay,ayb->xayb',ry,ddx1,etfy,optimize=True)
    erfetf3 = +xp.einsum('xyb,xb,xbY->xyY',ddy1,coeff,etfy,optimize=True)

    dydy = (etferf1+etferf3+erfetf1+erfetf3)/R
    dxdy = (etferf2+erfetf2)/R

    return dydy,dxdy


def Hbo_dav(H,i):
    # Building the BO Hamiltonian
    def Hxbo(xdav):
        x = xdav.reshape((-1,)+H.boshape)        
        Hbodav = H.Vgrid[i]*x + Tx(H,x)## xxxcheck with xdav shape
        return Hbodav.reshape(xdav.shape)
    return Hxbo

def buildDiag(H,Ri):
    # Building the diagonal of the KE and PE of Hamiltonian
    # Useful for the preconditioner
    Nx, Ny = H.boshape
    ke  = xp.zeros([Nx,Ny])
    ke += xp.diag(H.ddx2)[:,None]
    ke += xp.diag(H.ddy2)[None,:]
    ke *= -1 / (2*H.mur)
    diag = H.Vgrid[Ri] + ke
    return diag.ravel()

def ps_ham(H,ddx_terms,ddy_terms,Ri, dxdy_term=None):
    # Building the PS Hamiltonian
    def Hx_ps(xdav):
        x = xdav.reshape((-1,)+H.boshape).astype(complex) 
               
        Hpsdav = (
            H.Vgrid[Ri]*x + Tx(H,x) 
            +xp.einsum('xay,Bay->Bxy', ddx_terms, x, optimize=True) 
            +xp.einsum('xyb,Bxb->Bxy', ddy_terms, x, optimize=True) )
        if dxdy_term is not None:
            Hpsdav += xp.einsum('xayb,Bab->Bxy', dxdy_term, x, optimize=True) 

        return Hpsdav.reshape(xdav.shape)
    
    return Hx_ps



def apply_pr(H, xdav):
    # Applying the linar momentum operator in radial direction
    x = xdav.reshape((-1,)+H.boshape).astype(complex) 
    r = xp.sqrt(
            H.x[:,None]**2 + H.y[None,:]**2) #shape xy
    r[r<1e-10] = 1e-10 
    
    dxdr = H.x[:,None]/r ## sin(gamma)cos(psi)
    dydr = H.y[None,:]/r ## sin(gamma)sin(psi)
    
    ### Symmetrized product
    ddr1 = (0-1j)*0.5*(xp.einsum('xy, xa, Bxy -> Bay ', dxdr , H.ddx, x, optimize=True)
                    + xp.einsum('ay, xa, Bxy -> Bay', dxdr, H.ddx, x, optimize=True)
                    + xp.einsum('xy, yb, Bxy -> Bxb', dydr, H.ddy, x, optimize=True)
                    + xp.einsum('xb, yb, Bxy -> Bxb', dydr, H.ddy, x, optimize=True)
    )

    return ddr1.reshape(xdav.shape)

def apply_l(H, xdav):
    # Angular momentum operator
    x = xdav.reshape((-1,)+H.boshape).astype(complex) 

    lz_a = xp.einsum('x,yb,Bxb -> Bxy', H.x,H.ddy, x, optimize=True)
    lz_b = xp.einsum('y,xa,Bay -> Bxy', H.y,H.ddx, x, optimize=True)
    lz = -1j*(lz_a -lz_b)

    return lz



def parse_args():
    parser = ap.ArgumentParser(
        prog='3body-2D',
        description="computes the lowest k eigenvalues of a 3-body potential in 2D")

    def odd_int(s):
        v = int(s)
        if v % 2 != 1:
            raise ap.ArgumentTypeError(f'NR must be odd, got {v}')
        return v

    class ArrayAction(ap.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            setattr(namespace, self.dest, xp.array(values, dtype=float))

    parser.add_argument('-k', metavar='num_eigenvalues', default=2, type=int)
    parser.add_argument('-t', metavar="num_threads", default=4, type=int)
    parser.add_argument('-g_1', metavar='g_1', required=True, type=float)
    parser.add_argument('-g_2', metavar='g_2', required=True, type=float)
    parser.add_argument('-M_1', required=True, type=float)
    parser.add_argument('-M_2', required=True, type=float)
    parser.add_argument('-J', default=0, type=float)
    parser.add_argument('-R', dest="NR", metavar="NR", default=51, type=odd_int)
    parser.add_argument('-x', dest="Nx", metavar="Nx", default=51, type=int)
    parser.add_argument('-y', dest="Ny", metavar="Ny", default=51, type=int)
    parser.add_argument('-theta', dest="theta", metavar="theta", default=0, type=float)
    parser.add_argument('--potential', choices=['soft_coulomb', 'borgis'],
                        default='borgis')
    parser.add_argument('--extent', metavar="X", action=ArrayAction,
                        nargs=6, help="Rmin Rmax xmin xmax ymin ymax, in Bohr "
                        "(typically set automatically)")
    parser.add_argument('--exact_diagonalization', action='store_true')
    parser.add_argument('--bo_spectrum', metavar='spec.npz', type=Path, default=None)
    # parser.add_argument('--preconditioner', choices=['naive', 'V1', 'BO', 'BO-int', 'jfull'],
    #                     default="naive", type=str)
    parser.add_argument('--verbosity', default=2, type=int)
    parser.add_argument('--backend', default='numpy')
    parser.add_argument('--iterations', metavar='max_iterations', default=10000, type=int)
    parser.add_argument('--subspace', metavar='max_subspace', default=1000, type=int)
    parser.add_argument('--guess', metavar="guess.npz", type=Path, default=None)
    parser.add_argument('--evecs', metavar="guess.npz", type=Path, default=None)
    parser.add_argument('--save', metavar="filename")
    parser.add_argument('--no_ERF', action='store_true')
    parser.add_argument('--Gammasq', action='store_true')

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

    #kwargs = dict(optimize=True)
    #if xp.backend == 'torch':
    #    kwargs = {}
    #xp.einsum(..., **kwargs)  

    H = Hamiltonian(args)

    start_script = perf_counter()
    
    NR,Nx,Ny = H.shape
    Nelec = Nx*Ny 
    
    ival = xp.zeros([NR,1])
    Ad_n = xp.zeros(NR)

    EPS = xp.zeros((NR,NR))
    
    Rval, Pval = H.RP_grid
    gammacoeff_R = -1j*(Pval-1/(2*Rval))/H.mu12
    gammacoeff_theta = -1j*(H.J/H.R)/H.mu12

    ### other electronic observables
    pPS = xp.zeros((3, H.shape[0], H.shape[0]), dtype=xp.complex128) # <pe>(R,P)
    rPS = xp.zeros((H.shape[0], H.shape[0]), dtype=xp.complex128)
    GammaPS = xp.zeros((2, H.shape[0], H.shape[0]), dtype=xp.complex128)
    rBO = xp.zeros((H.shape[0]), dtype=xp.complex128)
    lPS = xp.zeros((H.shape[0], H.shape[0]), dtype=xp.complex128)
    l2PS = xp.zeros((H.shape[0], H.shape[0]), dtype=xp.complex128)
    l2BO = xp.zeros((H.shape[0]), dtype=xp.complex128)
    p2PS = xp.zeros((H.shape[0], H.shape[0]), dtype=xp.complex128)
    p2BO = xp.zeros((H.shape[0]), dtype=xp.complex128)
    

    evecs_prev = True
    for i in range(H.shape[0]):
        print("i",i,flush=True)
        diag = buildDiag(H,i)  

        guess = xp.exp(-(H.Vgrid[i] - xp.min(H.Vgrid[i]))**2/27.211**2).ravel()
        if evecs_prev == True:
            guess_bo = guess
        else:
            guess_bo = evecs


        conv, e_approx, evecs = lib.davidson1(
            Hbo_dav(H,i),
            guess_bo,
            lambda dx, e, x0: dx/(diag-e+1e-5),
            nroots=args.k,
            max_cycle=args.iterations,
            verbose=args.verbosity,
            max_space=args.subspace,
            max_memory=get_davidson_mem(0.75),
            #tol=1e-12, #FIXME:DEBUG
            tol=1e-12,
            )

        print("Davidson:", e_approx)
        print(conv)

        psi0_bo = evecs[0].reshape(H.boshape)
        rBO[i] = xp.einsum('xy,xy,xy->', psi0_bo.conj(), H.r, psi0_bo)
        psi0_lz = apply_l(H,evecs[0])
        lzBO_i = xp.sum(psi0_bo.conj()*psi0_lz)
        l2BO_i = xp.sum(psi0_lz.conj()*psi0_lz)
        l2BO[i] = l2BO_i
        print("lzBO:", lzBO_i)
        print("l2BO:", l2BO_i)
        psi0_p2 = (-2*H.mur)*Tx(H,psi0_bo)
        p2BO[i] = xp.sum(psi0_bo.conj()*psi0_p2)
        print("p2BO:", p2BO[i])


        Ad_n[i] = e_approx[0]
        ival[i,0] = e_approx[0]
        r1e2, r2e2 = H.V(H.R[i], H.x_grid, H.y_grid, spitvals=True)
        t1 = 1/(1+xp.exp(r1e2-r2e2))
        t2 = 1/(1+xp.exp(r2e2-r1e2))

        # Building Gamma ETF terms
        gammaetf1x,gammaetf1y= Gamma_etf(H, t1)
        gammaetf2x,gammaetf2y= Gamma_etf(H, t2)
        gammaetfx = (H.M_2*gammaetf1x-H.M_1*gammaetf2x)/(H.M_1+H.M_2)
        gammaetfy = (H.M_2*gammaetf1y-H.M_1*gammaetf2y)/(H.M_1+H.M_2)

        # Building Gamma ERF terms
        Jya, Jyb, Jyc, Jyd = Gamma_erf(H, t1, t2,i)

        # Building Gamma in y direction 
        ddy_terms = gammacoeff_theta[i]*(Jya-Jyb-Jyd+gammaetfy)
        if args.Gammasq:
            # Building Gamma ETF squared terms
            etfx_sq = xp.einsum('xay,aXy->xXy', gammaetfx, gammaetfx)
            etfy_sq = xp.einsum('xyb,xbY->xyY', gammaetfy, gammaetfy)
            # Building diagonal of Gamma ETF squared terms
            t_etfsq = (H.M_2*t1-H.M_1*t2)/((H.M_1+H.M_2))
            diag_etf_sq = (t_etfsq**2*xp.diag(H.ddx2)[:,None]
                          + t_etfsq**2*xp.diag(H.ddy2)[None,:]).flatten()
            # Building Gamma ERF squared terms
            (dydy_erf, dxdy_erf, dxdx_erf, diagsq_erf) = Gamma_sq_erf(H, t1, t2, i)
            # Building Gamma ETF-ERF terms
            dydy_etf_erf, dxdy_etf_erf = Gamma_etf_erf(H, gammaetfy, t1,t2, i)

            # Building Gamma squared terms in y direction
            ddy_terms += ( -dydy_erf- etfy_sq- dydy_etf_erf)/(2*H.mu12)
            # Building diagonal of Gamma squared terms
            gammasqdiag = -(diagsq_erf + diag_etf_sq)/(2*H.mu12)
            if args.no_ERF:
                # Building ONLY Gamma ETF in y direction
                ddy_terms = gammacoeff_theta[i]*(gammaetfy) - (etfy_sq)/(2*H.mu12)
                gammasqdiag = -(diag_etf_sq)/(2*H.mu12)
            
            diag += gammasqdiag

                            

        for j in range(NR):

            print("Atom Ri",i,"Atom Pj",j,flush=True)
            print("P",H.P_R[j])
            # Building Gamma in x direction
            ddx_terms = gammacoeff_R[i,j]*gammaetfx - gammacoeff_theta[i]*Jyc            
            if args.no_ERF:
                ddx_terms = gammacoeff_R[i,j]*gammaetfx
        
            Hps = ps_ham(H,ddx_terms,ddy_terms,i)

            if args.Gammasq:
                # Building Gamma ETF squared terms in x direction
                ddx_terms += - etfx_sq/(2*H.mu12) - dxdx_erf/(2*H.mu12)
                # Building Gamma ETF-ERF terms in x direction
                dxdy_terms = - (dxdy_erf+dxdy_etf_erf)/(2*H.mu12)
                Hps = ps_ham(H,ddx_terms,ddy_terms,i, dxdy_terms)
                if args.no_ERF:
                    # Building ONLY Gamma ETF in x direction
                    ddx_terms = gammacoeff_R[i,j]*gammaetfx - etfx_sq/(2*H.mu12)
                    Hps = ps_ham(H,ddx_terms,ddy_terms,i)


            if evecs_prev == True:
                guess_ps = evecs
                evecs_prev = False
            else:
                guess_ps = evecs_save

            
            with timer_ctx(f"Davidson of size {H.size}"):
                conv, e_ps_approx, evecs_save = lib.davidson1(
                    Hps,
                    guess_ps,
                    lambda dx, e, x0: dx/(diag-e+(1e-5)),
                    nroots=args.k,
                    max_cycle=args.iterations,
                    verbose=args.verbosity,
                    max_space=args.subspace,
                    max_memory=get_davidson_mem(0.75),
                    tol=1e-12,
                )

            print("Davidson:", e_ps_approx)
            print(conv)
            # Building electronic observables for PS
            EPS[i, j] = e_ps_approx[0]
            pe_r = xp.sum(evecs_save[0].conj()*apply_pr(H,evecs_save[0])) # < 0 | p_e | 0 > for PS
            psi0 = evecs_save[0].reshape(H.boshape)
            pe_x = xp.einsum('xy, xa, ay ->', psi0.conj(), (-1j)*H.ddx, psi0, optimize=True) # < 0 | p_x | 0 > for PS
            pe_y = xp.einsum('xy, yb, xb ->', psi0.conj(), (-1j)*H.ddy, psi0, optimize=True) # < 0 | p_y | 0 > for PS
            pPS[:,i,j] = xp.asarray([pe_x,pe_y,pe_r])

            print("<pe> on gs:", pe_x.real, pe_y.real)
            Gamma_x = xp.einsum('xy,xay,ay->',psi0.conj(), 1j*gammaetfx, psi0)
            Gamma_y = xp.einsum('xy,xyb,xb->',psi0.conj(), 1j*gammaetfy, psi0)
            print("<Gamma> on gs:", Gamma_x.real, Gamma_y.real)
            GammaPS[:,i,j] = xp.asarray([Gamma_x, Gamma_y])
            psi0_lz = apply_l(H,evecs_save[0])
            l00z = xp.sum(psi0.conj()*psi0_lz)
            l200_z = xp.sum(psi0_lz.conj()*psi0_lz)
            print("<l> on gs:", l00z)
            print("<l^2> on gs", l200_z)
            lPS[i,j] = l00z
            l2PS[i,j] = l200_z
            psi0_p2 = (-2*H.mur)*Tx(H,psi0)
            px2 = xp.einsum('xy, xa, ay ->', psi0.conj(), H.ddx2, psi0)
            py2 = xp.einsum('xy, yb, xb ->', psi0.conj(),H.ddy2, psi0)
            p2PS[i,j] = px2 + py2 
            check = xp.sum(psi0.conj()*psi0_p2)
            print("check diff:", check-px2-py2)
            print("p2PS:", px2.real, py2.real, p2PS[i,j])
                    
            rPS[i,j] = xp.einsum('xy,xy,xy->', psi0.conj(), H.r, psi0, optimize=True)
            print("<r> on gs:", rPS[i,j].real)

            print() # add a new line between each RP point

    # Solving for BO energies and observables
    Hbo_new = -1/(2*H.mu12)*(H.ddR2 - xp.diag(H.J**2/H.R**2)) +xp.diag(Ad_n)
    Ad_vn_new, Unv_bo = xp.linalg.eigh((Hbo_new))
    e_bo_new = xp.sort(Ad_vn_new.flatten())
    bo_new = e_bo_new[1] - e_bo_new[0]
    print("e_bo_new",e_bo_new[0:10])
    print("BO new vib gap",bo_new,flush=True)

    # Building BO observables
    R_bo = xp.sum(Unv_bo[:,0].conj()*H.R*Unv_bo[:,0]).real
    print("R00 BO: <chi_0| R| chi_0 >:", R_bo)

    rBO_RP = xp.zeros(rPS.shape, dtype=xp.complex128)
    rBO_RP = rBO[:,None]
    HrBO = inverse_weyl_transform(rBO_RP, H.shape[0], H.R, H.P_R)
    rBO_chi00 = xp.sum(Unv_bo[:,0].conj()*(HrBO@Unv_bo[:,0]))
    rBO_chi01 = xp.sum(Unv_bo[:,1].conj()*(HrBO@Unv_bo[:,0]))
    print("r00 BO: <chi0|r|chi0>:", rBO_chi00, rBO_chi01)

    HP_R = inverse_weyl_transform(Pval, H.shape[0], H.R, H.P_R)
    PBO_chi = xp.sum(Unv_bo[:,1].conj()*(HP_R@Unv_bo[:,0]))
    print("P01 BO <chi1|P|chi0>:", PBO_chi)

    l2BO_RP = xp.zeros(rPS.shape, dtype=xp.complex128)
    l2BO_RP = l2BO[:,None]
    Hl2BO = inverse_weyl_transform(l2BO_RP, H.shape[0], H.R, H.P_R)
    l2BO_chi00 = xp.sum(Unv_bo[:,0].conj()*(Hl2BO@Unv_bo[:,0]))
    l2BO_chi01 = xp.sum(Unv_bo[:,1].conj()*(Hl2BO@Unv_bo[:,0]))
    print("l200 BO: <chi0|l^2|chi0>:", l2BO_chi00)
    print("l201 BO: <chi1|l^2|chi0>:", l2BO_chi01)

    p2BO_RP = xp.zeros(rPS.shape, dtype=xp.complex128)
    p2BO_RP = p2BO[:,None]
    Hp2BO = inverse_weyl_transform(p2BO_RP, H.shape[0], H.R, H.P_R)
    p2BO_chi00 = xp.sum(Unv_bo[:,0].conj()*(Hp2BO@Unv_bo[:,0]))
    p2BO_chi01 = xp.sum(Unv_bo[:,1].conj()*(Hp2BO@Unv_bo[:,0]))
    print("p200 BO: <chi0|p^2|chi0>:", p2BO_chi00)
    print("p201 BO: <chi1|p^2|chi0>:", p2BO_chi01)

    #EPS_bo = xp.zeros((H.shape[0], H.shape[0]))
    #Helmat = xp.repeat(ival,H.shape[0],axis=1)
    #EPS_bo += Helmat   
    #EPS_bo += 1/(2*H.mu12)*(Pval**2-(1/4/Rval**2)+H.J**2/Rval**2)
    #HPS_bo = inverse_weyl_transform(EPS_bo, H.shape[0], H.R, H.P_R)
    #EPSv_bo = batch_eigvalsh(HPS_bo)
    #print("e_bo_new Weyl",EPSv_bo[0:10])
    #print("Weyl BO vib gap",EPSv_bo[1]-EPSv_bo[0],flush=True)

    # PS energies and observables
    EPS += 1/(2*H.mu12)*(Pval**2-(1/4/Rval**2)+H.J**2/Rval**2)
    HPS = inverse_weyl_transform((EPS), H.shape[0], H.R, H.P_R)
    EPSv = batch_eigvalsh(HPS)
    EPSv, UPSv = xp.linalg.eigh(HPS)
    print("PSWeyl",EPSv[0:10])
    print("PS vib gap",EPSv[1]-EPSv[0],flush=True)


    Hpe_x = inverse_weyl_transform(pPS[0], H.shape[0], H.R, H.P_R)
    Hpe_y = inverse_weyl_transform(pPS[1], H.shape[0], H.R, H.P_R)
    Hpe_r = inverse_weyl_transform(pPS[2], H.shape[0], H.R, H.P_R)

    pe_chix = xp.sum(UPSv[:,1].conj()*(Hpe_x@UPSv[:,0]))
    pe_chiy = xp.sum(UPSv[:,1].conj()*(Hpe_y@UPSv[:,0]))
    pe_chir = xp.sum(UPSv[:,1].conj()*(Hpe_r@UPSv[:,0]))
    print("pe01 <chi_1|pe|chi0>:", pe_chix, pe_chiy, pe_chir)

    pe_chix = UPSv[:,0].conj().T@Hpe_x@UPSv[:,0]
    pe_chiy = UPSv[:,0].conj().T@Hpe_y@UPSv[:,0]
    pe_chir = UPSv[:,0].conj().T@Hpe_r@UPSv[:,0]
    print("pe00 <chi_0|pe|chi0>:",   pe_chiy, pe_chir)

    R_ps = xp.sum(UPSv[:,0].conj()*H.R*UPSv[:,0]).real
    print("R00 PS: <chi_0| R |chi_0 >:", R_ps)

    Hrps = inverse_weyl_transform(rPS, H.shape[0], H.R, H.P_R)
    r00_ps = xp.sum(UPSv[:,0].conj()*(Hrps@UPSv[:,0]))
    r01_ps = xp.sum(UPSv[:,1].conj()*(Hrps@UPSv[:,0]))
    print("r00 PS <chi0|r|chi0>:", r00_ps)
    print("r01 PS <chi1|r|chi0>:", r01_ps)

    HGamma_x = inverse_weyl_transform(GammaPS[0], H.shape[0], H.R, H.P_R)
    Gamma_x_ps_01 = xp.sum(UPSv[:,1].conj()*(HGamma_x@UPSv[:,0]))
    print("G01 PS: <chi_1| Gamma_x |chi0>:", Gamma_x_ps_01)
     
    PPS_chi = xp.sum(UPSv[:,1].conj()*(HP_R@UPSv[:,0]))
    print("P01 PS <chi1|P_R|chi0>:", PPS_chi)
    PG_chi = xp.sum(UPSv[:,1].conj()*((HP_R-HGamma_x)@UPSv[:,0]))
    print("PG01 PS <chi1|P_R - Gamma_x|chi0>:", PG_chi)

     
    Hlzps = inverse_weyl_transform(lPS, H.shape[0], H.R, H.P_R)
    Hl2ps = inverse_weyl_transform(l2PS, H.shape[0], H.R, H.P_R)

    l00z = xp.sum(UPSv[:,0].conj()*(Hlzps@UPSv[:,0]))
    l01z = xp.sum(UPSv[:,1].conj()*(Hlzps@UPSv[:,0]))
    l002 = xp.sum(UPSv[:,0].conj()*(Hl2ps@UPSv[:,0]))
    l012 = xp.sum(UPSv[:,1].conj()*(Hl2ps@UPSv[:,0]))

    print("l00 <chi_0 | l_z|chi_0>:", l00z)
    print("l01 < chi1| l_z|chi_0>", l01z)
    print("l200  <chi_0 | l^2 |chi_0>:", l002)
    print("l200  <chi_1 | l^2 |chi_0>:", l012)

    Hp2PS = inverse_weyl_transform(p2PS, H.shape[0], H.R, H.P_R)
    p2PS_chi00 = xp.sum(UPSv[:,0].conj()*(Hp2PS@UPSv[:,0]))
    p2PS_chi01 = xp.sum(UPSv[:,1].conj()*(Hp2PS@UPSv[:,0]))
    print("p200 PS: <chi0|p^2|chi0>:", p2PS_chi00)
    print("p201 PS: <chi1|p^2|chi0>:", p2PS_chi01)




    if args.evecs:
        numpy.savez(args.evecs, R=H.R, P=H.P_R, EPS=EPS, HPS=HPS,EPSv=EPSv, UPSv=UPSv, pPS=pPS, rPS=rPS)
        
        
        
        
        
        
        