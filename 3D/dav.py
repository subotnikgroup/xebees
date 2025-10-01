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
from hamiltonian import  KE, KE_FFT, KE_Borisov_3D, inverse_weyl_transform
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
        'R', 'P_R', 'R_grid', 'r', 'ph', 'r_grid','p_grid', 'th', 'pg', 't_grid','RP_grid', 'ddph2', 'ddph1',
        'ddR2', 'ddr2', 'ddth2', 'ddth1', 'ddr1','axes','Vgrid','rb_grid','tb_grid','pb_grid',
        'Rinv2', 'rinv2', 'diag', '_preconditioner_data','Pg','Pphi','Ptheta',
        'shape', 'boshape','size','guess','k','mu12','_Vfunc',
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
        self.Pphi = args.Pphi
        self.Ptheta = args.Ptheta

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
            'erf_coulomb':(potentials.erf_coulomb, potentials.extents_erf_coulomb),
            'borgis': (potentials.borgis, potentials.extents_borgis)
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
        self.th = xp.linspace(xp.pi/(2*(args.Nth+1)), xp.pi-xp.pi/(2*(args.Nth+1)), args.Nth, endpoint=True)
        self.ph = xp.linspace(0, 2*xp.pi, args.Nph, endpoint=False)##XXXXX  check this
        #self.ph = xp.linspace(xp.pi/args.Nph, 2*xp.pi+xp.pi/args.Nph, args.Nph, endpoint=False)
        self.axes = (self.R, self.r, self.ph, self.th)

        self.shape = (args.NR, args.Nr, args.Nph, args.Nth)
        self.boshape = (args.Nr, args.Nph, args.Nth)
        self.size = args.NR * args.Nr * args.Nph * args.Nth

        dR = self.R[1] - self.R[0]
        dr = self.r[1] - self.r[0]
        dth = self.th[1] - self.th[0]
        dph = self.ph[1] - self.ph[0] 

        self.P_R  = xp.fft.fftshift(xp.fft.fftfreq(args.NR, dR)) * 2 * xp.pi
        self.RP_grid = xp.meshgrid(self.R, self.P_R, indexing='ij')
        # N.B.: These all lack the factor of -1/(2 * mu)
        # We also are throwing away the returned jacobian of R/r
        #self.ddR2, _ = KE_Borisov(self.R, bare=True)
        self.ddR2    = KE(args.NR, dR, bare=True, cyclic=False)
        #self.ddr2, _ = KE_Borisov_3D(self.r, bare=True)
        #self.ddr1, _ = KE_Borisov_3D(self.r, bare=True, order=1)
        self.ddr2 = KE(args.Nr, dr, bare=True, cyclic=False)
        self.ddr1 = KE(args.Nr, dr, bare=True, cyclic=False, order=1)

        # Part of the reason for using a cyclic *stencil* for gamma
        # rather than KE_FFT is that it wasn't immediately obvious how
        # I would represent ∂/∂γ. (∂²/∂γ² was clear.)  N.B.: The
        # default stencil degree is 11
        self.ddth2 = KE(args.Nth, dth, bare=True, cyclic=True)
        self.ddth1 = KE(args.Nth, dth, bare=True, cyclic=True, order=1)

        self.ddph2 = KE(args.Nph, dph, bare=True, cyclic=True)
        self.ddph1 = KE(args.Nph, dph, bare=True, cyclic=True, order=1)
        
        self.R_grid, self.rb_grid, self.pb_grid, self.tb_grid = xp.meshgrid(self.R, self.r, self.ph, self.th, indexing='ij')
        self.r_grid, self.p_grid, self.t_grid,  = xp.meshgrid(self.r, self.ph, self.th, indexing='ij')
        self.Vgrid = self.V(self.R_grid, self.rb_grid, self.pb_grid, self.tb_grid)

        #print((self.Vgrid).tolist())
        #exit()
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

    def V(self, R, r, phi, theta, spitvals=False):

        mu12 = self.mu12
        M_1 = self.M_1
        M_2 = self.M_2

        kappa2 = r*R*xp.sin(theta)*xp.cos(phi)

        r1e2 = (r)**2 + (R)**2*(mu12/M_1)**2 - 2*kappa2*mu12/M_1
        r2e2 = (r)**2 + (R)**2*(mu12/M_2)**2 + 2*kappa2*mu12/M_2

        r1e = xp.sqrt(xp.where(r1e2 < 0, 0, r1e2))
        r2e = xp.sqrt(xp.where(r2e2 < 0, 0, r2e2))
        
        if spitvals == True:
            return r1e2,r2e2
        else:
            return self._Vfunc(R, r1e, r2e, (self.g_1, self.g_2))

    
    #def build_Hel(self, Ridx=None):
    #    NR, Nx, Ny = self.shape
    #    Nelec = Nx*Ny
    #    Hel = xp.empty((NR, Nelec, Nelec), dtype=self.dtype)
    #    Hel[:] = -1/(2*self.mur)*(xp.kron(self.ddx2,xp.eye(Ny)) + xp.kron(xp.eye(Nx), self.ddy2))
#
    #    if Ridx is None:
    #        Ridx = xp.arange(NR)
    #    else:
    #        Ridx = xp.atleast_1d(Ridx)
    #        NR,  = Ridx.shape
#
    #    Hel[:, xp.arange(Nelec), xp.arange(Nelec)] +=(  # extract diagonal at every R
    #        xp.reshape(self.Vgrid[Ridx], (NR, Nelec))   # + V
    #    )
#
    #    return xp.squeeze(Hel)

    
def Gamma_etf_polar(R,r,phi,theta,ddr,ddph,ddth,M_1,M_2,mu12,r1e2,r2e2):

    Nth = len(ddth)
    Nr = len(ddr)
    Nph = len(ddph)
    
    theta1 = xp.exp(-r1e2)
    theta2 = xp.exp(-r2e2)
    partition = theta1 + theta2

    costheta = xp.cos(theta)[0,0,:]
    sintheta = xp.sin(theta)[0,0,:]
    cosphi = xp.cos(phi)[0,:,0]
    sinphi = xp.sin(phi)[0,:,0]

    re = r[:,0,0]
    invr = 1/re

    t1 = xp.diag((theta1/partition).ravel())
    t2 = xp.diag((theta2/partition).ravel())

    #xp.fill_diagonal(spg, xp.diag(spg) * singamma[0,:])
    #xp.fill_diagonal(cpg, xp.diag(cpg) * cosgamma[0,:])

      #xp.kron(xp.kron(ddr, xp.diag(cosphi)), xp.diag(sintheta))\
          #+xp.kron(xp.kron(xp.diag(invr), xp.diag(cosphi)), xp.diag(costheta)*ddth)\
    px_old =      -xp.kron(xp.kron(xp.diag(invr), ddph), xp.diag(1/sintheta))

    xdav = xp.random.rand(H.shape[1],H.shape[2],H.shape[3])

     #xp.einsum('ij,kl,mn,jln->ikm',ddr,xp.diag(cosphi),xp.diag(sintheta),xdav)\
        #+ xp.einsum('i,j,kl,ijl->ijk',invr,cosphi,xp.diag(costheta)*ddth,xdav)
    px = xp.einsum('i,kl,j,ilj->ikj',invr,xp.diag(sinphi)*ddph,1/sintheta,xdav)
    orig = px_old@xdav.flatten()
    #print("orig",orig)
    
    py =  xp.kron(xp.kron(ddr, xp.diag(singamma)), xp.diag(sinpsi)) +\
          xp.kron(xp.kron(xp.diag(invr), xp.diag(singamma)*ddg), xp.diag(sinpsi)) +\
          xp.kron(xp.kron(xp.diag(invr), xp.diag(singamma)), xp.diag(cospsi)*ddp)

    pz =  xp.kron(xp.kron(ddr, xp.diag(cosgamma)), xp.eye(Np)) -\
          xp.kron(xp.kron(xp.diag(invr), singamma*ddg), xp.eye(Np)) 

    t1px = xp.dot(t1,px)
    pxt1 = xp.dot(px,t1)
    t2px = xp.dot(t2,px)
    pxt2 = xp.dot(px,t2)

    t1py = xp.dot(t1,py)
    pyt1 = xp.dot(py,t1)
    t2py = xp.dot(t2,py)
    pyt2 = xp.dot(py,t2)

    t1pz = xp.dot(t1,pz)
    pzt1 = xp.dot(pz,t1)
    t2pz = xp.dot(t2,pz)
    pzt2 = xp.dot(pz,t2)

    gammaetf1x = -0.5*(t1px + pxt1)
    gammaetf1y = -0.5*(t1py + pyt1)
    gammaetf1z = -0.5*(t1pz + pzt1)

    gammaetf2x = -0.5*(t2px + pxt2)   
    gammaetf2y = -0.5*(t2py + pyt2)
    gammaetf2z = -0.5*(t2pz + pzt2)

    return gammaetf1x, gammaetf1y, gammaetf1z, gammaetf2x, gammaetf2y, gammaetf2z

def Gamma_erf_polar(R,r,p,g,ddr,ddp,ddg,M_1,M_2,mu12,r1e2,r2e2):

    Ng = len(ddg)
    Nr = len(ddr)
    Np = len(ddp)
    
    theta1 = xp.exp(-r1e2)
    theta2 = xp.exp(-r2e2)
    partition = theta1 + theta2

    cosgamma = xp.cos(g+xp.pi/2)[0,0,:]
    singamma = xp.sin(g+xp.pi/2)[0,0,:]
    cospsi = xp.cos(p)[0,:,0]
    sinpsi = xp.sin(p)[0,:,0]

    re = r[:,0,0]
    invr = 1/re

    t1 = xp.diag((theta1/partition).ravel())
    t2 = xp.diag((theta2/partition).ravel())

    sincosgamma = singamma*cosgamma

    Jxa =  xp.kron(xp.kron(xp.diag(re)*ddr, xp.diag(sincosgamma)), xp.diag(sinpsi)) -\
           xp.kron(xp.kron(xp.eye(Nr), singamma*ddg), xp.diag(sinpsi))

    Jxb =  xp.kron(xp.kron(xp.diag(re)*ddr, xp.diag(sincosgamma)), xp.diag(sinpsi)) +\
           xp.kron(xp.kron(xp.eye(Nr), xp.diag(sincosgamma)*ddg), xp.diag(sinpsi)) +\
           xp.kron(xp.kron(xp.eye(Nr), xp.diag(sincosgamma)), xp.diag(cospsi)*ddp)

    Jya =  xp.kron(xp.kron(xp.diag(re)*ddr, xp.diag(sincosgamma)), xp.diag(cospsi)) +\
           xp.kron(xp.kron(xp.eye(Nr), xp.diag(sincosgamma)*ddg), xp.diag(cospsi)) +\
           xp.kron(xp.kron(xp.eye(Nr), xp.diag(sincosgamma)), xp.diag(sinpsi)*ddp)

    Jyb = xp.kron(xp.kron(xp.diag(re)*ddr, xp.diag(sincosgamma)), xp.diag(cosgamma)) -\
          xp.kron(xp.kron(xp.eye(Nr), singamma*singamma*ddg), xp.diag(cosgamma))

    Jyc = xp.kron(xp.kron(ddr, xp.diag(cosgamma)), xp.eye(Np)) -\
          xp.kron(xp.kron(xp.diag(invr), singamma*ddg), xp.eye(Np))  

    Jx = Jxa-Jxb
    Jym1 = Jya-Jyb+(Jyc*R*mu12/M_1*xp.kron(xp.kron(xp.eye(Nr),xp.eye(Ng)),xp.eye(Np)))
    Jym2 = Jya-Jyb-(Jyc*R*mu12/M_2*xp.kron(xp.kron(xp.eye(Nr),xp.eye(Ng)),xp.eye(Np)))
    
    J1x = -0.5*(xp.dot(t1,Jx) + xp.dot(Jx,t1))
    J2x = -0.5*(xp.dot(t2,Jx) + xp.dot(Jx,t2))

    J1y = -0.5*(xp.dot(t1,Jym1) + xp.dot(Jym1,t1))
    J2y = -0.5*(xp.dot(t2,Jym2) + xp.dot(Jym2,t2))

    gammaerf1y = -1/R*(-J1y-J2y)
    gammaerf1z = -1/R*(J1x+J2x)

    gammaerf2y = 1/R*(-J1y-J2y)
    gammaerf2z = 1/R*(J1x+J2x)
    
    return gammaerf1y, gammaerf1z, gammaerf2y, gammaerf2z


def compute_EPS(info):

    Rval, Pval, Htot_bo, gammacoeff_R, gammacoeff_phi, gammacoeff_theta, \
    Gammatotr, Gammatotp, Gammatott, Gammasqtotr, Gammasqtotp, Gammasqtott, mu12 = info
    
    #print("i,j",Rval,Pval,gammacoeff_R[Rval,Pval],flush=True)           
    
    Htot = Htot_bo[Rval]+(gammacoeff_R[Rval]*Gammatotr)+(gammacoeff_phi[Rval]*Gammatotp)+(gammacoeff_theta[Rval]*Gammatott)
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
    parser.add_argument('-theta', dest="Nth", metavar="Nth", default=250, type=int)
    parser.add_argument('-phi', dest="Nph", metavar="Nph", default=250, type=int)
    parser.add_argument('--verbosity', default=2, type=int)
    parser.add_argument('--iterations', metavar='max_iterations', default=10000, type=int)
    parser.add_argument('--subspace', metavar='max_subspace', default=200, type=int)
    parser.add_argument('--guess', metavar="guess.npz", type=Path, default=None)
    parser.add_argument('--evecs', metavar="guess.npz", type=Path, default=None)
    parser.add_argument('--save', metavar="filename")
    parser.add_argument('--potential', choices=['erf_coulomb', 'borgis'],
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
    
    NR,Nr,Nph,Nth = H.shape
    Nelec = Nr*Nth*Nph 
    
    
    
    #testdav = xp.einsum('ij,kl,mn,jln->ikm',xp.diag(H.rinv2), xp.diag(xp.sin(H.g+xp.pi/2)**2),H.ddp2,xdav)
    #testnorm = xp.kron(xp.kron(xp.diag(H.rinv2), xp.diag(xp.sin(H.g+xp.pi/2)**2)),H.ddp2)@xdot


    #(xdav.reshape((-1,) + H.shape) * H.Vgrid).reshape(x.shape)

    #xbo = xp.random.rand(NR,Nr,Nph,Nth)
    #check = xp.repeat(xdav,NR,axis=1)
    #print(check.shape)
    #print("shape",xbo[-2].shape)
    #print("PE",xdav.repeat(3))
    #print("KE",H.)
    

    #print("Htotbo",xdav.reshape((-1,) + H.shape) )
    #print("Htotbo",(xdav.reshape((-1,) + H.shape) * H.Vgrid).reshape(x.shape))
    
    #print("Htotbo",H.Vgrid[1])

    #print("testdav",testdav)
    #print("testnorm",testnorm)
    #print("diff",xp.sum(xp.abs(testdav.flatten()-testnorm)))
    sinthiinv = 1/xp.sin(H.th)
    sinthiinv[xp.abs(sinthiinv)>1e6]=0
    sinthiinvsq = xp.square(sinthiinv)

    #ddth1_new = H.ddth1.copy()
    #xp.fill_diagonal(ddth1_new, xp.diag(H.ddth1) * xp.sin(H.th))
    
    thbig = xp.diag(sinthiinv)@(H.ddth1@(xp.diag(xp.sin(H.th))@H.ddth1))

    Hel = -1/(2*H.mur)*(
        xp.kron(xp.kron(H.ddr2, xp.eye(Nph)), xp.eye(Nth)) \
        +xp.kron(xp.kron(xp.diag(H.rinv2),xp.eye(Nph)),thbig)\
        +xp.kron(xp.kron(xp.diag(H.rinv2), H.ddph2),xp.diag(sinthiinvsq))
        )

    Hel_old = -1/(2*H.mur)*(
       xp.kron(xp.kron(H.ddr2, xp.eye(Nph)), xp.eye(Nth)) \
       +xp.kron(xp.kron(xp.eye(Nr), xp.eye(Nph) ), H.ddth1)\
       +xp.kron(xp.kron(xp.diag(H.rinv2), xp.eye(Nph)), H.ddth2)\
       +xp.kron(xp.kron(xp.diag(H.rinv2), H.ddph2),xp.diag(sinthiinvsq))
       )
    
    

    def Tx(xdav):

        #sinthiinv = 1/xp.sin(H.th)
        ##sinphiinv[xp.isinf(sinphiinv)]=0
        #sinthiinv[xp.abs(sinthiinv)>1e6]=0
        ##sinphiinv[xp.abs(sinphiinv)>1/xp.finfo(xp.float64).eps/10]=0
        #sinthiinvsq = xp.square(sinthiinv)
        #xdav = xdav.reshape((-1,) + H.boshape)
        #Hel_dav = -1/(2*H.mur)*(
        #    xp.einsum('ij,Bjkl->Bikl',H.ddr2,xdav)\
        #    #+xp.einsum('ij,Bjkl->Bikl',H.ddr1,xdav)\
        #    #+xp.einsum('ij,kl,Bjlm->Bikm',xp.diag(H.rinv2),xp.diag(xp.cos(H.ph)*sinphiinv)*H.ddph1,xdav)\
        #    +xp.einsum('ij,kl,Bjlm->Bikm',xp.diag(H.rinv2),H.ddth1,xdav)\
        #    +xp.einsum('ij,kl,Bjlm->Bikm',xp.diag(H.rinv2), H.ddth2,xdav)\
        #    +xp.einsum('ij,kl,mn,Bjln->Bikm',xp.diag(H.rinv2), xp.diag(sinthiinvsq),H.ddph2,xdav)
        #    )
        ##print("Heldav",type(Hel_dav))
        ##exit()

        sinthiinv = 1/xp.sin(H.th)
        sinthiinv[xp.abs(sinthiinv)>1e6]=0       
        sinthiinvsq = xp.square(sinthiinv)
        xdav = xdav.reshape((-1,) + H.boshape)
        
        #Hel_dav_old = -1/(2*H.mur)*(           
        #    xp.einsum('ij,Bjkl->Bikl',H.ddr2,xdav)\
        #    +xp.einsum('ij,Bklj->Bkli',H.ddth1,xdav)\
        #    +xp.einsum('i,jk,Bilk->Bilj',H.rinv2, H.ddth2,xdav)\
        #    +xp.einsum('i,j,kl,Bilj->Bikj',H.rinv2, sinthiinvsq, H.ddph2,xdav)
        #    )

        Hel_dav = -1/(2*H.mur)*(           
            xp.einsum('ij,Bjkl->Bikl',H.ddr2,xdav)\
            +xp.einsum('i,jk,Bilk->Bilj',H.rinv2,thbig,xdav)\
            +xp.einsum('i,j,kl,Bilj->Bikj',H.rinv2, sinthiinvsq, H.ddph2,xdav)
            )

        return Hel_dav.reshape(xdav.shape)

   
    #orig = Hel@xdav.flatten()
    #print("orig",orig.shape)
    #print("orig",orig)
    #new = Tx(xdav)
    #print("new",new)
    #print("diff",xp.linalg.norm(orig-new.flatten()))


    xdav = xp.random.rand(H.shape[1],H.shape[2],H.shape[3])
    xdot = xdav.flatten()
 
    
    def _preconditioner_naive(H, dx, e, x0, Ri):
        diagH = buildDiag(H,Ri)
        diagd = diagH - (e - 1e-5)
        return dx/diagd

    def buildDiag(H,Ri):
        #rinv2 = 1/(H.r_grid)**2
        #tanthinv = 1/xp.tan(H.t_grid)
        #tanthinv[xp.abs(tanthinv)>1e6]=0
        #sinsqthinv = 1/xp.sin(H.t_grid)**2
        #sinsqthinv[xp.abs(sinsqthinv)>1e6]=0
#
        #ke  = xp.zeros([Nr,Nph,Nth])
        #ke += xp.diag(H.ddr2)[:,None,None]
        ##ke += xp.diag(H.ddr1)[:,None,None]
        #ke += rinv2*tanthinv*xp.diag(H.ddth1)[None,:,None]
        #ke += rinv2*xp.diag(H.ddth2)[None,:,None]
        #ke += rinv2*sinsqthinv*xp.diag(H.ddph2)[None,None,:]
        #ke *= -1 / (2*H.mur)
        #diag = H.Vgrid[Ri] + ke

        rinv2 = 1/(H.r_grid)**2
        tanthinv = 1/xp.tan(H.t_grid)
        tanthinv[xp.abs(tanthinv)>1e6]=0
        sinsqthinv = 1/xp.sin(H.t_grid)**2
        sinsqthinv[xp.abs(sinsqthinv)>1e6]=0
        sinthinv = 1/xp.sin(H.t_grid)
        sinthinv[xp.abs(sinthinv)>1e6]=0
        print("tgrid",(H.t_grid).shape)
       

        ke  = xp.zeros([Nr,Nph,Nth])
        ke += xp.diag(H.ddr2)[:,None,None]
        #ke += rinv2*tanthinv*xp.diag(H.ddth1)[None,None,:]
        #ke += xp.diag(H.ddth1)[None,None,:]
        ke += rinv2*xp.diag(thbig)[None,None,:]
        #ke += rinv2*xp.diag(H.ddth2)[None,None,:]
        ke += rinv2*sinsqthinv*xp.diag(H.ddph2)[None,:,None]
        ke *= -1 / (2*H.mur)
        diag = H.Vgrid[Ri] + ke
        return diag.ravel()

    Htot_bo = xp.zeros([NR,Nelec,Nelec])
    Htot_bo[:] = Hel
    Htot_bo[:,xp.arange(Nelec),xp.arange(Nelec)] += xp.reshape(H.Vgrid[:],(NR,Nelec))#XXXXXcheck this
    Htot_bo_old = xp.zeros([NR,Nelec,Nelec])
    Htot_bo_old[:] = Hel_old
    Htot_bo_old[:,xp.arange(Nelec),xp.arange(Nelec)] += xp.reshape(H.Vgrid[:],(NR,Nelec)) 
    ival = xp.zeros([NR,1])
    Ad_n = xp.zeros(NR)

    
    eigvals_list = []
    for i in range(NR):
        print("Atom Ri",i)
        diag = buildDiag(H,i)
        def Hbo_dav(xdav):
            x = xdav.reshape((-1,)+H.boshape)
            Hbodav = H.Vgrid[i]*x + Tx(x)
            return Hbodav.reshape(xdav.shape)
        
        xdav = xp.random.rand(H.shape[1],H.shape[2],H.shape[3])

        orig = Htot_bo[i]@xdav.flatten()     
        #print("orig",orig.shape)
        #print("orig",orig)
        new = Hbo_dav(xdav)
        #print("new",new)
        print("diff1",xp.linalg.norm(orig-new.flatten()))
#
        origdiag = xp.diag(Htot_bo[i]) 
        newdiag = buildDiag(H,i)

        #print("new",newdiag)
        print("diff2",xp.linalg.norm(origdiag-newdiag))

        print("check",xp.sum((Htot_bo[i]-xp.conj(Htot_bo[i].T))))



        eigvals = xp.linalg.eigvalsh(Htot_bo[i])
        eigvals_old = xp.linalg.eigvalsh(Htot_bo_old[i])
        print("eigvals",eigvals[0:5]) 
        print("eigval_old",eigvals_old[0:5])
        
        #eigvals_list.append(eigvals[0:5].tolist())
        ##
        guess_bo = xp.exp(-(H.Vgrid[i] - xp.min(H.Vgrid[i]))**2/27.211**2).ravel()#
        with timer_ctx(f"Davidson of size {H.size}"):
            conv, e_approx, evecs = lib.davidson1(
                Hbo_dav,
                guess_bo,
                #H.diag,
                #_preconditioner_naive(H, dx, e, x0,i),
                lambda dx, e, x0: dx/(diag-e+1e-5),
                nroots=args.k,
                max_cycle=args.iterations,
                verbose=args.verbosity,
                max_space=args.subspace,
                max_memory=get_davidson_mem(0.75),
                #tol=1e-12, #FIXME:DEBUG
                tol=1e-10,
            )
        print("Davidson:", e_approx)
        print(conv)#
        #Ad_n[i] = e_approx[0]
        #ival[i,0] = e_approx[0]
        #eigvals = xp.linalg.eigvalsh(Htot_bo[i])
        #print("eigvals",eigvals) 
        #print("diff",e_approx[0]-eigvals[0])
        #exit()

    print("eigval",eigvals_list)

    Rval, Pval = H.RP_grid

    Hbo_new = -1/(2*H.mu12)*(H.ddR2 - xp.diag(H.Pphi**2/H.R**2)- xp.diag(H.Ptheta**2/H.R**2)) +xp.diag(Ad_n)
    Ad_vn_new = batch_eigvalsh(Hbo_new)
    e_bo_new = xp.sort(Ad_vn_new.flatten())
    bo_new = e_bo_new[1] - e_bo_new[0]
    print("BO new vib gap",bo_new,flush=True)

    EPS_bo = xp.zeros((H.shape[0], H.shape[0]))
    Helmat = xp.repeat(ival,H.shape[0],axis=1)
    EPS_bo += Helmat   
    EPS_bo += 1/(2*H.mu12)*(Pval**2+H.Pphi**2/Rval**2+H.Ptheta**2/Rval**2)
    HPS_bo = inverse_weyl_transform(EPS_bo, H.shape[0], H.R, H.P_R)
    EPSv_bo = batch_eigvalsh(HPS_bo)
    print("Weyl BO vib gap",EPSv_bo[1]-EPSv_bo[0],flush=True)

    exit()
    
    EPS = xp.zeros((H.shape[0], H.shape[0]))
    EPSsq = xp.zeros((H.shape[0], H.shape[0]))

    gammacoeff_R = -1j*H.P_R/H.mu12 
    gammacoeff_phi = -1j*(H.Pphi/H.R)/H.mu12
    gammacoeff_theta = +1j*(H.Ptheta/H.R)/H.mu12

    Gammasqtotr = xp.zeros([Nelec,Nelec,Nelec],dtype=complex)
    Gammasqtott = xp.zeros([Nelec,Nelec,Nelec],dtype=complex)
    Gammatotr = xp.zeros([Nelec,Nelec,Nelec],dtype=complex)
    Gammatott = xp.zeros([Nelec,Nelec,Nelec],dtype=complex)



    for i in range(H.shape[0]):
        print("i",i,flush=True)

        r1e2, r2e2 = H.V(H.R[i], H.r_grid, H.p_grid, H.t_grid, spitvals=True)
        with timer_ctx("build gamma"):
            gammaetf1r, gammaetf1p, gammaetf1t, gammaetf2r, gammaetf2p, gammaetf2t = Gamma_etf_polar(H.R[i],H.r_grid,H.p_grid,H.t_grid,H.ddr1,H.ddph1,H.ddth1,H.M_1,H.M_2,H.mu12,r1e2,r2e2)
            exit()
            gammaerf1p, gammaerf1t, gammaerf2p, gammaerf2t = Gamma_erf_polar(H.R[i],H.r_grid,H.ph_grid,H.th_grid,H.ddr1,H.ddph1,H.ddth1,H.M_1,H.M_2,H.mu12,r1e2,r2e2)

        gamma1r = gammaetf1r
        gamma2r = gammaetf2r
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

    Rval, Pval = H.RP_grid

    Hbo_new = -1/(2*H.mu12)*(H.ddR2 - xp.diag(H.Pphi**2/H.R**2)- xp.diag(H.Ptheta**2/H.R**2)) +xp.diag(Ad_n)
    Ad_vn_new = batch_eigvalsh(Hbo_new)
    e_bo_new = xp.sort(Ad_vn_new.flatten())
    bo_new = e_bo_new[1] - e_bo_new[0]
    print("BO new vib gap",bo_new,flush=True)
        
    EPS += 1/(2*H.mu12)*(Pval**2+H.Pphi**2/Rval**2+H.Ptheta**2/Rval**2)
    HPS = inverse_weyl_transform(EPS, H.shape[0], H.R, H.P_R)
    EPSv = batch_eigvalsh(HPS)
    print("PS vib gap",EPSv[1]-EPSv[0],flush=True)

    EPSsq += 1/(2*H.mu12)*(Pval**2+H.Pphi**2/Rval**2+H.Ptheta**2/Rval**2)
    HPSsq = inverse_weyl_transform(EPSsq, H.shape[0], H.R, H.P_R)
    EPSvsq = batch_eigvalsh(HPSsq)
    print("PS vib gap sq",EPSvsq[1]-EPSvsq[0],flush=True)

    EPS_bo = xp.zeros((H.shape[0], H.shape[0]))
    Helmat = xp.repeat(ival,H.shape[0],axis=1)
    EPS_bo += Helmat   
    EPS_bo += 1/(2*H.mu12)*(Pval**2+H.Pphi**2/Rval**2+H.Ptheta**2/Rval**2)
    HPS_bo = inverse_weyl_transform(EPS_bo, H.shape[0], H.R, H.P_R)
    EPSv_bo = batch_eigvalsh(HPS_bo)
    print("Weyl BO vib gap",EPSv_bo[1]-EPSv_bo[0],flush=True)

    end_script = perf_counter()  
    print("Numpy time",end_script-start_script,flush=True)

    
