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
        'R', 'P_R', 'R_grid', 'RP_grid',
        'x', 'y', 'z','x_grid','y_grid','z_grid', 'xb_grid','yb_grid','zb_grid',
        'ddR2', 'ddx2','ddx1','ddy2','ddy1','ddz2','ddz1',
        'axes','Vgrid', '_preconditioner_data','Pg','Pphi','Ptheta',
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

        if args.potential == 'borgis':
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

        
        R_min = extent[0]
        R_max = extent[1]
        x_min = -extent[2]
        x_max = extent[2]
        y_min = -extent[2]
        y_max = extent[2]
        z_min = -extent[2]
        z_max = extent[2]

        print("extent",extent)

        self.R = xp.linspace(R_min, R_max, args.NR)
        self.x = xp.linspace(x_min, x_max, args.Nx)
        self.y = xp.linspace(y_min, y_max, args.Ny)
        self.z = xp.linspace(z_min, z_max, args.Nz)

        self.axes = (self.R, self.x, self.y, self.z)

        self.shape = (args.NR, args.Nx, args.Ny, args.Nz)
        self.boshape = (args.Nx, args.Ny, args.Nz)
        self.size = args.NR * args.Nx * args.Ny * args.Nz

        dR = self.R[1] - self.R[0]
        dx = self.x[1] - self.x[0]
        dy = self.y[1] - self.y[0]
        dz = self.z[1] - self.z[0]
        
        self.P_R  = xp.fft.fftshift(xp.fft.fftfreq(args.NR, dR)) * 2 * xp.pi
        self.RP_grid = xp.meshgrid(self.R, self.P_R, indexing='ij')
        # N.B.: These all lack the factor of -1/(2 * mu)
        # We also are throwing away the returned jacobian of R/r
        #self.ddR2, _ = KE_Borisov(self.R, bare=True)
        self.ddR2  = KE(args.NR, dR, bare=True, cyclic=False)
    
        self.ddx2 = KE(args.Nx, dx, bare=True, cyclic=False)
        self.ddx1 = KE(args.Nx, dx, bare=True, cyclic=False, order=1) 

        self.ddy2 = KE(args.Ny, dy, bare=True, cyclic=False)
        self.ddy1 = KE(args.Ny, dy, bare=True, cyclic=False, order=1)

        self.ddz2 = KE(args.Nz, dz, bare=True, cyclic=False)
        self.ddz1 = KE(args.Nz, dz, bare=True, cyclic=False, order=1)
    
        self.R_grid, self.xb_grid, self.yb_grid, self.zb_grid = xp.meshgrid(self.R, self.x, self.y, self.z, indexing='ij')
        self.x_grid, self.y_grid, self.z_grid,  = xp.meshgrid(self.x, self.y, self.z, indexing='ij')
        self.Vgrid = self.V(self.R_grid, self.xb_grid, self.yb_grid, self.zb_grid)


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

    def V(self, R, r_x, r_y, r_z, spitvals=False):

        mu12 = self.mu12
        M_1 = self.M_1
        M_2 = self.M_2

        kappa2 = r_x*R

        r1e2 = r_x**2 +r_y**2 +r_z**2 + (R)**2*(mu12/M_1)**2 - 2*kappa2*mu12/M_1
        r2e2 = r_x**2 +r_y**2 +r_z**2 + (R)**2*(mu12/M_2)**2 + 2*kappa2*mu12/M_2

        r1e = xp.sqrt(xp.where(r1e2 < 0, 0, r1e2))
        r2e = xp.sqrt(xp.where(r2e2 < 0, 0, r2e2))
        
        if spitvals == True:
            return r1e2,r2e2
        else:
            return self._Vfunc(R, r1e, r2e, (self.g_1, self.g_2))

def Gamma_etf(R,rx,ry,rz,ddx,ddy,ddz,M_1,M_2,mu12,r1e2,r2e2,*xdav):

    if len(xdav)==1:
        xdavx1 = xdavy1 = xdavz1 = xdavx2 = xdavy2 = xdavz2  = xdav[0]
    else:
        xdavx1, xdavy1, xdavz1, xdavx2, xdavy2, xdavz2 = xdav       

    Nx = len(ddx)
    Ny = len(ddy)
    Nz = len(ddz)

    theta1 = xp.exp(-r1e2)
    theta2 = xp.exp(-r2e2)
    partition = theta1 + theta2

    t1 = theta1/partition
    t2 = theta2/partition

    t1px = xp.einsum('ijk,il,Bljk->Bijk',t1,ddx,xdavx1)
    pxt1 = xp.einsum('il,ljk,Bljk->Bijk',ddx,t1,xdavx1)
    t2px = xp.einsum('ijk,il,Bljk->Bijk',t2,ddx,xdavx2)
    pxt2 = xp.einsum('il,ljk,Bljk->Bijk',ddx,t2,xdavx2)

    t1py = xp.einsum('ijk,jl,Bilk->Bijk',t1,ddy,xdavy1)
    pyt1 = xp.einsum('il,jlk,Bjlk->Bjik',ddy,t1,xdavy1)
    t2py = xp.einsum('ijk,jl,Bilk->Bijk',t2,ddy,xdavy2)
    pyt2 = xp.einsum('il,jlk,Bjlk->Bjik',ddy,t2,xdavy2)

    t1pz = xp.einsum('ijk,kl,Bijl->Bijk',t1,ddz,xdavz1)
    pzt1 = xp.einsum('ij,klj,Bklj->Bkli',ddz,t1,xdavz1)
    t2pz = xp.einsum('ijk,kl,Bijl->Bijk',t2,ddz,xdavz2)
    pzt2 = xp.einsum('ij,klj,Bklj->Bkli',ddz,t2,xdavz2)


    gammaetf1x = -0.5*(t1px + pxt1)
    gammaetf1y = -0.5*(t1py + pyt1)
    gammaetf1z = -0.5*(t1pz + pzt1)

    gammaetf2x = -0.5*(t2px + pxt2)   
    gammaetf2y = -0.5*(t2py + pyt2)
    gammaetf2z = -0.5*(t2pz + pzt2)

    return gammaetf1x, gammaetf1y, gammaetf1z, gammaetf2x, gammaetf2y, gammaetf2z

def Gamma_etf_diag(R,rx,ry,rz,ddx,ddy,ddz,M_1,M_2,mu12,r1e2,r2e2):

    Nx = len(ddx)
    Ny = len(ddy)
    Nz = len(ddz)

    theta1 = xp.exp(-r1e2)
    theta2 = xp.exp(-r2e2)
    partition = theta1 + theta2

    t1 = (theta1/partition)
    t2 = (theta2/partition)

    t1px = xp.einsum('aij,ab->abij', t1, ddx)
    pxt1 = xp.einsum('ab,bij->abij', ddx, t1)
    gammasq1x = xp.einsum('abij,bcij->acij', (t1px+pxt1),(t1px+pxt1))
    diag_gammasq1x = 0.25*xp.einsum('aaij->aij', gammasq1x)

    t2px = xp.einsum('aij,ab->abij', t2, ddx)
    pxt2 = xp.einsum('ab,bij->abij', ddx, t2)
    gammasq2x = xp.einsum('abij,bcij->acij', (t2px+pxt2),(t2px+pxt2))
    diag_gammasq2x = 0.25*xp.einsum('aaij->aij', gammasq2x)

    gamma1x2x = xp.einsum('abij,bcij->acij', (t1px+pxt1),(t2px+pxt2))
    diag_gamma1x2x = 0.25*xp.einsum('aaij->aij', gamma1x2x)
    gamma2x1x = xp.einsum('abij,bcij->acij', (t2px+pxt2),(t1px+pxt1))
    diag_gamma2x1x = 0.25*xp.einsum('aaij->aij', gamma2x1x)

    t1py = xp.einsum('iaj,ab->iajb', t1, ddy)
    pyt1 = xp.einsum('ab,ibj->iajb', ddy, t1)
    gammasq1y = xp.einsum('iajb,ibjc->iajc', (t1py+pyt1),(t1py+pyt1))
    diag_gammasq1y = 0.25*xp.einsum('iaja->iaj', gammasq1y)

    t2py = xp.einsum('iaj,ab->iajb', t2, ddy)
    pyt2 = xp.einsum('ab,ibj->iajb', ddy, t2)
    gammasq2y = xp.einsum('iajb,ibjc->iajc', (t2py+pyt2),(t2py+pyt2))
    diag_gammasq2y = 0.25*xp.einsum('iaja->iaj', gammasq2y)
    
    gamma1y2y = xp.einsum('iajb,ibjc->iajc', (t1py+pyt1),(t2py+pyt2))
    diag_gamma1y2y = 0.25*xp.einsum('iaja->iaj', gamma1y2y)
    gamma2y1y = xp.einsum('iajb,ibjc->iajc', (t2py+pyt2),(t1py+pyt1))
    diag_gamma2y1y = 0.25*xp.einsum('iaja->iaj', gamma2y1y)

    t1pz = xp.einsum('ija,ab->ijab', t1, ddz)
    pzt1 = xp.einsum('ab,ijb->ijab', ddz, t1)
    gammasq1z = xp.einsum('ijab,ijbc->ijac', (t1pz+pzt1),(t1pz+pzt1))
    diag_gammasq1z = 0.25*xp.einsum('ijaa->ija', gammasq1z)
   
    t2pz = xp.einsum('ija,ab->ijab', t2, ddz)
    pzt2 = xp.einsum('ab,ijb->ijab', ddz, t2)
    gammasq2z = xp.einsum('ijab,ijbc->ijac', (t2pz+pzt2),(t2pz+pzt2))
    diag_gammasq2z = 0.25*xp.einsum('ijaa->ija', gammasq2z)

    gamma1z2z = xp.einsum('ijab,ijbc->ijac', (t1pz+pzt1),(t2pz+pzt2))
    diag_gamma1z2z = 0.25*xp.einsum('ijaa->ija', gamma1z2z)
    gamma2z1z = xp.einsum('ijab,ijbc->ijac', (t2pz+pzt2),(t1pz+pzt1))
    diag_gamma2z1z = 0.25*xp.einsum('ijaa->ija', gamma2z1z)

    output = (diag_gammasq1x, diag_gammasq2x, diag_gamma1x2x, diag_gamma2x1x, 
              diag_gammasq1y, diag_gammasq2y, diag_gamma1y2y, diag_gamma2y1y,
              diag_gammasq1z, diag_gammasq2z, diag_gamma1z2z, diag_gamma2z1z)


    return output



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
    parser.add_argument('-x', dest="Nx", metavar="Nx", default=400, type=int)
    parser.add_argument('-y', dest="Ny", metavar="Ny", default=250, type=int)
    parser.add_argument('-z', dest="Nz", metavar="Nz", default=250, type=int)
    parser.add_argument('--verbosity', default=2, type=int)
    parser.add_argument('--iterations', metavar='max_iterations', default=10000, type=int)
    parser.add_argument('--subspace', metavar='max_subspace', default=1000, type=int)
    parser.add_argument('--guess', metavar="guess.npz", type=Path, default=None)
    parser.add_argument('--evecs', metavar="guess.npz", type=Path, default=None)
    parser.add_argument('--save', metavar="filename")
    parser.add_argument('--potential', choices=['erf_coulomb', 'borgis'],
                        default='borgis')
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
    
    NR,Nx,Ny,Nz = H.shape
    Nelec = Nx*Ny*Nz 
    
    xdav = xp.random.rand(H.shape[1],H.shape[2],H.shape[3])
    
    def Tx(xdav):
        xdav = xdav.reshape((-1,) + H.boshape)
        Hel_dav = -1/(2*H.mur)*(
            xp.einsum('ij,Bjkl->Bikl',H.ddx2,xdav)\
            +xp.einsum('ij,Bkjl->Bkil',H.ddy2,xdav)\
            +xp.einsum('ij,Bklj->Bkli',H.ddz2,xdav)\
            )
        return Hel_dav.reshape(xdav.shape)

    def _preconditioner_naive(H, dx, e, x0, Ri):
        diagH = buildDiag(H,Ri)
        diagd = diagH - (e - 1e-5)
        return dx/diagd

    def buildDiag(H,Ri):
        ke  = xp.zeros([Nx,Ny,Nz])
        ke += xp.diag(H.ddx2)[:,None,None]
        ke += xp.diag(H.ddy2)[None,:,None]
        ke += xp.diag(H.ddz2)[None,None,:]
        ke *= -1 / (2*H.mur)
        diag = H.Vgrid[Ri] + ke #XXXXXFix Vgrid
        return diag.ravel()

    ival = xp.zeros([NR,1])
    Ad_n = xp.zeros(NR)

    EPS = xp.zeros((H.shape[0], H.shape[0]))
    #EPSsq = xp.zeros((H.shape[0], H.shape[0]))
#
    gammacoeff_R = -1j*H.P_R/H.mu12 
    gammacoeff_phi = -1j*(H.Pphi/H.R)/H.mu12
    gammacoeff_theta = +1j*(H.Ptheta/H.R)/H.mu12
#
    #def buildDiagps(H,Ri):
    #    ke  = xp.zeros([Nx,Ny,Nz])
    #    ke += xp.diag(H.ddx2)[:,None,None]
    #    ke += xp.diag(H.ddy2)[None,:,None]
    #    ke += xp.diag(H.ddz2)[None,None,:]
    #    ke *= -1 / (2*H.mur)
    #    
    #    diag = H.Vgrid[Ri] + ke 
    #    return diag.ravel()

    def buildDiagpssq(H,Ri):

        (diag_gammasq1x, diag_gammasq2x, diag_gamma1x2x, diag_gamma2x1x,\
         diag_gammasq1y, diag_gammasq2y, diag_gamma1y2y, diag_gamma2y1y,\
         diag_gammasq1z, diag_gammasq2z, diag_gamma1z2z, diag_gamma2z1z) = Gamma_etf_diag(H.R[i],H.x_grid,H.y_grid,H.z_grid,H.ddx1,H.ddy1,H.ddz1,H.M_1,H.M_2,H.mu12,r1e2,r2e2)

        ke  = xp.zeros([Nx,Ny,Nz])
        ke += xp.diag(H.ddx2)[:,None,None]
        ke += xp.diag(H.ddy2)[None,:,None]
        ke += xp.diag(H.ddz2)[None,None,:]
        ke += (((H.M_2**2*diag_gammasq1x)+(H.M_1**2*diag_gammasq2x)-(H.M_1*H.M_2*diag_gamma1x2x)-(H.M_1*H.M_2*diag_gamma2x1x))/(H.M_1+H.M_2)**2)
        ke += (((H.M_2**2*diag_gammasq1y)+(H.M_1**2*diag_gammasq2y)-(H.M_1*H.M_2*diag_gamma1y2y)-(H.M_1*H.M_2*diag_gamma2y1y))/(H.M_1+H.M_2)**2)
        ke += (((H.M_2**2*diag_gammasq1z)+(H.M_1**2*diag_gammasq2z)-(H.M_1*H.M_2*diag_gamma1z2z)-(H.M_1*H.M_2*diag_gamma2z1z))/(H.M_1+H.M_2)**2) 
        ke *= -1 / (2*H.mur)
        
        diag = H.Vgrid[Ri] + ke #XXXXXFix Vgrid
        return diag.ravel()


    for i in range(NR):
        print("Atom Ri",i,flush=True)
        diag = buildDiag(H,i)
        
        def Hbo_dav(xdav):
            x = xdav.reshape((-1,)+H.boshape)
            
            Hbodav = H.Vgrid[i]*x + Tx(x)
            return Hbodav.reshape(xdav.shape)

        guess_bo = xp.exp(-(H.Vgrid[i] - xp.min(H.Vgrid[i]))**2/27.211**2).ravel()
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
        Ad_n[i] = e_approx[0]
        ival[i,0] = e_approx[0]

        r1e2, r2e2 = H.V(H.R[i], H.x_grid, H.y_grid, H.z_grid, spitvals=True)
        #diagsq = buildDiagpssq(H,i)

        for j in range(H.shape[0]):
            print("Atom Ri",i,"Atom Rj",j,flush=True)
        
            def ps_ham(xdav):           
                x = xdav.reshape((-1,)+H.boshape)
                Tx = -1/(2*H.mur)*(
                    xp.einsum('ij,Bjkl->Bikl',H.ddx2,x)\
                    +xp.einsum('ij,Bkjl->Bkil',H.ddy2,x)\
                    +xp.einsum('ij,Bklj->Bkli',H.ddz2,x)\
                    )
                gammaetf1x, gammaetf1y, gammaetf1z, gammaetf2x, gammaetf2y, gammaetf2z = Gamma_etf(H.R[i],H.x_grid,H.y_grid,H.z_grid,H.ddx1,H.ddy1,H.ddz1,H.M_1,H.M_2,H.mu12,r1e2,r2e2,x)
                gamma1x = gammaetf1x
                gamma2x = gammaetf2x
                gamma1y = gammaetf1y
                gamma2y = gammaetf2y
                gamma1z = gammaetf1z
                gamma2z = gammaetf2z
                Gammatotx = (H.M_2*gamma1x-H.M_1*gamma2x)/(H.M_1+H.M_2)
                Gammatoty = (H.M_2*gamma1y-H.M_1*gamma2y)/(H.M_1+H.M_2)
                Gammatotz = (H.M_2*gamma1z-H.M_1*gamma2z)/(H.M_1+H.M_2)

                gammasq1x, gammasq1y, gammasq1z, gammasq2x, gammasq2y, gammasq2z = Gamma_etf(H.R[i],H.x_grid,H.y_grid,H.z_grid,H.ddx1,H.ddy1,H.ddz1,H.M_1,H.M_2,H.mu12,r1e2,r2e2, gammaetf1x, gammaetf1y, gammaetf1z, gammaetf2x, gammaetf2y, gammaetf2z)
                gamma1x2x, gamma1y2y, gamma1z2z, gamma2x1x, gamma2y1y, gamma2z1z = Gamma_etf(H.R[i],H.x_grid,H.y_grid,H.z_grid,H.ddx1,H.ddy1,H.ddz1,H.M_1,H.M_2,H.mu12,r1e2,r2e2, gammaetf2x, gammaetf2y, gammaetf2z, gammaetf1x, gammaetf1y, gammaetf1z)

                Gammasqtotx = ((H.M_2**2*gammasq1x)+(H.M_1**2*gammasq2x)-(H.M_1*H.M_2*gamma1x2x)-(H.M_1*H.M_2*gamma2x1x))/(H.M_1+H.M_2)**2
                Gammasqtoty = ((H.M_2**2*gammasq1y)+(H.M_1**2*gammasq2y)-(H.M_1*H.M_2*gamma1y2y)-(H.M_1*H.M_2*gamma2y1y))/(H.M_1+H.M_2)**2
                Gammasqtotz = ((H.M_2**2*gammasq1z)+(H.M_1**2*gammasq2z)-(H.M_1*H.M_2*gamma1z2z)-(H.M_1*H.M_2*gamma2z1z))/(H.M_1+H.M_2)**2

                Hbodav = H.Vgrid[i]*x + Tx + (gammacoeff_R[j]*Gammatotx)+(gammacoeff_phi[i]*Gammatoty)+(gammacoeff_theta[i]*Gammatotz) 
                #Htotsq = Hbodav - (Gammasqtotx +Gammasqtoty + Gammasqtotz)/(2*H.mu12) 
                return Hbodav.reshape(xdav.shape)
                #return Htotsq.reshape(xdav.shape)

            guess_ps = xp.exp(-(H.Vgrid[i] - xp.min(H.Vgrid[i]))**2/27.211**2).ravel()
            with timer_ctx(f"Davidson of size {H.size}"):
                conv, e_approx, evecs = lib.davidson1(
                    ps_ham,
                    guess_ps,
                    #H.diag,
                    #_preconditioner_naive(H, dx, e, x0,i),
                    #lambda dx, e, x0: dx/(diagsq-e+1e-5),
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
            #EPSsq[i, j] = e_approx[0]
            EPS[i, j] = e_approx[0]      
               

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

    EPS += 1/(2*H.mu12)*(Pval**2+H.Pphi**2/Rval**2+H.Ptheta**2/Rval**2)
    HPS = inverse_weyl_transform(EPS, H.shape[0], H.R, H.P_R)
    EPSv = batch_eigvalsh(HPS)
    print("PS vib gap",EPSv[1]-EPSv[0],flush=True)

    #EPSsq += 1/(2*H.mu12)*(Pval**2+H.Pphi**2/Rval**2+H.Ptheta**2/Rval**2)
    #HPSsq = inverse_weyl_transform(EPSsq, H.shape[0], H.R, H.P_R)
    #EPSvsq = batch_eigvalsh(HPSsq)
    #print("PS vib gap sq",EPSvsq[1]-EPSvsq[0],flush=True)

