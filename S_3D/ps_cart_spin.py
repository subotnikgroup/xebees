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
        'shape','boshape','bospinshape','size','guess','k','mu12','_Vfunc',
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
        self.bospinshape = (2,args.Nx, args.Ny, args.Nz)
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

def Gamma_etf(R,ddx,ddy,ddz,t1):

    t1px = xp.einsum('ijk,il->iljk',t1,ddx,optimize=True)
    pxt1 = xp.einsum('il,ljk->iljk',ddx,t1,optimize=True)

    t1py = xp.einsum('ijk,jl->ijlk',t1,ddy,optimize=True)
    pyt1 = xp.einsum('il,jlk->jilk',ddy,t1,optimize=True)

    t1pz = xp.einsum('ikj,jl->ikjl',t1,ddz,optimize=True)
    pzt1 = xp.einsum('il,jkl->jkil',ddz,t1,optimize=True)

    gammaetf1x = -0.5*(t1px + pxt1)
    gammaetf1y = -0.5*(t1py + pyt1)
    gammaetf1z = -0.5*(t1pz + pzt1)

    return gammaetf1x, gammaetf1y, gammaetf1z

def Gamma_erf_spin(R,M1,M2,t1,t2):
    
    #sx = -0.5j*xp.array([[0,1],[1,0]])
    #sy = -0.5*xp.array([[0,1],[-1,0]])
    #sz = -0.5j*xp.array([[1,0],[0,-1]])
    J1xs = -0.5j*t1
    J1ys = 0.5*t1
    J2xs = -0.5j*t2
    J2ys = 0.5*t2

    gammaerf1ys = -1/R*(-J1ys-J2ys)
    gammaerf1zs = -1/R*(J1xs+J2xs)

    gammaerf2ys = -gammaerf1ys 
    gammaerf2zs = -gammaerf1zs
    Gammaerfys = (H.M_2*gammaerf1ys-H.M_1*gammaerf2ys)/(H.M_1+H.M_2)
    Gammaerfzs = (H.M_2*gammaerf1zs-H.M_1*gammaerf2zs)/(H.M_1+H.M_2)

    return Gammaerfys, Gammaerfzs


def Gamma_erf_orb(R,rx,ry,rz,M1,M2,mu12,gammaetf,t1,t2):

    (gammaetf1x, gammaetf1y, gammaetf1z, gammaetf2x, gammaetf2y, gammaetf2z) = gammaetf
    

    J1xa = ry[None,:,None,None]*gammaetf1z
    J1xb = -(rz[None,None,None,:]*gammaetf1y)

    J1ya = (rz[None,None,None,:]*gammaetf1x)
    J1yb = -(rx[:,None,None,None]*gammaetf1z)
    J1yc = +(R*mu12/M1)*gammaetf1z

    
    J2xa = (ry[None,:,None,None]*gammaetf2z)
    J2xb = -(rz[None,None,None,:]*gammaetf2y)

    J2ya = (rz[None,None,None,:]*gammaetf2x)
    J2yb = -(rx[:,None,None,None]*gammaetf2z)
    J2yc = +(R*mu12/M2)*gammaetf2z

    gammaerf1ya = -1/R*(-J1ya-J2ya)
    gammaerf1yb = -1/R*(-J1yb-J2yb)
    gammaerf1yc = -1/R*(-J1yc-J2yc)

    gammaerf1za = -1/R*(J1xa+J2xa)
    gammaerf1zb = -1/R*(J1xb+J2xb)
    
    gammaerf2ya = -gammaerf1ya 
    gammaerf2yb = -gammaerf1yb 
    gammaerf2yc = -gammaerf1yc 
    gammaerf2za = -gammaerf1za
    gammaerf2zb = -gammaerf1zb
    
    
    Gammaerfya = (H.M_2*gammaerf1ya-H.M_1*gammaerf2ya)/(H.M_1+H.M_2)
    Gammaerfyb = (H.M_2*gammaerf1yb-H.M_1*gammaerf2yb)/(H.M_1+H.M_2)
    Gammaerfyc = (H.M_2*gammaerf1yc-H.M_1*gammaerf2yc)/(H.M_1+H.M_2)

    Gammaerfza = (H.M_2*gammaerf1za-H.M_1*gammaerf2za)/(H.M_1+H.M_2)
    Gammaerfzb = (H.M_2*gammaerf1zb-H.M_1*gammaerf2zb)/(H.M_1+H.M_2)
   
    
    return Gammaerfya, Gammaerfyb, Gammaerfyc, Gammaerfza, Gammaerfzb


def Tx(xdav):

    #Hel_dav = -1/(2*H.mur)*(
    #    xp.einsum('ij,Bjkl->Bikl',H.ddx2,xdav,optimize=True)
    #    +xp.einsum('ij,Bkjl->Bkil',H.ddy2,xdav,optimize=True)
    #    +xp.einsum('ij,Bklj->Bkli',H.ddz2,xdav,optimize=True)
    #    )
    Hel_dav = -1/(2*H.mur)*(
        xp.einsum('ij,Bsjkl->Bsikl',H.ddx2,xdav,optimize=True)
        +xp.einsum('ij,Bskjl->Bskil',H.ddy2,xdav,optimize=True)
        +xp.einsum('ij,Bsklj->Bskli',H.ddz2,xdav,optimize=True)
        )
    return Hel_dav.reshape(xdav.shape)

def soc_ham(H,R,rx,ry,rz,ddx,ddy,ddz):

    def soc(xdav):
        xdav = xdav.reshape((-1,) + H.bospinshape)

        rR1 = ((rx-(R*H.mu12/H.M_1))**2 + ry**2 +rz**2)**(1.5)
        rR2 = ((rx+(R*H.mu12/H.M_2))**2 + ry**2 +rz**2)**(1.5)
        c1 = 1/4*(1/137)**2*H.g_1/rR1
        c2 = 1/4*(1/137)**2*H.g_2/rR2
        #xaybzc xx'yy'zz'
        Hsocdav = (
        xp.einsum('sS,y,zc,Bsxyz->Bsxyc', sx, ry, ddz,xdav,optimize=True)*(c1+c2)
        -xp.einsum('sS,z,yb,Bsxyz->Bsxbz', sx, rz, ddy,xdav,optimize=True)*(c1+c2)
        +xp.einsum('sS,z,xa,Bsxyz->Bsayz', sy, rz, ddx,xdav,optimize=True)*(c1+c2)
        -xp.einsum('sS,x,zc,Bsxyz->Bsxyc', sy, rx-(R*H.mu12/H.M_1), ddz,xdav,optimize=True)*c1
        -xp.einsum('sS,x,zc,Bsxyz->Bsxyc', sy, rx+(R*H.mu12/H.M_2), ddz,xdav,optimize=True)*c2
        +xp.einsum('sS,x,yb,Bsxyz->Bsxbz', sz, rx-(R*H.mu12/H.M_1), ddy,xdav,optimize=True)*c1
        +xp.einsum('sS,x,yb,Bsxyz->Bsxbz', sz, rx+(R*H.mu12/H.M_2), ddy,xdav,optimize=True)*c2
        -xp.einsum('sS,y,xa,Bsxyz->Bsayz', sz, ry, ddx,xdav,optimize=True)*(c1+c2)
        )

        return Hsocdav.reshape(xdav.shape)

    return soc

def ps_ham(H,term1,term2,term3,coeffgammaerfy,coeffgammaerfz):
    sz = xp.array([[1,0],[0,-1]])
    def Hx_ps(xdav):
        x = xdav.reshape((-1,)+H.bospinshape).astype(complex) 
        Hpsdav = (
            H.Vgrid[i]*x + Tx(x) +
            +xp.einsum('xayz,Bsxyz->Bsayz', term1, x, optimize=True) 
            +xp.einsum('xybz,Bsxyz->Bsxbz', term2, x, optimize=True) 
            +xp.einsum('xyzc,Bsxyz->Bsxyc', term3, x, optimize=True)
            #+xp.einsum('sS,xyzc,BSxyz->BSxyc',sz,term3,x,optimize=True) 
            +xp.einsum('sS,xyz,Bsxyz->BSxyz',sy,coeffgammaerfy,x,optimize=True)
            +xp.einsum('sS,xyz,Bsxyz->BSxyz',sx,coeffgammaerfz,x,optimize=True)
            #+xp.einsum('sS,xyz,Bsxyz->BSxyz',sy,coeffgammaerfy,x,optimize=True)
            #+xp.einsum('sS,xyz,Bsxyz->BSxyz',sx,coeffgammaerfz,x,optimize=True)
            #+xp.einsum('sS,xyz,BSxyz->BSxyz',sy,coeffgammaerfy,x,optimize=True)
            #+xp.einsum('sS,xyz,BSxyz->BSxyz',sx,coeffgammaerfz,x,optimize=True)
        )
        
        return Hpsdav.reshape(xdav.shape)
        
    return Hx_ps

def Hbo_dav(H,i):
    #xdav = xp.ones(H.bospinshape)
    #x = xdav.reshape((-1,)+H.bospinshape)  
    #print("x",x.shape)      
    #Hbodav = H.Vgrid[i]*x + Tx(x)
    #print("Hbodav one",Hbodav[:,0,:,:,:])
    #print("Hbodav two",Hbodav[:,1,:,:,:])
    def Hxbo(xdav):
        x = xdav.reshape((-1,)+H.bospinshape)
        #print("xshape",x.shape)
        #xup = x[:,0,:,:,:]
        #xdown = x[:,1,:,:,:] 
        #Vxup = H.Vgrid[i]*xup
        #Vxdown = H.Vgrid[i]*xdown      
        #Hbodav = xp.zeros(x.shape)
        #Hbodav[:,0,:,:,:] = Vxup + Tx(xup)## xxxcheck with xdav shape
        #Hbodav[:,1,:,:,:] = Vxdown + Tx(xdown)
        
        Hbodav = H.Vgrid[i]*x + Tx(x)
        
        return Hbodav.reshape(xdav.shape)
    return Hxbo

def buildDiag(H,Ri):
    ke  = xp.zeros([Nx,Ny,Nz])
    ke += xp.diag(H.ddx2)[:,None,None]
    ke += xp.diag(H.ddy2)[None,:,None]
    ke += xp.diag(H.ddz2)[None,None,:]
    ke *= -1 / (2*H.mur)
    diag = H.Vgrid[Ri] + ke #XXXXXFix Vgrid
    diagravel = diag.ravel()
    diagspin = xp.append(diagravel,diagravel)
    return diagspin


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

    #kwargs = dict(optimize=True)
    #if xp.backend == 'torch':
    #    kwargs = {}
    #xp.einsum(..., **kwargs)  

    H = Hamiltonian(args)

    start_script = perf_counter()
    
    NR,Nx,Ny,Nz = H.shape
    Nelec = Nx*Ny*Nz 
    
    ival = xp.zeros([NR,1])
    Ad_n = xp.zeros(NR)

    Rval, Pval = H.RP_grid

    EPS = xp.zeros((H.shape[0], H.shape[0]))
    gammacoeff_R = -1j*(Pval-1/Rval)/H.mu12 
    gammacoeff_phi = +1j*(H.Pphi/H.R)/H.mu12
    gammacoeff_theta = +1j*(H.Ptheta/H.R-1/H.R)/H.mu12
    sx = xp.array([[0,1],[1,0]])
    sy = xp.array([[0,1],[-1,0]])

    with timer_ctx(f"R for loop"):
        for i in range(NR):
            print("Atom Ri",i,flush=True)
            diag = buildDiag(H,i)   

            guess_ns = xp.exp(-(H.Vgrid[i] - xp.min(H.Vgrid[i]))**2/27.211**2).ravel()
            guess_spin = xp.repeat(guess_ns, 2)

            if i==0:
                guess_bo = guess_spin
            else:
                guess_bo = evecs
            #guess_spin = xp.append(guess_ns, guess_ns)

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
                tol=1e-10,
            )
            print("Davidson:", e_approx)
            print(conv)
            Ad_n[i] = e_approx[0]
            ival[i,0] = e_approx[0]
#
            #exit()
            
    
            #r1e2, r2e2 = H.V(H.R[i], H.x_grid, H.y_grid, H.z_grid, spitvals=True)
            #theta1 = xp.exp(-r1e2)
            #theta2 = xp.exp(-r2e2)
            #partition = theta1 + theta2
    #
            #t1 = theta1/partition
            #t2 = theta2/partition
    #
            #gammaetf1x,gammaetf1y,gammaetf1z = Gamma_etf(H.R[i],H.ddx1,H.ddy1,H.ddz1,t1)
            #gammaetf2x,gammaetf2y,gammaetf2z = Gamma_etf(H.R[i],H.ddx1,H.ddy1,H.ddz1,t2)
            #gammaetf = (gammaetf1x, gammaetf1y, gammaetf1z, gammaetf2x, gammaetf2y, gammaetf2z)
            #gammaerfya, gammaerfyb, gammaerfyc, gammaerfza, gammaerfzb = Gamma_erf_orb(H.R[i],H.x,H.y,H.z,H.M_1,H.M_2,H.mu12,gammaetf,t1,t2)
            #gammaerfsy, gammaerfsz = Gamma_erf_spin(H.R[i],H.M_1,H.M_2,t1,t2)
#
            #gammaetfx = (H.M_2*gammaetf1x-H.M_1*gammaetf2x)/(H.M_1+H.M_2)
            #gammaetfy = (H.M_2*gammaetf1y-H.M_1*gammaetf2y)/(H.M_1+H.M_2)
            #gammaetfz = (H.M_2*gammaetf1z-H.M_1*gammaetf2z)/(H.M_1+H.M_2)
            #    
            ##Gammatot =  (gammaetfx, gammaetfy, gammaetfz, gammaerfya, gammaerfyb, gammaerfyc, gammaerfza, gammaerfzb)
            #
            #term2 = (
            #        gammacoeff_phi[i] * gammaetfy +
            #        gammacoeff_theta[i] * gammaerfzb
            #    )
            #term3 = (
            #        gammacoeff_phi[i] * (gammaerfyb + gammaerfyc) +
            #        gammacoeff_theta[i] * (gammaetfz + gammaerfza)
            #    )
#
            #coeffgammaerfy = gammacoeff_phi[i]*gammaerfsy
            #coeffgammaerfz = gammacoeff_theta[i]*gammaerfsz
            #
            #
            #with timer_ctx(f"P for loop"):
            #    for j in range(H.shape[0]):
            #    
            #        print("Atom Ri",i,"Atom Rj",j,flush=True)
            #        #gammacoeff = (gammacoeff_R[i,j], gammacoeff_phi[i], gammacoeff_theta[i])
#
            #        term1 = (
            #            gammacoeff_R[i,j] * gammaetfx +
            #            gammacoeff_phi[i] * gammaerfya
            #        )
            #        if i==0:
            #            guess_ps = guess_spin
            #        else:
            #            guess_ps = evecs_save
#
            #        with timer_ctx(f"Davidson of size {H.size}"):
            #            conv, e_ps_approx, evecs_save = lib.davidson1(
            #                ps_ham(H,term1,term2,term3,coeffgammaerfy,coeffgammaerfz),
            #                guess_ps,
            #                lambda dx, e, x0: dx/(diag-e+1e-5),
            #                nroots=args.k,
            #                max_cycle=args.iterations,
            #                verbose=args.verbosity,
            #                max_space=args.subspace,
            #                max_memory=get_davidson_mem(0.75),
            #                #tol=1e-12, #FIXME:DEBUG
            #                tol=1e-10,
            #            )
    #
            #        print("Davidson:", e_ps_approx)
            #        print(conv)#
            #        EPS[i, j] = e_ps_approx[0]
            #        exit()
                    
    #EPS = xp.loadtxt("rij_matrix.txt")
    #ivalload = xp.loadtxt("ri_values.txt")
    #ival = ivalload.reshape([NR,1])
    #Ad_n= ivalload

    Hbo_new = +1/(2*H.mu12)*(-H.ddR2 + xp.diag(H.Pphi**2/H.R**2)+ xp.diag(H.Ptheta**2/H.R**2)+xp.diag(1/(2*H.R)**2)) +xp.diag(Ad_n)
    Ad_vn_new = batch_eigvalsh(Hbo_new)
    e_bo_new = xp.sort(Ad_vn_new.flatten())
    print("e_bo_new",e_bo_new[0:10])
    bo_new = e_bo_new[1] - e_bo_new[0]
    print("BO new vib gap",bo_new,flush=True)

    EPS_bo = xp.zeros((H.shape[0], H.shape[0]))
    Helmat = xp.repeat(ival,H.shape[0],axis=1)
    EPS_bo += Helmat   
    EPS_bo += 1/(2*H.mu12)*(Pval**2+H.Pphi**2/Rval**2+H.Ptheta**2/Rval**2+1/(2*Rval)**2)
    HPS_bo = inverse_weyl_transform(EPS_bo, H.shape[0], H.R, H.P_R)
    EPSv_bo = batch_eigvalsh(HPS_bo)
    print("EPSv_bo",EPSv_bo[0:10])
    print("Weyl BO vib gap",EPSv_bo[1]-EPSv_bo[0],flush=True)
#
    #EPS += 1/(2*H.mu12)*(Pval**2+H.Pphi**2/Rval**2+H.Ptheta**2/Rval**2+1/(2*Rval)**2)
    #HPS = inverse_weyl_transform(EPS, H.shape[0], H.R, H.P_R)
    #EPSv = batch_eigvalsh(HPS)
    #print("PS vib gap",EPSv[1]-EPSv[0],flush=True)

