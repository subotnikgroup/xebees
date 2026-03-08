from numpy.fft import fft, fftshift
from scipy.integrate import simpson
from scipy.sparse.linalg import lobpcg
from scipy.interpolate import RegularGridInterpolator

from sys import stderr
import argparse as ap
from pathlib import Path
from itertools import product, chain

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
from hamiltonian import  KE, KE_FFT, KE_Borisov_3D, inverse_weyl_transform, inverse_weyl_transform_old
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
        'ddR2', 'ddx2','ddx1','ddy2','ddy1','ddz2','ddz1', 'ddr1'
        'axes','Vgrid', '_preconditioner_data','Pg','Pphi','Ptheta',
        'shape', 'boshape','size','guess','k','mu12','_Vfunc',
        '_locked','max_threads', 'axes'
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

def Gamma_erf(R,rx,ry,rz,M1,M2,mu12,gammaetf):

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
    xdav = xdav.reshape((-1,) + H.boshape)
    Hel_dav = -1/(2*H.mur)*(
        xp.einsum('ij,Bjkl->Bikl',H.ddx2,xdav,optimize=True)
        +xp.einsum('ij,Bkjl->Bkil',H.ddy2,xdav,optimize=True)
        +xp.einsum('ij,Bklj->Bkli',H.ddz2,xdav,optimize=True)
        )
    return Hel_dav.reshape(xdav.shape)

def ps_ham(H,term1,term2,term3):

    #gammaetfx, gammaetfy, gammaetfz, gammaerfya, gammaerfyb, gammaerfyc, gammaerfza, gammaerfzb = Gammatot
    #
    #gammacoeff_R, gammacoeff_phi, gammacoeff_theta = gammacoeff
#
    #term1 = (
    #    gammacoeff_R * gammaetfx +
    #    gammacoeff_phi * gammaerfya
    #)
    #term2 = (
    #    gammacoeff_phi * gammaetfy +
    #    gammacoeff_theta * gammaerfzb
    #)
    #term3 = (
    #    gammacoeff_phi * (gammaerfyb + gammaerfyc) +
    #    gammacoeff_theta * (gammaetfz + gammaerfza)
    #)
        
    def Hx_ps(xdav):
        x = xdav.reshape((-1,)+H.boshape).astype(complex) 
        
        #xaybzc xx'yy'zz'
        #Hpsdav = (H.Vgrid[i]*x + Tx(x) 
        # +xp.einsum('xayz,Bxyz->Bayz',gammacoeff_R*gammaetfx,x,optimize=True)
        # +xp.einsum('xybz,Bxyz->Bxbz',gammacoeff_phi*gammaetfy,x,optimize=True)
        # 
        # +xp.einsum('xayz,Bxyz->Bayz',gammacoeff_phi*gammaerfya,x,optimize=True)#etfx
        # +xp.einsum('xyzc,Bxyz->Bxyc',gammacoeff_phi*gammaerfyb,x,optimize=True)#etfz
        # +xp.einsum('xyzc,Bxyz->Bxyc',gammacoeff_phi*gammaerfyc,x,optimize=True)#etfz
        # 
        # +xp.einsum('xyzc,Bxyz->Bxyc',gammacoeff_theta*gammaetfz,x,optimize=True)
        # +xp.einsum('xyzc,Bxyz->Bxyc',gammacoeff_theta*gammaerfza,x,optimize=True)#etfz
        # +xp.einsum('xybz,Bxyz->Bxbz',gammacoeff_theta*gammaerfzb,x,optimize=True)#etfy
        #)
        
        Hpsdav = (
            H.Vgrid[i]*x + Tx(x) +
            xp.einsum('xayz,Bxyz->Bayz', term1, x, optimize=True) +
            xp.einsum('xybz,Bxyz->Bxbz', term2, x, optimize=True) +
            xp.einsum('xyzc,Bxyz->Bxyc', term3, x, optimize=True)
        )
        return Hpsdav.reshape(xdav.shape)
        

    return Hx_ps

def apply_pr(H, xdav):
    x = xdav.reshape((-1,)+H.boshape).astype(complex) 
    ### ddr1 = dx/dr ddx1 + dy/dr ddy1 + dz/dr ddz1 
    ### dx/dr = sin(gamma)cos(psi)
    ### cos(gamma) = z / r --> sin(gamma) = (x^2+y^2)^(0.5)/r
    ### cos(psi) = x/ (x^2+y^2)^(0.5) --> sin(psi) = y / (x^2+y^2)^(0.5)

    r = xp.sqrt(
            H.x[:,None,None]**2 + H.y[None,:,None]**2 + H.z[None,None,:]**2) #shape xyz
    
    dxdr = H.x[:,None,None]/r ## sin(gamma)cos(psi)
    dydr = H.y[None,:,None]/r ## sin(gamma)sin(psi)
    dzdr = H.z[None,None,:]/r ## cos(gamma)
    ### Symmetrized product
    ddr1 = (0-1j)*0.5*(xp.einsum('xyz, xa, Bxyz -> Bayz ', dxdr , H.ddx1, x, optimize=True)
                    + xp.einsum('ayz, xa, Bxyz -> Bayz', dxdr, H.ddx1, x, optimize=True)
                    + xp.einsum('xyz, yb, Bxyz -> Bxbz', dydr, H.ddy1, x, optimize=True)
                    + xp.einsum('xbz, yb, Bxyz -> Bxbz', dydr, H.ddy1, x, optimize=True)
                    + xp.einsum('xyz, zc, Bxyz -> Bxyc', dzdr, H.ddz1, x, optimize=True)
                    + xp.einsum('xyc, zc, Bxyz -> Bxyc', dzdr, H.ddz1, x, optimize=True))
    return ddr1.reshape(xdav.shape)

def Hbo_dav(H,i):
    def Hxbo(xdav):
        x = xdav.reshape((-1,)+H.boshape)        
        Hbodav = H.Vgrid[i]*x + Tx(x)## xxxcheck with xdav shape
        return Hbodav.reshape(xdav.shape)
    return Hxbo

def buildDiag(H,Ri):
    ke  = xp.zeros([Nx,Ny,Nz])
    ke += xp.diag(H.ddx2)[:,None,None]
    ke += xp.diag(H.ddy2)[None,:,None]
    ke += xp.diag(H.ddz2)[None,None,:]
    ke *= -1 / (2*H.mur)
    diag = H.Vgrid[Ri] + ke #XXXXXFix Vgrid
    return diag.ravel()


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
    parser.add_argument('--backend', default='cupy')
    parser.add_argument('-splits', default=0, type=int)
    parser.add_argument('-split_idx', default=1, type=int)
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

    ###

    EPS = xp.zeros((H.shape[0], H.shape[0]))
    pPS = xp.zeros((4, H.shape[0], H.shape[0]), dtype=xp.complex128) # <pe>(R,P)

    gammacoeff_R = -1j*(Pval-1/Rval)/H.mu12 
    gammacoeff_phi = +1j*(H.Pphi/H.R)/H.mu12
    gammacoeff_theta = +1j*(H.Ptheta/H.R-1/H.R)/H.mu12
    if args.splits > 0:
        sequence = generalized_sequence(NR, args.splits, args.split_idx)
        print("sequence",sequence)
        iR = sequence[0]
        print("iR",iR)
    else:
        iR = NR//2
        sequence = list(chain(
            [iR],
            range(iR - 1, -1, -1),
            range(iR + 1, NR)))

    jR = NR//2
    ps_sequence = list( chain(
            [jR],
            range(jR - 1, -1, -1),
            range(jR + 1, NR)))
    evecs_prev = True

    with timer_ctx(f"R for loop"):
        for i in sequence:
            print("Atom Ri",i,flush=True)
            diag = buildDiag(H,i)       

            guess = xp.exp(-(H.Vgrid[i] - xp.min(H.Vgrid[i]))**2/27.211**2).ravel()
            if evecs_prev == True:
                guess_bo = guess
            else:
                guess_bo = evecs
            conv, e_approx, evecs = lib.davidson1(
                Hbo_dav(H,i),
                guess,
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
    
            r1e2, r2e2 = H.V(H.R[i], H.x_grid, H.y_grid, H.z_grid, spitvals=True)
            theta1 = xp.exp(-r1e2)
            theta2 = xp.exp(-r2e2)
            partition = theta1 + theta2
    
            t1 = theta1/partition
            t2 = theta2/partition
    
            gammaetf1x,gammaetf1y,gammaetf1z = Gamma_etf(H.R[i],H.ddx1,H.ddy1,H.ddz1,t1)
            gammaetf2x,gammaetf2y,gammaetf2z = Gamma_etf(H.R[i],H.ddx1,H.ddy1,H.ddz1,t2)
            gammaetf = (gammaetf1x, gammaetf1y, gammaetf1z, gammaetf2x, gammaetf2y, gammaetf2z)
            gammaerfya, gammaerfyb, gammaerfyc, gammaerfza, gammaerfzb = Gamma_erf(H.R[i],H.x,H.y,H.z,H.M_1,H.M_2,H.mu12,gammaetf)

            gammaetfx = (H.M_2*gammaetf1x-H.M_1*gammaetf2x)/(H.M_1+H.M_2)
            gammaetfy = (H.M_2*gammaetf1y-H.M_1*gammaetf2y)/(H.M_1+H.M_2)
            gammaetfz = (H.M_2*gammaetf1z-H.M_1*gammaetf2z)/(H.M_1+H.M_2)
                
            Gammatot =  (gammaetfx, gammaetfy, gammaetfz, gammaerfya, gammaerfyb, gammaerfyc, gammaerfza, gammaerfzb)

            term2 = (
                    gammacoeff_phi[i] * gammaetfy 
                     + gammacoeff_theta[i] * gammaerfzb
                )
            term3 = (
                     gammacoeff_phi[i] * (gammaerfyb + gammaerfyc) +
                    gammacoeff_theta[i] * (gammaetfz + gammaerfza)
                )
            
            with timer_ctx(f"P for loop"):
                for j in ps_sequence:
                
                    print("Atom Ri",i,"Atom Pj",j,flush=True)
                    gammacoeff = (gammacoeff_R[i,j], gammacoeff_phi[i], gammacoeff_theta[i])

                    term1 = (
                        gammacoeff_R[i,j] * gammaetfx 
                         + gammacoeff_phi[i] * gammaerfya
                    )
                    if evecs_prev == True and j==NR//2:
                        guess_ps = evecs
                        evecs_prev = False
                    else:
                        guess_ps = evecs_save
                    
                    Hx_ps = ps_ham(H,term1,term2,term3)
                    with timer_ctx(f"Davidson of size {H.size}"):
                        conv, e_ps_approx, evecs_save = lib.davidson1(
                            Hx_ps,
                            guess_ps,
                            lambda dx, e, x0: dx/(diag-e-(1e-5-1e-5j)),
                            nroots=args.k,
                            max_cycle=args.iterations,
                            verbose=args.verbosity,
                            max_space=args.subspace,
                            max_memory=get_davidson_mem(0.75),
                            #tol=1e-12, #FIXME:DEBUG
                            tol=1e-12,
                        )
    
                    print("Davidson:", e_ps_approx)

                    pe_r = xp.sum(evecs_save[0].conj()*apply_pr(H,evecs_save[0])) # < 0 | p_e | 0 > for PS
                    psi0 = evecs_save[0].reshape(H.boshape)
                    pe_x = xp.einsum('xyz, xa, ayz ->', psi0.conj(), (0-1j)*H.ddx1, psi0)
                    pe_y = xp.einsum('xyz, yb, ayz ->', psi0.conj(), (0-1j)*H.ddy1, psi0)
                    pe_z = xp.einsum('xyz, zc, xyc ->', psi0.conj(), (0-1j)*H.ddz1, psi0)

                    print("<pe> on g.s.", pe_x.real, pe_y.real, pe_z.real, pe_r.real)
                    print("<x>,", xp.einsum('xyz, x, xyz-> ', psi0.conj(), H.x, psi0 ))
                    print(conv)#

                    EPS[i, j] = e_ps_approx[0]
                    pPS[:,i,j] = xp.asarray([pe_x,pe_y,pe_z,pe_r])
                    

                    
    #EPS = xp.loadtxt("rij_matrix.txt")
    #ivalload = xp.loadtxt("ri_values.txt")
    #ival = ivalload.reshape([NR,1])
    #Ad_n= ivalload

    Hbo_new = +1/(2*H.mu12)*(-H.ddR2 + xp.diag(H.Pphi**2/H.R**2)+ xp.diag(H.Ptheta**2/H.R**2)+xp.diag(1/(2*H.R)**2)) +xp.diag(Ad_n)
    Ad_vn_new, Unv_bo = xp.linalg.eigh(Hbo_new)
    e_bo_new = xp.sort(Ad_vn_new.flatten())
    bo_new = e_bo_new[1] - e_bo_new[0]
    print("e_bo_new",e_bo_new[0:10])
    print("BO new vib gap",bo_new,flush=True)

    R_bo = xp.sum(Unv_bo[:,0].conj()*H.R*Unv_bo[:,0]).real
    print("BO bond length: <chi_0| R| chi_0 >:", R_bo)

    EPS_bo = xp.zeros((H.shape[0], H.shape[0]))
    Helmat = xp.repeat(ival,H.shape[0],axis=1)
    EPS_bo += Helmat   
    EPS_bo += 1/(2*H.mu12)*(Pval**2+H.Pphi**2/Rval**2+H.Ptheta**2/Rval**2+1/(2*Rval)**2)
    HPS_bo = inverse_weyl_transform(EPS_bo, H.shape[0], H.R, H.P_R)
    EPSv_bo = batch_eigvalsh(HPS_bo)
    print("e_bo_new Weyl",EPSv_bo[0:10])
    print("Weyl BO vib gap",EPSv_bo[1]-EPSv_bo[0],flush=True)

    EPS += 1/(2*H.mu12)*(Pval**2+H.Pphi**2/Rval**2+H.Ptheta**2/Rval**2+1/(2*Rval)**2)
    HPS = inverse_weyl_transform_old(EPS, H.shape[0], H.R, H.P_R)
    EPSv = batch_eigvalsh(HPS)
    EPSv, UPSv = xp.linalg.eigh(HPS)
    print("e_bo_new Weyl",EPSv[0:10])
    print("PS vib gap",EPSv[1]-EPSv[0],flush=True)

    print("Real pPS check", [xp.sum(xp.abs(pPS[i].imag)) for i in range(4)])

    print("Weyl check HPS Hermitian", xp.sum(xp.abs(HPS.conj().T - HPS)))
    with xp.printoptions(precision=3):
        print(HPS[:4,:4])

    Hpe_x = inverse_weyl_transform_old(pPS[0], H.shape[0], H.R, H.P_R)
    Hpe_y = inverse_weyl_transform_old(pPS[1], H.shape[0], H.R, H.P_R)
    Hpe_z = inverse_weyl_transform_old(pPS[2], H.shape[0], H.R, H.P_R)
    Hpe_r = inverse_weyl_transform_old(pPS[3], H.shape[0], H.R, H.P_R)
    print("check Hpe_x Weyl hermitian", xp.sum(xp.abs(Hpe_x.conj().T-Hpe_x)))
    with xp.printoptions(precision=3):
        print(Hpe_x[:4,:4])
    
    dR = H.R[1]-H.R[0]
    pe_chix = xp.sum(UPSv[:,1].conj()*Hpe_x@UPSv[:,0])
    pe_chiy = xp.sum(UPSv[:,1].conj()*Hpe_y@UPSv[:,0])
    pe_chiz = xp.sum(UPSv[:,1].conj()*Hpe_z@UPSv[:,0])
    pe_chir = xp.sum(UPSv[:,1].conj()*Hpe_r@UPSv[:,0])
    print("<chi_1|pe| chi0>:", pe_chix, pe_chiy, pe_chiz, pe_chir)

    R_ps = xp.sum(UPSv[:,0].conj()*H.R*UPSv[:,0]).real
    print("PS bond length: <chi_0| R| chi_0 >:", R_ps)

    xp.savez("test_PS_3D.npz", R=H.R, P=H.P_R, EPS=EPS, HPS=HPS,EPSv=EPSv, UPSv=UPSv, pPS=pPS)

