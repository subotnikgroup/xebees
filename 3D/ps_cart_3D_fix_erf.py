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
        'R', 'P_R', 'R_grid', 'RP_grid','r',
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
        self.r = xp.sqrt(self.x[:,None,None]**2+self.y[None,:,None]**2 + self.z[None,None,:]**2)


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
        #self.ddR2  = KE(args.NR, dR, bare=True, cyclic=False)
        self.ddR2  = KE_FFT(args.NR, self.P_R, self.R)
    
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


def Gamma_etf(H, t1):
    ddx, ddy, ddz = H.ddx1, H.ddy1, H.ddz1

    t1px = xp.einsum('ijk,il->iljk', t1, ddx, optimize=True)
    pxt1 = xp.einsum('il,ljk->iljk', ddx, t1, optimize=True)
    t1py = xp.einsum('ijk,jl->ijlk', t1, ddy, optimize=True)
    pyt1 = xp.einsum('il,jlk->jilk', ddy, t1, optimize=True)
    t1pz = xp.einsum('ikj,jl->ikjl', t1, ddz, optimize=True)
    pzt1 = xp.einsum('il,jkl->jkil', ddz, t1, optimize=True)
    gammaetf1x = -0.5*(t1px + pxt1)
    gammaetf1y = -0.5*(t1py + pyt1)
    gammaetf1z = -0.5*(t1pz + pzt1)
    
    return gammaetf1x, gammaetf1y, gammaetf1z


def Gamma_erf_orb(H,Ridx, t1, t2):
    rx, ry, rz = H.x,H.y,H.z
    Nx, Ny, Nz = H.boshape
    #ddx1 = xp.zeros([Nx,Nx,Ny,Nz])
    #ddy1 = xp.zeros([Nx,Ny,Ny,Nz])
    #ddz1 = xp.zeros([Nx,Ny,Nz,Nz])

    ddx1 = H.ddx1[:, :, None, None]
    ddy1 = H.ddy1[None, :, :, None]
    ddz1 = H.ddz1[None, None, :, :]
    coeff = H.R[Ridx]/2*((H.M_2*t1-H.M_1*t2)/(H.M_1+H.M_2))

    Jya = -1/H.R[Ridx]*xp.einsum('x,xybz->xybz', rx, ddy1, optimize=True)
    Jyb = -1/H.R[Ridx]*xp.einsum('y,xayz->xayz', ry, ddx1, optimize=True)
    Jyc = -1/H.R[Ridx]*xp.einsum('xyz,xybz->xybz', coeff, ddy1, optimize=True)
    Jyd = -1/H.R[Ridx]*xp.einsum('xybz,xbz->xybz', ddy1, coeff,optimize=True)

    Jza = 1/H.R[Ridx]*xp.einsum('z,xayz->xayz', rz, ddx1, optimize=True)
    Jzb = 1/H.R[Ridx]*xp.einsum('x,xyzc->xyzc', rx, ddz1, optimize=True)
    Jzc = 1/H.R[Ridx]*xp.einsum('xyz,xyzc->xyzc', coeff,ddz1,optimize=True)
    Jzd = 1/H.R[Ridx]*xp.einsum('xyzc,xyc->xyzc', ddz1, coeff,optimize=True)

    return Jya, Jyb, Jyc, Jyd, Jza, Jzb, Jzc, Jzd

def Gamma_sq_etf(H,t1):
    t1px =  -0.5*xp.einsum('xyz,xa->xayz', t1, H.ddx2, optimize=True)
    t1px += -0.5*xp.einsum('xa,ayz->xayz', H.ddx2, t1, optimize=True)
    t1py =  -0.5*xp.einsum('xyz,yb->xybz', t1, H.ddy2, optimize=True)
    t1py += -0.5*xp.einsum('yb,xbz->xybz', H.ddy2, t1, optimize=True)
    t1pz  = -0.5*xp.einsum('xyz,zc->xyzc', t1, H.ddz2, optimize=True)
    t1pz += -0.5*xp.einsum('zc,xyc->xyzc', H.ddz2, t1, optimize=True)

    diag = (t1*xp.diag(H.ddx2)[:,None,None]+ t1*xp.diag(H.ddy2)[None,:,None]+t1*xp.diag(H.ddz2)[None,None,:]
            ).reshape(H.x.size*H.y.size*H.z.size)
    return t1px, t1py, t1pz, diag

def Gamma_sq_erf(H, t1,t2, Ridx):
    ## Gsq = { t1/4M1 + t2/4M2 , K-1*(l1^2 + l2^2)}
    ##l^2 = l_y^2 + l_z^2, K matrix removes l_x^2
    wt = t1/(4*H.M_1) + t2/(4*H.M_2) # mass weighted theta
    wt /= H.mu12*H.R[Ridx]**2 # Include K-1 in wt
    wR = (H.M_2-H.M_1)/(H.M_1+H.M_2)*H.R[Ridx] # mass weighted R = (mu12/M1 - mu12/M2)R
    wR2 = (H.M_1**2 +H.M_2**2)/(H.M_1+H.M_2)**2*H.R[Ridx]**2 # mass weighted R^2

    ## 5 unique derivative terms
    ## Gsq_z ~ C1*ddy2 + C2*ddx2 + C3*ddx1*ddy1 +C4*ddy1 + C5*ddx1
    ## return objects all of the same shape for minimal terms
    ## (C1+C4), (C2+C5), C3

    Cz1  = 0.5*xp.einsum('xyz, x, yb -> xybz', wt, H.x**2 - 2*H.x*wR + wR2, H.ddy2)    
    Cz1 += 0.5*xp.einsum('xbz, x, yb -> xybz', wt, H.x**2 - 2*H.x*wR + wR2, H.ddy2)
    Cz2 =  0.5*xp.einsum('xyz, y, xa -> xayz', wt, H.y**2, H.ddx2)
    Cz2 += 0.5*xp.einsum('ayz, y, xa -> xayz', wt, H.y**2, H.ddx2)
    Cz3 =  0.5*xp.einsum('xyz,x,y,xa,yb -> xaybz', wt, -2*(H.x-wR), H.y, H.ddx1, H.ddy1)
    Cz3 += 0.5*xp.einsum('abz,x,y,xa,yb -> xaybz', wt, -2*(H.x-wR), H.y, H.ddx1, H.ddy1)
    Cz4 =  0.5*xp.einsum('xyz,y,yb -> xybz', wt, -H.y, H.ddy1)
    Cz4 += 0.5*xp.einsum('xbz,b,yb -> xybz', wt, -H.y, H.ddy1)
    Cz5 =  0.5*xp.einsum('xyz,x,xa -> xayz', wt, -H.x, H.ddx1)
    Cz5 += 0.5*xp.einsum('ayz,a,xa -> xayz', wt, -H.x, H.ddx1)

    Cy1 =  0.5*xp.einsum('xyz, z, xa -> xayz', wt, H.z**2, H.ddx2)
    Cy1 += 0.5*xp.einsum('ayz, z, xa -> xayz', wt, H.z**2, H.ddx2)
    Cy2  = 0.5*xp.einsum('xyz, x, zc -> xyzc', wt, H.x**2 - 2*H.x*wR + wR2, H.ddz2)    
    Cy2 += 0.5*xp.einsum('xyc, x, zc -> xyzc', wt, H.x**2 - 2*H.x*wR + wR2, H.ddz2)
    Cy3 =  0.5*xp.einsum('xyz,x,z,xa,zc -> xayzc', wt, -2*(H.x-wR), H.z, H.ddx1, H.ddz1)
    Cy3 += 0.5*xp.einsum('ayc,x,z,xa,zc -> xayzc', wt, -2*(H.x-wR), H.z, H.ddx1, H.ddz1)
    Cy4 =  0.5*xp.einsum('xyz,z,zc -> xyzc', wt, -H.z, H.ddz1)
    Cy4 += 0.5*xp.einsum('xyc,c,zc -> xyzc', wt, -H.z, H.ddz1)
    Cy5 =  0.5*xp.einsum('xyz,x,xa -> xayz', wt, -H.x, H.ddx1)
    Cy5 += 0.5*xp.einsum('ayz,a,xa -> xayz', wt, -H.x, H.ddx1)

    ## diagonal elements of Gamma 
    diag =  wt*(H.x**2 - 2*H.x*wR + wR2)[:,None,None]*(xp.diag(H.ddy2)[None,:,None] +xp.diag(H.ddz2)[None,None,:])
    diag += wt*(H.y[None,:,None]+H.z[None,None,:])*xp.diag(H.ddx2)[:,None,None]
 
    ddx_terms  = Cz2+Cz5+Cy1+Cy5 # xayz
    ddy_terms  = Cz1+Cz4         # xybz
    ddz_terms  = Cy2+Cy4         # xyzc
    dxdy_terms = Cz3             # xaybz
    dxdz_terms = Cy3             # xayzc

    return ddx_terms, ddy_terms, ddz_terms, dxdy_terms, dxdz_terms, (diag).reshape(H.x.size*H.y.size*H.z.size)

def Tx(H,xdav):
    xdav = xdav.reshape((-1,) + H.boshape)
    Hel_dav = -1/(2*H.mur)*(
        xp.einsum('ij,Bjkl->Bikl',H.ddx2,xdav,optimize=True)
        +xp.einsum('ij,Bkjl->Bkil',H.ddy2,xdav,optimize=True)
        +xp.einsum('ij,Bklj->Bkli',H.ddz2,xdav,optimize=True)
        )
    return Hel_dav.reshape(xdav.shape)

def ps_ham(H,ddx_terms,ddy_terms,ddz_terms,Ri, dxdy_terms=None):
        
    def Hx_ps(xdav):
        x = xdav.reshape((-1,)+H.boshape).astype(complex) 
        
        
        Hpsdav = (
            H.Vgrid[Ri]*x + Tx(H,x) 
            +xp.einsum('xayz,Bayz->Bxyz', ddx_terms, x, optimize=True) 
            +xp.einsum('xybz,Bxbz->Bxyz', ddy_terms, x, optimize=True) 
            +xp.einsum('xyzc,Bxyc->Bxyz', ddz_terms, x, optimize=True)
        )
        if dxdy_terms is not None:
            Hdxdy = dxdy_terms[0]
            Hdxdz = dxdy_terms[1]
            Hpsdav += xp.einsum('xaybz, Babz->Bxyz', Hdxdy, x, optimize=True)
            Hpsdav += xp.einsum('xayzc, Bayc->Bxyz', Hdxdz, x, optimize=True)
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
    r[r<1e-10] = 1e-10 
    
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

def apply_l(H, xdav):
    x = xdav.reshape((-1,)+H.boshape).astype(complex) 

    lx_a = xp.einsum('y,zc,Bxyc -> Bxyz', H.y,H.ddz1, x, optimize=True)
    lx_b = xp.einsum('z,yb,Bxbz -> Bxyz', H.z,H.ddy1, x, optimize=True)
    lx = -1j*(lx_a - lx_b)

    ly_a = xp.einsum('z,xa,Bayz -> Bxyz', H.z, H.ddx1, x, optimize=True)
    ly_b = xp.einsum('x,zc,Bxyc -> Bxyz', H.x, H.ddz1, x, optimize=True)
    ly = -1j*(ly_a-ly_b)

    lz_a = xp.einsum('x,yb,Bxbz -> Bxyz', H.x,H.ddy1, x, optimize=True)
    lz_b = xp.einsum('y,xa,Bayz -> Bxyz', H.y,H.ddx1, x, optimize=True)
    lz = -1j*(lz_a -lz_b)

    return (lx,ly,lz)
    # return (lx.reshape(xdav.shape),ly.reshape(xdav.shape),lz.reshape(xdav.shape))

def Hbo_dav(H,i):
    def Hxbo(xdav):
        x = xdav.reshape((-1,)+H.boshape)        
        Hbodav = H.Vgrid[i]*x + Tx(H,x)## xxxcheck with xdav shape
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

    def generalized_sequence(NR, num_splits, split_idx):
        nodes = xp.linspace(0, NR, num_splits + 1, dtype=xp.int32).tolist()
        parts = []
        midpoint_idx = num_splits // 2
        for i in range(num_splits):
            start = nodes[i]
            end = nodes[i + 1]
            if i < midpoint_idx:
                if i == 0:
                    chunk = np.arange(end, start - 1, -1)
                else:
                    chunk = np.arange(end, start, -1)
            else:
                if i == num_splits - 1:
                    chunk = np.arange(start + 1, end)
                else:
                    chunk = np.arange(start + 1, end + 1)
            parts.append(chunk)
        return parts[split_idx - 1]
    folder = os.getcwd()

    H = Hamiltonian(args)

    start_script = perf_counter()
    
    NR,Nx,Ny,Nz = H.shape
    Nelec = Nx*Ny*Nz 
    
    ival = xp.zeros([NR,1])
    Ad_n = xp.zeros(NR)

    Rval, Pval = H.RP_grid

    ### E(R,P)
    EPS = xp.zeros((H.shape[0], H.shape[0]))

    ### other electronic observables
    pPS = xp.zeros((4, H.shape[0], H.shape[0]), dtype=xp.complex128) # <pe>(R,P)
    rPS = xp.zeros((H.shape[0], H.shape[0]), dtype=xp.complex128)
    GammaPS = xp.zeros((3, H.shape[0], H.shape[0]), dtype=xp.complex128)
    rBO = xp.zeros((H.shape[0]), dtype=xp.complex128)
    lPS = xp.zeros((3, H.shape[0], H.shape[0]), dtype=xp.complex128)
    l2PS = xp.zeros((H.shape[0], H.shape[0]), dtype=xp.complex128)
    l2BO = xp.zeros((H.shape[0]), dtype=xp.complex128)
    TePS = xp.zeros((H.shape[0], H.shape[0]), dtype=xp.complex128)
    TeBO = xp.zeros((H.shape[0]), dtype=xp.complex128)

    # gammacoeff_R = -1j*(Pval-1/Rval)/H.mu12 
    # gammacoeff_phi = +1j*(H.Pphi/H.R)/H.mu12
    # gammacoeff_theta = +1j*(H.Ptheta/H.R-1/H.R)/H.mu12

    gammacoeff_R = -1j*(Pval)/H.mu12 #-1/Rval)/H.mu12 
    gammacoeff_phi = +1j*(H.Pphi/H.R)/H.mu12
    gammacoeff_theta = +1j*(H.Ptheta/H.R)/H.mu12#-1/H.R)/H.mu12

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
            psi0_bo = evecs[0].reshape(H.boshape)
            rBO[i] = xp.einsum('xyz,xyz,xyz->', psi0_bo.conj(), H.r, psi0_bo)
            psi0_lx, psi0_ly, psi0_lz = apply_l(H,evecs[0])
            l00x = xp.sum(psi0_bo.conj()*psi0_lx)
            l00y = xp.sum(psi0_bo.conj()*psi0_ly)
            l00z = xp.sum(psi0_bo.conj()*psi0_lz)
            l200_x = xp.sum(psi0_lx.conj()*psi0_lx)
            l200_y = xp.sum(psi0_ly.conj()*psi0_ly)
            l200_z = xp.sum(psi0_lz.conj()*psi0_lz)
            l200 = l200_x+l200_y+l200_z
            l2BO[i] = l200
            print("lxyz on gs:", l00x, l00y, l00z)
            print("l2 on gs:", l200)
            psi0_Te = Tx(H,psi0_bo)#*2*H.mur
            TeBO[i] = xp.sum(psi0_bo.conj()*psi0_Te)
    
            r1e2, r2e2 = H.V(H.R[i], H.x_grid, H.y_grid, H.z_grid, spitvals=True)
            theta1 = xp.exp(-r1e2)
            theta2 = xp.exp(-r2e2)
            partition = theta1 + theta2
    
            #t1 = theta1/partition
            #t2 = theta2/partition

            t1 = 1/(1+xp.exp(r1e2-r2e2))
            t2 = 1/(1+xp.exp(r2e2-r1e2))
    
            gammaetf1x,gammaetf1y,gammaetf1z = Gamma_etf(H, t1)
            gammaetf2x,gammaetf2y,gammaetf2z = Gamma_etf(H, t2)
            
            Jya, Jyb, Jyc, Jyd, Jza, Jzb, Jzc, Jzd = Gamma_erf_orb(H,i, t1, t2)

            gammaetfx = (H.M_2*gammaetf1x-H.M_1*gammaetf2x)/(H.M_1+H.M_2)
            gammaetfy = (H.M_2*gammaetf1y-H.M_1*gammaetf2y)/(H.M_1+H.M_2)
            gammaetfz = (H.M_2*gammaetf1z-H.M_1*gammaetf2z)/(H.M_1+H.M_2)


            ddy_terms = (                    
                   gammacoeff_phi[i] * (Jya- Jyc- Jyd+gammaetfy)
               )
            
            ddz_terms = (
                   gammacoeff_theta[i]*(-Jzb+Jzc+Jzd+gammaetfz)                   
               )
            
            if args.no_ERF:
                ddy_terms = (                    
                        gammacoeff_phi[i] * (gammaetfy)
                    )
                
                ddz_terms = (
                        gammacoeff_theta[i]*(gammaetfz)                   
                    )

            if args.Gammasq:
                wt = (t1/(4*H.M_1) + t2/(4*H.M_2))
                Gsq_etf_ddx, Gsq_etf_ddy, Gsq_etf_ddz, Gsq_etf_diag = Gamma_sq_etf(H,wt)

                ddy_terms += Gsq_etf_ddy
                ddz_terms += Gsq_etf_ddz
                diag += Gsq_etf_diag
                if not args.no_ERF:
                    Gsq_erf_ddx, Gsq_erf_ddy, Gsq_erf_ddz, Gsq_erf_dxdy, Gsq_erf_dxdz, Gsq_erf_diag = Gamma_sq_erf(H,t1,t2,i)
                    ddy_terms += Gsq_erf_ddy
                    ddz_terms += Gsq_erf_ddz

            with timer_ctx(f"P for loop"):
                for j in ps_sequence:
                
                    print("Atom Ri",i,"Atom Pj",j,flush=True)
                    
                    ddx_terms = (gammacoeff_R[i,j] * gammaetfx - gammacoeff_phi[i] * Jyb +
                                gammacoeff_theta[i]* (Jza))
                    if args.no_ERF:
                        ddx_terms = (gammacoeff_R[i,j] * gammaetfx)
                    
                    Hx_ps = ps_ham(H,ddx_terms,ddy_terms,ddz_terms,i)

                    if args.Gammasq:
                        ddx_terms += Gsq_etf_ddx
                        if not args.no_ERF:
                            ddx_terms += Gsq_erf_ddx
                            Hx_ps = ps_ham(H,ddx_terms,ddy_terms,ddz_terms,i, dxdy_terms=(Gsq_erf_dxdy, Gsq_erf_dxdz))


                    if evecs_prev == True and j==NR//2:
                        guess_ps = evecs
                        evecs_prev = False
                    else:
                        guess_ps = evecs_save
                    
                    
                    with timer_ctx(f"Davidson of size {H.size}"):
                        conv, e_ps_approx, evecs_save = lib.davidson1(
                            Hx_ps,
                            guess_ps,
                            #lambda dx, e, x0: dx/(diag-e-(1e-5-1e-5j)),
                            lambda dx, e, x0: dx/(diag-e+(1e-5)),
                            nroots=args.k,
                            max_cycle=args.iterations,
                            verbose=args.verbosity,
                            max_space=args.subspace,
                            max_memory=get_davidson_mem(0.75),
                            #tol=1e-12, #FIXME:DEBUG
                            tol=1e-12,
                        )

                    print("Davidson:", e_ps_approx)
                    print(conv)#
                    EPS[i, j] = e_ps_approx[0]

                    ### electronic observables for PS
                    pe_r = xp.sum(evecs_save[0].conj()*apply_pr(H,evecs_save[0])) # < 0 | p_e | 0 > for PS
                    psi0 = evecs_save[0].reshape(H.boshape)
                    pe_x = xp.einsum('xyz, xa, ayz ->', psi0.conj(), (-1j)*H.ddx1, psi0, optimize=True)
                    pe_y = xp.einsum('xyz, yb, xbz ->', psi0.conj(), (-1j)*H.ddy1, psi0, optimize=True)
                    pe_z = xp.einsum('xyz, zc, xyc ->', psi0.conj(), (-1j)*H.ddz1, psi0, optimize=True)
                    print("<pe> on g.s.", pe_x.real, pe_y.real, pe_z.real, pe_r.real)
                    pPS[:,i,j] = xp.asarray([pe_x,pe_y,pe_z,pe_r])
                    
                    Gamma_x = xp.einsum('xyz,xayz,ayz->',psi0.conj(), 1j*gammaetfx, psi0)
                    Gamma_y = xp.einsum('xyz,xybz,xbz->',psi0.conj(), 1j*gammaetfy, psi0)
                    Gamma_z = xp.einsum('xyz,xyzc,xyc->',psi0.conj(), 1j*gammaetfz, psi0)
                    print("<Gamma> on gs:", Gamma_x.real, Gamma_y.real, Gamma_z.real)
                    GammaPS[:,i,j] = xp.asarray([Gamma_x, Gamma_y, Gamma_z])

                    psi0_lx, psi0_ly, psi0_lz = apply_l(H,evecs_save[0])
                    psi1_lx, psi1_ly, psi1_lz = apply_l(H,evecs_save[1])
                    
                    l00x = xp.sum(psi0.conj()*psi0_lx)
                    l00y = xp.sum(psi0.conj()*psi0_ly)
                    l00z = xp.sum(psi0.conj()*psi0_lz)
                    l200_x = xp.sum(psi0_lx.conj()*psi0_lx)
                    l200_y = xp.sum(psi0_ly.conj()*psi0_ly)
                    l200_z = xp.sum(psi0_lz.conj()*psi0_lz)
                    l200 = l200_x+l200_y+l200_z

                    print("<l> on gs:", l00x, l00y, l00z)
                    print("<l^2> on gs", l200)
                    lPS[:,i,j] = xp.stack((l00x,l00y,l00z))
                    l2PS[i,j] = l200

                    psi0_Te = Tx(H,psi0)#*2*H.mur
                    TePS[i,j] = xp.sum(psi0.conj()*psi0_Te)
                    
                    rPS[i,j] = xp.einsum('xyz,xyz,xyz->', psi0.conj(), H.r, psi0, optimize=True)
                    print("<r> on gs:", rPS[i,j].real)
                    print() # add a new line between each RP point
            
                    

    if args.splits > 0:

        numpy.save(os.path.join(folder, f'matrix_{args.potential}_Pth_{args.Ptheta}_Pph_{args.Pphi}_m_{args.M_1}_m_{args.M_2}_Ad_n_split_{args.split_idx}.npy'), Ad_n)
        numpy.save(os.path.join(folder, f'matrix_{args.potential}_Pth_{args.Ptheta}_Pph_{args.Pphi}_m_{args.M_1}_m_{args.M_2}_EPS_split_{args.split_idx}.npy'), EPS)


    else:
        
        numpy.save(os.path.join(folder, f'matrix_{args.potential}_Pth_{args.Ptheta}_Pph_{args.Pphi}_m_{args.M_1}_m_{args.M_2}_Ad_n.npy'), Ad_n)
        numpy.save(os.path.join(folder, f'matrix_{args.potential}_Pth_{args.Ptheta}_Pph_{args.Pphi}_m_{args.M_1}_m_{args.M_2}_EPS_split_{args.split_idx}.npy'), EPS)
        
        # BO energies and observables
        Hbo_new = +1/(2*H.mu12)*(-H.ddR2 + xp.diag(H.Pphi**2/H.R**2)+ xp.diag(H.Ptheta**2/H.R**2)+xp.diag(1/(2*H.R)**2)) +xp.diag(Ad_n)
        Ad_vn_new, Unv_bo = xp.linalg.eigh(Hbo_new)
        e_bo_new = xp.sort(Ad_vn_new.flatten())
        bo_new = e_bo_new[1] - e_bo_new[0]
        print("e_bo_new",e_bo_new[0:10])
        print("BO new vib gap",bo_new,flush=True)

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
        print("l200 BO: <chi0|r|chi0>:", l2BO_chi00)
        print("l201 BO: <chi1|r|chi0>:", l2BO_chi01)

        TeBO_RP = xp.zeros(rPS.shape, dtype=xp.complex128)
        TeBO_RP = TeBO[:,None]
        HTeBO = inverse_weyl_transform(TeBO_RP, H.shape[0], H.R, H.P_R)
        TeBO_chi00 = xp.sum(Unv_bo[:,0].conj()*(HTeBO@Unv_bo[:,0]))
        TeBO_chi01 = xp.sum(Unv_bo[:,1].conj()*(HTeBO@Unv_bo[:,0]))
        print("Te BO, 00, 01:", TeBO_chi00, TeBO_chi01 )


        EPS_bo = xp.zeros((H.shape[0], H.shape[0]))
        Helmat = xp.repeat(ival,H.shape[0],axis=1)
        EPS_bo += Helmat   
        EPS_bo += 1/(2*H.mu12)*(Pval**2+H.Pphi**2/Rval**2+H.Ptheta**2/Rval**2+1/(2*Rval)**2)
        HPS_bo = inverse_weyl_transform(EPS_bo, H.shape[0], H.R, H.P_R)
        EPSv_bo = batch_eigvalsh(HPS_bo)
        print("e_bo_new Weyl",EPSv_bo[0:10])
        print("Weyl BO vib gap",EPSv_bo[1]-EPSv_bo[0],flush=True)

        # PS energies and observables
        EPS += 1/(2*H.mu12)*(Pval**2+H.Pphi**2/Rval**2+H.Ptheta**2/Rval**2+1/(2*Rval)**2)
        HPS = inverse_weyl_transform_old(EPS, H.shape[0], H.R, H.P_R)
        EPSv = batch_eigvalsh(HPS)
        EPSv, UPSv = xp.linalg.eigh(HPS)
        print("e_bo_new Weyl",EPSv[0:10])
        print("PS vib gap",EPSv[1]-EPSv[0],flush=True)


        Hpe_x = inverse_weyl_transform(pPS[0], H.shape[0], H.R, H.P_R)
        Hpe_y = inverse_weyl_transform(pPS[1], H.shape[0], H.R, H.P_R)
        Hpe_z = inverse_weyl_transform(pPS[2], H.shape[0], H.R, H.P_R)
        Hpe_r = inverse_weyl_transform(pPS[3], H.shape[0], H.R, H.P_R)

        pe_chix = xp.sum(UPSv[:,1].conj()*(Hpe_x@UPSv[:,0]))
        pe_chiy = xp.sum(UPSv[:,1].conj()*(Hpe_y@UPSv[:,0]))
        pe_chiz = xp.sum(UPSv[:,1].conj()*(Hpe_z@UPSv[:,0]))
        pe_chir = xp.sum(UPSv[:,1].conj()*(Hpe_r@UPSv[:,0]))
        print("pe01 <chi_1|pe|chi0>:", pe_chix, pe_chiy, pe_chiz, pe_chir)
        

        pe_chix = UPSv[:,0].conj().T@Hpe_x@UPSv[:,0]
        pe_chiy = UPSv[:,0].conj().T@Hpe_y@UPSv[:,0]
        pe_chiz = UPSv[:,0].conj().T@Hpe_z@UPSv[:,0]
        pe_chir = UPSv[:,0].conj().T@Hpe_r@UPSv[:,0]
        print("pe00 <chi_0|pe|chi0>:", pe_chix, pe_chiy, pe_chiz, pe_chir)

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

        
        Hlxps = inverse_weyl_transform(lPS[0], H.shape[0], H.R, H.P_R)
        Hlyps = inverse_weyl_transform(lPS[1], H.shape[0], H.R, H.P_R)
        Hlzps = inverse_weyl_transform(lPS[2], H.shape[0], H.R, H.P_R)
        Hl2ps = inverse_weyl_transform(l2PS-1/H.mu12/Rval, H.shape[0], H.R, H.P_R)
        #  -1/H.mu12/Rval**2/4
        # with xp.printoptions(suppress=True, precision=4):
        #     print(l2PS[NR//2-2:NR//2+1, NR//2-2:NR//2+1])
        #     print(Hl2ps[:6,:6])
            # print(Rval**2[:6])

        l00x = xp.sum(UPSv[:,0].conj()*(Hlxps@UPSv[:,0]))
        l00y = xp.sum(UPSv[:,0].conj()*(Hlyps@UPSv[:,0]))
        l00z = xp.sum(UPSv[:,0].conj()*(Hlzps@UPSv[:,0]))

        l01x = xp.sum(UPSv[:,1].conj()*(Hlxps@UPSv[:,0]))
        l01y = xp.sum(UPSv[:,1].conj()*(Hlyps@UPSv[:,0]))
        l01z = xp.sum(UPSv[:,1].conj()*(Hlzps@UPSv[:,0]))

        l002 = xp.sum(UPSv[:,0].conj()*(Hl2ps@UPSv[:,0]))
        l012 = xp.sum(UPSv[:,1].conj()*(Hl2ps@UPSv[:,0]))

        print("l00 <chi_0 | l_xyz|chi_0>:", l00x, l00y, l00z)
        print("l01 < chi1| l_xyz|chi_0>", l01x, l01y, l01z)
        print("l200  <chi_0 | l^2 |chi_0>:", l002)
        print("l200  <chi_1 | l^2 |chi_0>:", l012)

        HTePS = inverse_weyl_transform(TePS, H.shape[0], H.R, H.P_R)
        TePS_chi00 = xp.sum(Unv_bo[:,0].conj()*(HTePS@Unv_bo[:,0]))
        TePS_chi01 = xp.sum(Unv_bo[:,1].conj()*(HTePS@Unv_bo[:,0]))
        print("Te PS, 00, 01:", TePS_chi00, TePS_chi01 )




        if args.evecs:
            numpy.savez(args.evecs, R=H.R, P=H.P_R, EPS=EPS, HPS=HPS,EPSv=EPSv, UPSv=UPSv, pPS=pPS, rPS=rPS, l2PS=l2PS)
        
        
        
        
        
        
        