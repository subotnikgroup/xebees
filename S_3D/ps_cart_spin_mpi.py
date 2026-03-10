from sys import stderr
import argparse as ap
from pathlib import Path

import concurrent.futures as cf
from itertools import product, chain
from functools import reduce, partial
import operator

import os, sys
sys.path.append(os.path.abspath("lib"))

import xp
import numpy as np # only use this for reading and writing objects
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
        'R', 'P_R', 'R_grid', 'RP_grid','_Efunc','P_x','P_y','P_z',
        'x', 'y', 'z','x_grid','y_grid','z_grid', 'xb_grid','yb_grid','zb_grid',
        'ddR2', 'ddx2','ddx1','ddy2','ddy1','ddz2','ddz1',
        'axes','Vgrid', '_preconditioner_data','Pg','Pphi','Ptheta',
        'shape','boshape','bospinshape','size','guess','k','mu12','_Vfunc',
        '_locked','max_threads','alpha','soc','sx','sy','sz','E1','E2','si'
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
        self.alpha = args.alpha
        
        self.soc = args.soc

        if not hasattr(args, "potential"):
            args.extent = 'soft_coulomb'

        if args.potential == 'borgis':
            print(f"Waring: All masses scaled to AMU for {args.potential}!")
            self.m_e *= AMU_TO_AU
            self.M_1 *= AMU_TO_AU
            self.M_2 *= AMU_TO_AU

        #print("M_1", self.M_1, "M_2", self.M_2, "m_e", self.m_e)

        self.mu   = xp.sqrt(self.M_1*self.M_2*self.m_e/(self.M_1+self.M_2+self.m_e))
        self.mur  = (self.M_1+self.M_2)*self.m_e/(self.M_1+self.M_2+self.m_e)
        self.mu12 = self.M_1*self.M_2/(self.M_1+self.M_2)
        self._Vfunc, extent_func, self._Efunc = {
            'soft_coulomb': (potentials.soft_coulomb, potentials.extents_soft_coulomb, None),
            'borgis': (partial(potentials.borgis, asymmetry_param=1), potentials.extents_borgis, potentials.Efield_borgis),
            'erf_coulomb':(potentials.erf_coulomb, potentials.extents_erf_coulomb, potentials.Efield_coulomb)
            }[args.potential]

        extent = extent_func(self.mu12)
        soc_const =  self.alpha/137**2/self.m_e**2/2 # alpha* g_e/c²me²/4
        print("soc const, alpha", soc_const, self.alpha)

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

        #print("R",self.R)
        #exit()

        self.axes = (self.R, self.x, self.y, self.z)

        self.shape = (args.NR, args.Nx, args.Ny, args.Nz)
        self.boshape = (args.Nx, args.Ny, args.Nz)
        self.bospinshape = (2,args.Nx, args.Ny, args.Nz)
        self.size = args.NR * args.Nx * args.Ny * args.Nz

        dR = self.R[1] - self.R[0]
        dx = self.x[1] - self.x[0]
        dy = self.y[1] - self.y[0]
        dz = self.z[1] - self.z[0]
        
        # P_R grid goes -(n-1)*2pi/dR ...0 ... +(n-1)*2pi/dR
        self.P_R  = xp.fft.fftshift(xp.fft.fftfreq(args.NR, dR)) * 2 * xp.pi
        self.RP_grid = xp.meshgrid(self.R, self.P_R, indexing='ij')
            # N.B.: These all lack the factor of -1/(2 * mu)
        # We also are throwing away the returned jacobian of R/r
        #self.ddR2, _ = KE_Borisov_3D(self.R, bare=True)
        #self.ddR2  = KE(args.NR, dR, bare=True, cyclic=False)
        self.ddR2  = KE_FFT(args.NR, self.P_R, self.R)
    
        self.ddx2 = KE(args.Nx, dx, bare=True, cyclic=False)
        #self.P_x  = xp.fft.fftshift(xp.fft.fftfreq(args.Nx, dx)) * 2 * xp.pi
        #self.ddx2 = KE_FFT(args.Nx, self.P_x, self.x)
        self.ddx1 = KE(args.Nx, dx, bare=True, cyclic=False, order=1) 

        self.ddy2 = KE(args.Ny, dy, bare=True, cyclic=False)
        #self.P_y  = xp.fft.fftshift(xp.fft.fftfreq(args.Ny, dy)) * 2 * xp.pi
        #self.ddy2 = KE_FFT(args.Ny, self.P_y, self.y)
        self.ddy1 = KE(args.Ny, dy, bare=True, cyclic=False, order=1)

        self.ddz2 = KE(args.Nz, dz, bare=True, cyclic=False)
        #self.P_z  = xp.fft.fftshift(xp.fft.fftfreq(args.Nz, dz)) * 2 * xp.pi
        #self.ddz2 = KE_FFT(args.Nz, self.P_z, self.z)
        self.ddz1 = KE(args.Nz, dz, bare=True, cyclic=False, order=1)
    
        self.R_grid, self.xb_grid, self.yb_grid, self.zb_grid = xp.meshgrid(self.R, self.x, self.y, self.z, indexing='ij')
        self.x_grid, self.y_grid, self.z_grid,  = xp.meshgrid(self.x, self.y, self.z, indexing='ij')
        self.Vgrid = self.V(self.R_grid, self.xb_grid, self.yb_grid, self.zb_grid)
        
        #only pauli matrices no hbar/2 term
        #self.sx = xp.array([[0,1],[1,0]])
        #self.sy = xp.array([[0,-1j],[1j,0]])
        #self.sz = xp.array([[1,0],[0,-1]])
        self.sx = xp.array([[1,0],[0,-1]])
        self.sy = xp.array([[0,-1j],[1j,0]])
        self.sz = xp.array([[0,1],[1,0]])
        self.si = xp.eye(2)

        self.E1, self.E2 = self.Efield(self.R_grid, self.xb_grid, self.yb_grid, self.zb_grid)
        
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

    def Efield(self, R, r_x, r_y, r_z):
        mu12 = self.mu12
        M_1 = self.M_1
        M_2 = self.M_2

        kappa2 = r_x*R

        r1e2 = r_x**2 +r_y**2 +r_z**2 + (R)**2*(mu12/M_1)**2 - 2*kappa2*mu12/M_1
        r2e2 = r_x**2 +r_y**2 +r_z**2 + (R)**2*(mu12/M_2)**2 + 2*kappa2*mu12/M_2

        r1e = xp.sqrt(xp.where(r1e2 < 0, 0, r1e2))
        r2e = xp.sqrt(xp.where(r2e2 < 0, 0, r2e2))

        return (self._Efunc(r1e,self.g_1), self._Efunc(r2e,self.g_2))

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
   
                
    def Tx(self,xdav):
        Hel_dav = -1/(2*self.mur)*(
            xp.einsum('ij,BSjkl->BSikl',self.ddx2,xdav,optimize=True)
            +xp.einsum('ij,BSkjl->BSkil',self.ddy2,xdav,optimize=True)
            +xp.einsum('ij,BSklj->BSkli',self.ddz2,xdav,optimize=True)
            )
        return Hel_dav.reshape(xdav.shape)

    def soc_full(self,xdav,soc_data_i):

        w_y_coef12, w_z_coef12, w_x_coef1, w_x_coef2 = soc_data_i
        x = xdav.reshape((-1,) + self.bospinshape)
        sx, sy, sz = self.sx, self.sy, self.sz
        Hsocdav = 0.5j * (
            - xp.einsum('sS,zc,BSxyz,xyz->Bsxyc', sx, self.ddz1, x, w_y_coef12, optimize=True)
            + xp.einsum('sS,yb,BSxyz,xyz->Bsxbz', sx, self.ddy1, x, w_z_coef12, optimize=True)
            - xp.einsum('sS,xa,BSxyz,xyz->Bsayz', sy, self.ddx1, x, w_z_coef12, optimize=True)
            + xp.einsum('sS,zc,BSxyz,xyz->Bsxyc', sy, self.ddz1, x, w_x_coef1, optimize=True)
            + xp.einsum('sS,zc,BSxyz,xyz->Bsxyc', sy, self.ddz1, x, w_x_coef2, optimize=True)
            - xp.einsum('sS,yb,BSxyz,xyz->Bsxbz', sz, self.ddy1, x, w_x_coef1, optimize=True)
            - xp.einsum('sS,yb,BSxyz,xyz->Bsxbz', sz, self.ddy1, x, w_x_coef2, optimize=True)
            + xp.einsum('sS,xa,BSxyz,xyz->Bsayz', sz, self.ddx1, x, w_y_coef12, optimize=True)
        )
        
        return Hsocdav.reshape(xdav.shape)


    def ps_ham(self,term1,term2,term3,coeffgammaerfy,coeffgammaerfz,Ri,soc_data_i):

        def Hx_ps(xdav):
            x = xdav.reshape((-1,)+self.bospinshape).astype(complex) 
            #print("x shape:", x.shape)
            if self.soc =='full':
                Hpsdav = (
                    self.Vgrid[Ri]*x + self.Tx(x) + self.soc_full(x, soc_data_i)
                    +xp.einsum('xayz,BSxyz->BSayz', ddx_terms, x, optimize=True) 
                    +xp.einsum('xybz,BSxyz->BSxbz', ddy_terms, x, optimize=True) 
                    +xp.einsum('xyzc,BSxyz->BSxyc', ddz_terms, x, optimize=True)
                    +xp.einsum('sS,xyz,BSxyz->Bsxyz',self.sz,coeffgammaerfy,x,optimize=True)
                    +xp.einsum('sS,xyz,BSxyz->Bsxyz',self.sy,coeffgammaerfz,x,optimize=True)
                )
                
            elif self.soc =='no_spin_erf':
                Hpsdav = (
                    self.Vgrid[Ri]*x + self.Tx(x) + self.soc_full(x, soc_data_i)
                    +xp.einsum('xayz,BSxyz->BSayz', ddx_terms, x, optimize=True) 
                    +xp.einsum('xybz,BSxyz->BSxbz', ddy_terms, x, optimize=True) 
                    +xp.einsum('xyzc,BSxyz->BSxyc', ddz_terms, x, optimize=True)
                )
            elif self.soc =='no_soc':
                Hpsdav = (
                    self.Vgrid[Ri]*x + self.Tx(x)
                    #+xp.einsum('xayz,BSxyz->BSayz', ddx_terms, x, optimize=True) 
                    #+xp.einsum('xybz,BSxyz->BSxbz', ddy_terms, x, optimize=True) 
                    #+xp.einsum('xyzc,BSxyz->BSxyc', ddz_terms, x, optimize=True)
                    #+xp.einsum('sS,xyz,BSxyz->Bsxyz',self.sz,coeffgammaerfy,x,optimize=True)
                    #+xp.einsum('sS,xyz,BSxyz->Bsxyz',self.sy,coeffgammaerfz,x,optimize=True)
                )
            return Hpsdav.reshape(xdav.shape)

        return Hx_ps

    def Hbo_dav(self,Ri,soc_data_i):

        def Hxbo(xdav):
            #print("xdav shape:", xdav.shape)
            x = xdav.reshape((-1,)+self.bospinshape)               
            if self.soc =='full':  
                #print("Ri",Ri)              
                Hbodav = (
                    self.Vgrid[Ri]*x + self.Tx(x) + self.soc_full(x, soc_data_i)
                )               
            elif self.soc == 'no_soc':
                #print("Ri",i)
                Hbodav = (
                    self.Vgrid[Ri]*x + self.Tx(x)
                ) 
            elif self.soc =='no_spin_erf':
                Hbodav = (
                    self.Vgrid[Ri]*x + self.Tx(x) + self.soc_full(x, soc_data_i)
                ) 

            return Hbodav.reshape(xdav.shape)
        return Hxbo

    def buildDiag(self,Ri):
        NR,Nx,Ny,Nz = self.shape
        ke  = xp.zeros([Nx,Ny,Nz],dtype=self.ddx2.dtype)
        ke += xp.diag(self.ddx2)[:,None,None]
        ke += xp.diag(self.ddy2)[None,:,None]
        ke += xp.diag(self.ddz2)[None,None,:]
        ke *= -1 / (2*self.mur)
        diag = self.Vgrid[Ri] + ke #XXXXXFix Vgrid
        diagravel = diag.ravel()
        diagspin = xp.append(diagravel,diagravel)
        return diagspin

    def BO_energies(self,sequence,guess_spin):
        
        NR,Nx,Ny,Nz = self.shape
    
        Ad_nsg = xp.zeros(NR)
        Ad_nse = xp.zeros(NR)
        ivalg = xp.zeros([NR,1])
        ivale = xp.zeros([NR,1])

        evecs_prev = True
        for i in sequence:
            print("Atom Ri idx",i, "Atom Ri",self.R[i],flush=True)
            diag = self.buildDiag(i)   
            if evecs_prev == True:
                guess_bo = guess_spin
                evecs_prev = False
            else:
                guess_bo = evecs
            print("guess_bo",guess_bo.shape)
            
            E1, E2 = self.Efield(self.R[i], self.x_grid, self.y_grid, self.z_grid)
            c1 = 0.5 * (1/137)**2 * E1 * self.alpha / (self.m_e**2)
            c2 = 0.5 * (1/137)**2 * E2 * self.alpha / (self.m_e**2)
            coef12 = c1 + c2
            coef1, coef2 = c1, c2
            xi, yi, zi = self.x, self.y, self.z
            mu12, M1, M2 = self.mu12, self.M_1, self.M_2
            w_y_coef12 = yi[None, :, None] * coef12
            w_z_coef12 = zi[None, None, :] * coef12
            w_x_coef1  = (xi - self.R[i]*mu12/M1)[:, None, None] * coef1
            w_x_coef2  = (xi + self.R[i]*mu12/M2)[:, None, None] * coef2
            soc_data_i = (w_y_coef12, w_z_coef12, w_x_coef1, w_x_coef2)
            

            conv, e_approx, evecs = lib.davidson1(
                self.Hbo_dav(i,soc_data_i),
                guess_bo,
                lambda dx, e, x0: dx/(diag-e+1e-5),
                nroots=args.k,
                max_cycle=args.iterations,
                verbose=args.verbosity,
                max_space=args.subspace,
                max_memory=get_davidson_mem(0.75),
                #tol=1e-12, #FIXME:DEBUG
                tol=1e-10
            )
            print("Davidson:", e_approx)
            print(conv)
            Ad_nsg[i] = e_approx[0]
            Ad_nse[i] = e_approx[1]
            ivalg[i,0] = e_approx[0]
            ivale[i,0] = e_approx[1]

        return Ad_nsg, Ad_nse, ivalg, ivale  
  
def Gamma_etf(H, Ridx, t1):
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


def Gamma_erf_spin_old(H,Ridx,t1, t2):
    J1xs = -0.5j*t1
    J1ys = -0.5j*t1
    J2xs = -0.5j*t2
    J2ys = -0.5j*t2
    gammaerf1ys = -1/H.R[Ridx]*(-J1ys-J2ys)
    gammaerf1zs = -1/H.R[Ridx]*(J1xs+J2xs)
    gammaerf2ys = -gammaerf1ys
    gammaerf2zs = -gammaerf1zs
    Gammaerfys = (H.M_2*gammaerf1ys-H.M_1*gammaerf2ys)/(H.M_1+H.M_2)
    Gammaerfzs = (H.M_2*gammaerf1zs-H.M_1*gammaerf2zs)/(H.M_1+H.M_2)
    print("Gammaerfys",Gammaerfys[:,0,0])
    print("Gammaerfys",Gammaerfys[0,:,0])
    print("Gammaerfys",Gammaerfys[0,0,:])
    print("Gammaerfzs",Gammaerfzs[:,0,0])
    print("Gammaerfzs",Gammaerfzs[0,:,0])
    print("Gammaerfzs",Gammaerfzs[0,0,:])
    #gammaspinerfs = (gammaerf1ys,gammaerf1zs,gammaerf2ys,gammaerf2zs)
    return Gammaerfys, Gammaerfzs


def Gamma_erf_spin(H,Ridx,t1,t2):
    gammaerfys = 1/ H.R[Ridx]*(-0.5j*xp.ones(H.boshape))
    gammaerfzs = -1/ H.R[Ridx]*(-0.5j*xp.ones(H.boshape))
    return gammaerfys, gammaerfzs


def Gamma_erf_orb_old(H,Ridx, gammaetf, t1, t2):
    (gammaetf1x, gammaetf1y, gammaetf1z, gammaetf2x, gammaetf2y, gammaetf2z) = gammaetf
    rx, ry, rz = H.x,H.y,H.z

    J1xa = ry[None,:,None,None]*gammaetf1z   
    J1xb = -(rz[None,None,None,:]*gammaetf1y)
    J1ya = (rz[None,None,None,:]*gammaetf1x)
    J1yb = -(rx[:,None,None,None]*gammaetf1z)
    J1yc = +(H.R[Ridx]*H.mu12/H.M_1)*gammaetf1z
    
    J2xa = (ry[None,:,None,None]*gammaetf2z)
    J2xb = -(rz[None,None,None,:]*gammaetf2y)
    J2ya = (rz[None,None,None,:]*gammaetf2x)
    J2yb = -(rx[:,None,None,None]*gammaetf2z)
    J2yc = +(H.R[Ridx]*H.mu12/H.M_2)*gammaetf2z

    gammaerf1ya = -1/H.R[Ridx]*(-J1ya-J2ya)
    gammaerf1yb = -1/H.R[Ridx]*(-J1yb-J2yb)
    gammaerf1yc = -1/H.R[Ridx]*(-J1yc-J2yc)
    gammaerf1za = -1/H.R[Ridx]*(J1xa+J2xa)
    gammaerf1zb = -1/H.R[Ridx]*(J1xb+J2xb)
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

    Jya = xp.zeros_like(Jya)
    Jza = xp.zeros_like(Jza) #z ok y not ok
    Jyb = xp.zeros_like(Jyb)#y ok z not ok
    Jzb = xp.zeros_like(Jzb)#z ok y not ok
    Jyc = xp.zeros_like(Jyc)#y ok z not ok
    Jzc = xp.zeros_like(Jzc)#y ok z not ok
    Jyd = xp.zeros_like(Jyd)#y ok z not ok
    Jzd = xp.zeros_like(Jzd)#z ok y not ok

    return Jya, Jyb, Jyc, Jyd, Jza, Jzb, Jzc, Jzd

def R_Gamma_exp_old(H,Ridx,evecs_save,gammavals):

    Nx, Ny, Nz = H.boshape
    (gammaetfy, gammaerfya, gammaerfyb, gammaerfyc, gammaerfza, gammaerfzb, gammaerfsy, gammaerfsz) = gammavals

    evecs_gs = evecs_save[0,:].reshape(2, Nx, Ny, Nz)
    evecs_conj_gs = xp.conj(evecs_gs)
    evecs_es = evecs_save[1,:].reshape(2, Nx, Ny, Nz)
    evecs_conj_es = xp.conj(evecs_es)

    gammaerfybc = gammaerfyb + gammaerfyc 
    Gamma_y_gs_etf = xp.einsum('sxyz,sS,xybz,Sxbz->', evecs_conj_gs, H.si, gammaetfy, evecs_gs, optimize=True) 
    Gamma_y_gs_erf_a = xp.einsum('sxyz,sS,xayz,Sayz->', evecs_conj_gs, H.si, gammaerfya, evecs_gs, optimize=True)
    Gamma_y_gs_erf_bc = xp.einsum('sxyz,sS,xyzc,Sxyc->', evecs_conj_gs,H.si, gammaerfybc, evecs_gs, optimize=True)
    Gamma_y_gs_erf_s = xp.einsum('sxyz,sS,xyz,Sxyz->', evecs_conj_gs,H.sy, gammaerfsy, evecs_gs, optimize=True) 

    Gamma_y_es_etf = xp.einsum('sxyz,sS,xybz,Sxbz->', evecs_conj_es, H.si, gammaetfy, evecs_es, optimize=True) 
    Gamma_y_es_erf_a = xp.einsum('sxyz,sS,xayz,Sayz->', evecs_conj_es, H.si, gammaerfya, evecs_es, optimize=True)
    Gamma_y_es_erf_bc = xp.einsum('sxyz,sS,xyzc,Sxyc->', evecs_conj_es,H.si, gammaerfybc, evecs_es, optimize=True)
    Gamma_y_es_erf_s = xp.einsum('sxyz,sS,xyz,Sxyz->', evecs_conj_es,H.sy, gammaerfsy, evecs_es, optimize=True) 

    Gamma_z_gs_etf = xp.einsum('sxyz,sS,xyzc,Sxyc->', evecs_conj_gs, H.si, gammaetfz, evecs_gs, optimize=True) 
    Gamma_z_gs_erf_a = xp.einsum('sxyz,sS,xyzc,Sayc->', evecs_conj_gs, H.si, gammaerfza, evecs_gs, optimize=True)
    Gamma_z_gs_erf_b = xp.einsum('sxyz,sS,xybz,Sxbz->', evecs_conj_gs,H.si, gammaerfzb, evecs_gs, optimize=True)
    Gamma_z_gs_erf_s = xp.einsum('sxyz,sS,xyz,Sxyz->', evecs_conj_gs,H.sx, gammaerfsz, evecs_gs, optimize=True) 

    Gamma_z_es_etf = xp.einsum('sxyz,sS,xyzc,Sxyc->', evecs_conj_es, H.si, gammaetfz, evecs_es, optimize=True) 
    Gamma_z_es_erf_a = xp.einsum('sxyz,sS,xyzc,Sayc->', evecs_conj_es, H.si, gammaerfza, evecs_es, optimize=True)
    Gamma_z_es_erf_b = xp.einsum('sxyz,sS,xybz,Sxbz->', evecs_conj_es,H.si, gammaerfzb, evecs_es, optimize=True)
    Gamma_z_es_erf_s = xp.einsum('sxyz,sS,xyz,Sxyz->', evecs_conj_es,H.sx, gammaerfsz, evecs_es, optimize=True) 

    Gamma_y_gs = Gamma_y_gs_etf + Gamma_y_gs_erf_a + Gamma_y_gs_erf_bc + Gamma_y_gs_erf_s
    Gamma_y_es = Gamma_y_es_etf + Gamma_y_es_erf_a + Gamma_y_es_erf_bc + Gamma_y_es_erf_s
    Gamma_z_gs = Gamma_z_gs_etf + Gamma_z_gs_erf_a + Gamma_z_gs_erf_b  + Gamma_z_gs_erf_s
    Gamma_z_es = Gamma_z_es_etf + Gamma_z_es_erf_a + Gamma_z_es_erf_b  + Gamma_z_es_erf_s

    R_Gamma_y_gs = 1j*H.R[Ridx]*Gamma_z_gs
    R_Gamma_y_es = 1j*H.R[Ridx]*Gamma_z_es
    R_Gamma_z_gs = -1j*H.R[Ridx]*Gamma_y_gs
    R_Gamma_z_es = -1j*H.R[Ridx]*Gamma_y_es
    
    return R_Gamma_y_gs, R_Gamma_y_es, R_Gamma_z_gs, R_Gamma_z_es

def R_Gamma_exp(H,Ridx,evecs_save,gammavals):

    Nx, Ny, Nz = H.boshape
    (gammaetfy, gammaetfz, Jya, Jyb, Jyc, Jyd, Jza, Jzb, Jzc, Jzd, gammaerfsy, gammaerfsz) = gammavals

    evecs_gs = evecs_save[0,:].reshape(2, Nx, Ny, Nz)
    evecs_conj_gs = xp.conj(evecs_gs)
    evecs_es = evecs_save[1,:].reshape(2, Nx, Ny, Nz)
    evecs_conj_es = xp.conj(evecs_es)

    print("norm evecs_gs",xp.linalg.norm(evecs_gs-evecs_es))

    
    Gamma_y_gs_etf_erf_acd = xp.einsum('sxyz,sS,xybz,Sxbz->', evecs_conj_gs, H.si, (gammaetfy +Jya- Jyc- Jyd), evecs_gs, optimize=True)
    #Gamma_y_gs_etf_erf_acd = xp.einsum('sxyz,sS,xybz,Sxbz->', evecs_conj_gs, H.si, (gammaetfy), evecs_gs, optimize=True)
    Gamma_y_gs_erf_b = xp.einsum('sxyz,sS,xayz,Sabz->', evecs_conj_gs, H.si, (Jyb), evecs_gs, optimize=True)
    Gamma_y_gs_serf = xp.einsum('sxyz,sS,xyz,Sxyz->', evecs_conj_gs,H.sz, gammaerfsy, evecs_gs, optimize=True) 

    Gamma_z_gs_etf_erf_bcd = xp.einsum('sxyz,sS,xyzc,Sxyc->', evecs_conj_gs, H.si, (gammaetfz-Jzb+Jzc+Jzd), evecs_gs, optimize=True)
    Gamma_z_gs_erf_a = xp.einsum('sxyz,sS,xayz,Sayz->', evecs_conj_gs, H.si, (Jza), evecs_gs, optimize=True)
    Gamma_z_gs_serf = xp.einsum('sxyz,sS,xyz,Sxyz->', evecs_conj_gs,H.sy, gammaerfsz, evecs_gs, optimize=True) 

    Gamma_y_es_etf_erf_acd = xp.einsum('sxyz,sS,xybz,Sxbz->', evecs_conj_es, H.si, (gammaetfy +Jya- Jyc- Jyd), evecs_es, optimize=True)
    Gamma_y_es_erf_b = xp.einsum('sxyz,sS,xayz,Sabz->', evecs_conj_es, H.si, (Jyb), evecs_es, optimize=True)
    Gamma_y_es_serf = xp.einsum('sxyz,sS,xyz,Sxyz->', evecs_conj_es,H.sz, gammaerfsy, evecs_es, optimize=True) 

    Gamma_z_es_etf_erf_bcd = xp.einsum('sxyz,sS,xyzc,Sxyc->', evecs_conj_es, H.si, (gammaetfz-Jzb+Jzc+Jzd), evecs_es, optimize=True)
    Gamma_z_es_erf_a = xp.einsum('sxyz,sS,xayz,Sayz->', evecs_conj_es, H.si, (Jza), evecs_es, optimize=True)
    Gamma_z_es_serf = xp.einsum('sxyz,sS,xyz,Sxyz->', evecs_conj_es,H.sy, gammaerfsz, evecs_es, optimize=True) 
    
    Gamma_y_gs = Gamma_y_gs_etf_erf_acd + Gamma_y_gs_erf_b + Gamma_y_gs_serf
    Gamma_z_gs = Gamma_z_gs_etf_erf_bcd + Gamma_z_gs_erf_a + Gamma_z_gs_serf
    Gamma_y_es = Gamma_y_es_etf_erf_acd + Gamma_y_es_erf_b + Gamma_y_es_serf
    Gamma_z_es = Gamma_z_es_etf_erf_bcd + Gamma_z_es_erf_a + Gamma_z_es_serf

    #Gamma_y_gs = Gamma_y_gs_etf_erf_acd
    #Gamma_z_gs = Gamma_z_gs_etf_erf_bcd
    #Gamma_y_es = Gamma_y_es_etf_erf_acd
    #Gamma_z_es = Gamma_z_es_etf_erf_bcd
    
    R_Gamma_y_gs = 1j*H.R[Ridx]*Gamma_z_gs
    R_Gamma_y_es = 1j*H.R[Ridx]*Gamma_z_es
    R_Gamma_z_gs = -1j*H.R[Ridx]*Gamma_y_gs
    R_Gamma_z_es = -1j*H.R[Ridx]*Gamma_y_es
    
    return R_Gamma_y_gs, R_Gamma_y_es, R_Gamma_z_gs, R_Gamma_z_es



def exp_l_s(H,evecs_save,Ridx,t1,t2):
    Nx, Ny, Nz = H.boshape
    #cosgamma = x_grid/xp.sqrt(x_grid**2 + y_grid**2 + z_grid**2)
    #singamma = y_grid/xp.sqrt(x_grid**2 + y_grid**2 + z_grid**2)
    evecs_gs = evecs_save[0,:].reshape(2, Nx, Ny, Nz)
    evecs_conj_gs = xp.conj(evecs_gs)
    evecs_es = evecs_save[1,:].reshape(2, Nx, Ny, Nz)
    evecs_conj_es = xp.conj(evecs_es)

    exp_lx_gs_a = xp.einsum('sxyz,y,zc,sxyc->',evecs_conj_gs, H.y,H.ddz1, evecs_gs, optimize=True)
    exp_lx_gs_b = xp.einsum('sxyz,z,yb,sxbz->',evecs_conj_gs, H.z,H.ddy1, evecs_gs, optimize=True)
    exp_lx_gs = exp_lx_gs_a - exp_lx_gs_b

    exp_ly_gs_a = xp.einsum('sxyz,z,xa,sayz->',evecs_conj_gs, H.z, H.ddx1, evecs_gs, optimize=True)
    exp_ly_gs_b = xp.einsum('sxyz,x,zc,sxyc->',evecs_conj_gs, H.x, H.ddz1, evecs_gs, optimize=True)
    exp_ly_gs_c = H.R[Ridx]/(2*(H.M_1+H.M_2))*xp.einsum('sxyz,xyz,zc,sxyc->',evecs_conj_gs, (H.M_2*t1-H.M_1*t2),H.ddz1, evecs_gs, optimize=True)
    exp_ly_gs_d = H.R[Ridx]/(2*(H.M_1+H.M_2))*xp.einsum('sxyz,zc,xyc,sxyc->',evecs_conj_gs, H.ddz1,(H.M_2*t1-H.M_1*t2), evecs_gs, optimize=True)
    exp_ly_gs = exp_ly_gs_a - exp_ly_gs_b + exp_ly_gs_c + exp_ly_gs_d

    exp_lz_gs_a = xp.einsum('sxyz,x,yb,sxbz->',evecs_conj_gs, H.x,H.ddy1, evecs_gs, optimize=True)
    exp_lz_gs_b = xp.einsum('sxyz,y,xa,sayz->',evecs_conj_gs, H.y,H.ddx1, evecs_gs, optimize=True)
    exp_lz_gs_c = H.R[Ridx]/(2*(H.M_1+H.M_2))*xp.einsum('sxyz,xyz,yb,sxbz->',evecs_conj_gs, (H.M_2*t1-H.M_1*t2),H.ddy1, evecs_gs, optimize=True)
    exp_lz_gs_d = H.R[Ridx]/(2*(H.M_1+H.M_2))*xp.einsum('sxyz,yb,xbz,sxbz->',evecs_conj_gs, H.ddy1, (H.M_2*t1-H.M_1*t2), evecs_gs, optimize=True)
    exp_lz_gs = exp_lz_gs_a - exp_lz_gs_b - exp_lz_gs_c - exp_lz_gs_d

    exp_lx_es_a = xp.einsum('sxyz,y,zc,sxyc->',evecs_conj_es, H.y,H.ddz1, evecs_es, optimize=True)
    exp_lx_es_b = xp.einsum('sxyz,z,yb,sxbz->',evecs_conj_es, H.z,H.ddy1, evecs_es, optimize=True)
    exp_lx_es = exp_lx_es_a - exp_lx_es_b

    exp_ly_es_a = xp.einsum('sxyz,z,xa,sayz->',evecs_conj_es, H.z, H.ddx1, evecs_es, optimize=True)
    exp_ly_es_b = xp.einsum('sxyz,x,zc,sxyc->',evecs_conj_es, H.x, H.ddz1, evecs_es, optimize=True)
    exp_ly_es_c = H.R[Ridx]/(2*(H.M_1+H.M_2))*xp.einsum('sxyz,xyz,zc,sxyc->',evecs_conj_es, (H.M_2*t1-H.M_1*t2),H.ddz1, evecs_es, optimize=True)
    exp_ly_es_d = H.R[Ridx]/(2*(H.M_1+H.M_2))*xp.einsum('sxyz,zc,xyc,sxyc->',evecs_conj_es, H.ddz1,(H.M_2*t1-H.M_1*t2), evecs_es, optimize=True)
    exp_ly_es = exp_ly_es_a - exp_ly_es_b + exp_ly_es_c + exp_ly_es_d

    exp_lz_es_a = xp.einsum('sxyz,x,yb,sxbz->',evecs_conj_es, H.x,H.ddy1, evecs_es, optimize=True)
    exp_lz_es_b = xp.einsum('sxyz,y,xa,sayz->',evecs_conj_es, H.y,H.ddx1, evecs_es, optimize=True)
    exp_lz_es_c = H.R[Ridx]/(2*(H.M_1+H.M_2))*xp.einsum('sxyz,xyz,yb,sxbz->',evecs_conj_es, (H.M_2*t1-H.M_1*t2),H.ddy1, evecs_es, optimize=True)
    exp_lz_es_d = H.R[Ridx]/(2*(H.M_1+H.M_2))*xp.einsum('sxyz,yb,xbz,sxbz->',evecs_conj_es, H.ddy1, (H.M_2*t1-H.M_1*t2), evecs_es, optimize=True)
    exp_lz_es = exp_lz_es_a - exp_lz_es_b - exp_lz_es_c - exp_lz_es_d

    exp_l = (1j*exp_lx_gs, 1j*exp_lx_es, 1j*exp_ly_gs, 1j*exp_ly_es, 1j*exp_lz_gs, 1j*exp_lz_es)


    exp_sx_gs = xp.einsum('sxyz,sS,Sxyz->',evecs_conj_gs, H.sx, evecs_gs, optimize=True)
    exp_sx_es = xp.einsum('sxyz,sS,Sxyz->',evecs_conj_es, H.sx, evecs_es, optimize=True)
    exp_sy_gs = xp.einsum('sxyz,sS,Sxyz->',evecs_conj_gs, H.sy, evecs_gs, optimize=True)
    exp_sy_es = xp.einsum('sxyz,sS,Sxyz->',evecs_conj_es, H.sy, evecs_es, optimize=True)
    exp_sz_gs = xp.einsum('sxyz,sS,Sxyz->',evecs_conj_gs, H.sz, evecs_gs, optimize=True)
    exp_sz_es = xp.einsum('sxyz,sS,Sxyz->',evecs_conj_es, H.sz, evecs_es, optimize=True)

    exp_sx_gs = xp.einsum('sxyz,sS,Sxyz->',evecs_conj_gs, H.sx, evecs_gs, optimize=True)
    exp_sx_es = xp.einsum('sxyz,sS,Sxyz->',evecs_conj_es, H.sx, evecs_es, optimize=True)
    exp_sy_gs = xp.einsum('sxyz,sS,Sxyz->',evecs_conj_gs, H.sy, evecs_gs, optimize=True)
    exp_sy_es = xp.einsum('sxyz,sS,Sxyz->',evecs_conj_es, H.sy, evecs_es, optimize=True)
    exp_sz_gs = xp.einsum('sxyz,sS,Sxyz->',evecs_conj_gs, H.sz, evecs_gs, optimize=True)
    exp_sz_es = xp.einsum('sxyz,sS,Sxyz->',evecs_conj_es, H.sz, evecs_es, optimize=True)

    exp_s = (exp_sx_gs, exp_sx_es, exp_sy_gs, exp_sy_es, exp_sz_gs, exp_sz_es)

    return exp_l, exp_s

def parse_args():
    parser = ap.ArgumentParser(
        prog='3body-3D',
        description="computes the lowest k eigenvalues of a 3-body potential in 3D")

    def odd_int(s):
        v = int(s)
        if v % 2 != 1:
            raise ap.ArgumentTypeError(f'NR must be odd, got {v}')
        return v

    class NumpyArrayAction(ap.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            setattr(namespace, self.dest, xp.array(values, dtype=float))

    parser.add_argument('-k', metavar='num_eigenvalues', default=5, type=int)
    parser.add_argument('-t', metavar="num_threads", default=1, type=int)
    parser.add_argument('-g_1', metavar='g_1', required=True, type=float)
    parser.add_argument('-g_2', metavar='g_2', required=True, type=float)
    parser.add_argument('-M_1', required=True, type=float)
    parser.add_argument('-M_2', required=True, type=float)
    parser.add_argument('-splits', default=0, type=int)
    parser.add_argument('-split_idx', default=1, type=int)
    parser.add_argument('-Pphi', required=True, type=float)
    parser.add_argument('-Ptheta', required=True, type=float)
    parser.add_argument('-alpha', default=0, type=float)
    parser.add_argument('-R', dest="NR", metavar="NR", default=101, type=odd_int)
    parser.add_argument('-x', dest="Nx", metavar="Nx", default=250, type=int)
    parser.add_argument('-y', dest="Ny", metavar="Ny", default=250, type=int)
    parser.add_argument('-z', dest="Nz", metavar="Nz", default=250, type=int)
    parser.add_argument('--bo_spectrum', metavar='bo_spectrum', default=False, type=bool)
    parser.add_argument('-J', required=True, type=float)
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
    parser.add_argument('--soc', choices=['no_spin_erf','no_soc','lazy','full'], type=str, default='full')
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
    folder = os.getcwd()
    

    H = Hamiltonian(args)
    start_script = perf_counter()   
    NR,Nx,Ny,Nz = H.shape
    Nelec = 2*Nx*Ny*Nz 
    
    Ad_nsg = xp.zeros(NR)
    Ad_nse = xp.zeros(NR)
    ivalg = xp.zeros([NR,1])
    ivale = xp.zeros([NR,1])
    energy_bo = xp.zeros([NR,args.k])
    EPSg = xp.zeros((NR, NR))
    EPSe = xp.zeros((NR, NR))


    #print("HPS",inverse_weyl_transform(xp.ones((NR, NR)),NR,H.R,H.P_R))
    #exit()

    exp_RxGamma_y_gs = xp.zeros((NR, NR))
    exp_RxGamma_y_es = xp.zeros((NR, NR))
    exp_RxGamma_z_gs = xp.zeros((NR, NR))
    exp_RxGamma_z_es = xp.zeros((NR, NR))
    exp_sx_gs = xp.zeros((NR, NR))
    exp_sx_es = xp.zeros((NR, NR))
    exp_sy_gs = xp.zeros((NR, NR))
    exp_sy_es = xp.zeros((NR, NR))
    exp_sz_gs = xp.zeros((NR, NR))
    exp_sz_es = xp.zeros((NR, NR))
    exp_lx_gs = xp.zeros((NR, NR))
    exp_lx_es = xp.zeros((NR, NR))
    exp_ly_gs = xp.zeros((NR, NR))
    exp_ly_es = xp.zeros((NR, NR))
    exp_lz_gs = xp.zeros((NR, NR))
    exp_lz_es = xp.zeros((NR, NR))

    
    Rval, Pval = H.RP_grid
    gammacoeff_R = -1j*(Pval-1/Rval)/H.mu12 
    gammacoeff_phi = +1j*(H.Pphi/H.R)/H.mu12
    gammacoeff_theta = +1j*(H.Ptheta/H.R-1/H.R)/H.mu12


    def generalized_sequence(NR, num_splits,split_idx):
        nodes = xp.linspace(0, NR, num_splits + 1, dtype=xp.int32).tolist()
    
        parts = []
        midpoint_idx = num_splits // 2
    
        for i in range(num_splits):
          start = nodes[i]
          end = nodes[i+1]
          
          if i < midpoint_idx:
            if i == 0:
                chunk = np.arange(end, start - 1, -1)
            else:
                chunk = np.arange(end, start, -1)

          else:
            if i == num_splits - 1:
                chunk = np.arange(start+1, end)
            else:
              chunk = (np.arange(start + 1, end + 1))

          parts.append(chunk)

        return parts[split_idx-1]

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
    #sequence = np.array(range(NR//4,-1,-1),range(NR//2,NR//4,-1),range(NR//2+1,3*NR//4+1),range(3*NR//4+1,NR))
    #print("sequence",list(sequence))
    jR = NR//2
    ps_sequence = list( chain(
            [jR],
            range(jR - 1, -1, -1),
            range(jR + 1, NR)))
    #print("ps_sequence",list(ps_sequence))
    gammacoeff = (gammacoeff_R, gammacoeff_phi, gammacoeff_theta)

    

    evecs_prev = True
    guess_ns = xp.exp(-(H.Vgrid[iR] - xp.min(H.Vgrid[iR]))**2/27.211**2).ravel()
    guess_zeros = xp.zeros(len(guess_ns))
    guess_spin = xp.array([xp.append(guess_ns, guess_zeros),xp.append(guess_zeros, guess_ns)])

    if (args.bo_spectrum==True):
        Ad_nsg, Ad_nse, ivalg, ivale = H.BO_energies(sequence,guess_spin)
        Hbo_g = +1/(2*H.mu12)*(-H.ddR2 + xp.diag(H.Pphi**2/H.R**2)+ xp.diag(H.Ptheta**2/H.R**2)+xp.diag(1/(2*H.R)**2)) +xp.diag(Ad_nsg)
        Ad_vn_g = batch_eigvalsh(Hbo_g)
        e_bo_g = xp.sort(Ad_vn_g.flatten())
        print("e_bo_new g.s.",e_bo_g[0:10])
        bo_vib_ggap = e_bo_g[1] - e_bo_g[0]
        print("BO new vib gap g.s.",bo_vib_ggap,flush=True)

        Hbo_e = +1/(2*H.mu12)*(-H.ddR2 + xp.diag(H.Pphi**2/H.R**2)+ xp.diag(H.Ptheta**2/H.R**2)+xp.diag(1/(2*H.R)**2)) +xp.diag(Ad_nse)
        Ad_vn_e = batch_eigvalsh(Hbo_e)
        e_bo_e = xp.sort(Ad_vn_e.flatten())
        print("e_bo_new e.s.",e_bo_e[0:10])
        bo_vib_egap = e_bo_e[1] - e_bo_e[0]
        print("BO new vib gap e.s.",bo_vib_egap,flush=True)

        EPS_bog = xp.zeros((H.shape[0], H.shape[0]))
        Helmatg = xp.repeat(ivalg,H.shape[0],axis=1)
        EPS_bog += Helmatg   
        EPS_bog += 1/(2*H.mu12)*(Pval**2+H.Pphi**2/Rval**2+H.Ptheta**2/Rval**2+1/(2*Rval)**2)
        HPS_bog = inverse_weyl_transform(EPS_bog, H.shape[0], H.R, H.P_R)
        EPSv_bog = batch_eigvalsh(HPS_bog)
        print("EPSv_bo",EPSv_bog[0:10])
    
        EPS_boe = xp.zeros((H.shape[0], H.shape[0]))
        Helmate = xp.repeat(ivale,H.shape[0],axis=1)
        EPS_boe += Helmate   
        EPS_boe += 1/(2*H.mu12)*(Pval**2+H.Pphi**2/Rval**2+H.Ptheta**2/Rval**2+1/(2*Rval)**2)
        HPS_boe = inverse_weyl_transform(EPS_boe, H.shape[0], H.R, H.P_R)
        EPSv_boe = batch_eigvalsh(HPS_boe)
        print("EPSv_bo",EPSv_boe[0:10])
        
        exit()

    

    with timer_ctx(f"R for loop"):
        for i in sequence:
            print("Atom Ri idx",i, "Atom Ri",H.R[i],flush=True)
            diag = H.buildDiag(i)               
            if evecs_prev == True:
                guess_bo = guess_spin
            else:
                guess_bo = evecs
            E1, E2 = H.Efield(H.R[i], H.x_grid, H.y_grid, H.z_grid)
            c1 = 0.5 * (1/137)**2 * E1 * H.alpha / (H.m_e**2)
            c2 = 0.5 * (1/137)**2 * E2 * H.alpha / (H.m_e**2)
            coef12 = c1 + c2
            coef1, coef2 = c1, c2
            xi, yi, zi = H.x, H.y, H.z
            mu12, M1, M2 = H.mu12, H.M_1, H.M_2
            
            w_y_coef12 = yi[None, :,None] * coef12
            w_z_coef12 = zi[None, None, :] * coef12
            w_x_coef1  = (xi - H.R[i]*mu12/M1)[:, None, None] * coef1
            w_x_coef2  = (xi + H.R[i]*mu12/M2)[:, None, None] * coef2
            soc_data_i = (w_y_coef12, w_z_coef12, w_x_coef1, w_x_coef2)

            conv, e_approx, evecs = lib.davidson1(
                H.Hbo_dav(i,soc_data_i),
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
            
            Ad_nsg[i] = e_approx[0]
            Ad_nse[i] = e_approx[1]
            ivalg[i,0] = e_approx[0]
            ivale[i,0] = e_approx[1]
    
            r1e2, r2e2 = H.V(H.R[i], H.x_grid, H.y_grid, H.z_grid, spitvals=True)
            theta1 = xp.exp(-r1e2)
            theta2 = xp.exp(-r2e2)
            partition = (theta1 + theta2)


            t1 = 1/(1+xp.exp(r1e2-r2e2))
            t2 = 1/(1+xp.exp(r2e2-r1e2))
            
            #t1 = 1/(1+xp.exp(-r2e2)/(xp.exp(-r1e2)**2))
            #t2 = 1/(1+xp.exp(-r1e2)/(xp.exp(-r2e2)**2))
            gammaetf1x, gammaetf1y, gammaetf1z = Gamma_etf(H,i, t1)
            gammaetf2x, gammaetf2y, gammaetf2z = Gamma_etf(H,i, t2)

            Jya, Jyb, Jyc, Jyd, Jza, Jzb, Jzc, Jzd = Gamma_erf_orb(H,i, t1, t2)
            gammaerfsy, gammaerfsz = Gamma_erf_spin(H, i, t1, t2)
            gammaetfx = (H.M_2*gammaetf1x-H.M_1*gammaetf2x)/(H.M_1+H.M_2)
            gammaetfy = (H.M_2*gammaetf1y-H.M_1*gammaetf2y)/(H.M_1+H.M_2)
            gammaetfz = (H.M_2*gammaetf1z-H.M_1*gammaetf2z)/(H.M_1+H.M_2)
            
            ddy_terms = (                    
                    gammacoeff_phi[i] * (Jya- Jyc- Jyd+gammaetfy)
                )

            ddz_terms = (
                    gammacoeff_theta[i]*(-Jzb+Jzc+Jzd+gammaetfz)  
                )

            

            coeffgammaerfy = gammacoeff_phi[i]*gammaerfsy
            coeffgammaerfz = gammacoeff_theta[i]*gammaerfsz


            with timer_ctx(f"P for loop"):
                #Pseq = [NR//2 -i for i in range(NR//2+1)] + [NR//2+i+1 for i in range(NR//2-1)]
                #print("Pseq", Pseq)
                for j in ps_sequence:
                    print("Atom Ri",i,"Atom Pj",j,flush=True)

                    if evecs_prev == True and j==NR//2:
                        guess_ps = evecs
                        evecs_prev = False
                    else:
                        guess_ps = evecs_save
                    ddx_terms = (gammacoeff_R[i,j] * gammaetfx+ gammacoeff_phi[i] * Jyb +
                                 gammacoeff_theta[i]* (Jza))
            
                    
                    with timer_ctx(f"Davidson of size {H.size}"):
                        conv, e_ps_approx, evecs_save = lib.davidson1(
                            H.ps_ham(ddx_terms,ddy_terms,ddz_terms,coeffgammaerfy,coeffgammaerfz,i,soc_data_i),
                            guess_ps,
                            lambda dx, e, x0: dx/(diag-e+1e-5),
                            nroots=args.k,
                            max_cycle=args.iterations,
                            verbose=args.verbosity,
                            max_space=args.subspace,
                            max_memory=get_davidson_mem(0.75),
                            #tol=1e-12, #FIXME:DEBUG
                            tol=1e-10,
                        )
    
                    print("Davidson:", e_ps_approx)
                    print(conv)
                    EPSg[i, j] = e_ps_approx[0]
                    EPSe[i, j] = e_ps_approx[1]
                    #print("evecs_save",evecs_save[xp.where((xp.abs(xp.imag(evecs_save))>1e-6))])
                    
                    #gammavals = (gammaetfy, gammaerfya, gammaerfyb, gammaerfyc, gammaerfza, gammaerfzb, gammaerfsy, gammaerfsz)
                    gammavals = (gammaetfy, gammaetfz, Jya, Jyb, Jyc, Jyd, Jza, Jzb, Jzc, Jzd, gammaerfsy, gammaerfsz)
                    R_Gamma_y_gs, R_Gamma_y_es, R_Gamma_z_gs, R_Gamma_z_es = R_Gamma_exp(H,i,evecs_save,gammavals)
                    print("R_Gamma_y_gs",R_Gamma_y_gs)
                    print("R_Gamma_y_es",R_Gamma_y_es)
                    print("R_Gamma_z_gs",R_Gamma_z_gs)
                    print("R_Gamma_z_es",R_Gamma_z_es)
                    exp_RxGamma_y_gs[i,j] = R_Gamma_y_gs.real
                    exp_RxGamma_y_es[i,j] = R_Gamma_y_es.real
                    exp_RxGamma_z_gs[i,j] = R_Gamma_z_gs.real
                    exp_RxGamma_z_es[i,j] = R_Gamma_z_es.real

                    if (R_Gamma_y_es.imag > 1e-10 or R_Gamma_y_gs.imag > 1e-10 or R_Gamma_z_es.imag > 1e-10 or R_Gamma_z_gs.imag > 1e-10):
                        print("R_Gamma_y_es",R_Gamma_y_es)
                        print("R_Gamma_y_gs",R_Gamma_y_gs)
                        print("R_Gamma_z_es",R_Gamma_z_es)
                        print("R_Gamma_z_gs",R_Gamma_z_gs)
                        #exit()

                    exp_l, exp_s = exp_l_s(H,evecs_save,i,t1,t2)
                    lx_gs, lx_es, ly_gs, ly_es, lz_gs, lz_es = exp_l
                    sx_gs, sx_es, sy_gs, sy_es, sz_gs, sz_es = exp_s


                    if (sx_es.imag > 1e-10 or sx_gs.imag > 1e-10 or sy_es.imag > 1e-10 or sy_gs.imag > 1e-10 or sz_es.imag > 1e-10 or sz_gs.imag > 1e-10):
                        print("sx_es",sx_es)
                        print("sx_gs",sx_gs)
                        print("sy_es",sy_es)
                        print("sy_gs",sy_gs)
                        print("sz_es",sz_es)
                        print("sz_gs",sz_gs)
                        print("lx_es",lx_es)
                        print("lx_gs",lx_gs)
                        print("ly_es",ly_es)
                        print("ly_gs",ly_gs)
                        print("lz_es",lz_es)
                        print("lz_gs",lz_gs)
                        
                    exp_sx_gs[i,j] = sx_gs.real
                    exp_sx_es[i,j] = sx_es.real
                    exp_sy_gs[i,j] = sy_gs.real
                    exp_sy_es[i,j] = sy_es.real
                    exp_sz_gs[i,j] = sz_gs.real
                    exp_sz_es[i,j] = sz_es.real

                    exp_lx_gs[i,j] = lx_gs.real
                    exp_lx_es[i,j] = lx_es.real
                    exp_ly_gs[i,j] = ly_gs.real
                    exp_ly_es[i,j] = ly_es.real
                    exp_lz_gs[i,j] = lz_gs.real
                    exp_lz_es[i,j] = lz_es.real

                    print("sx_es",sx_es)
                    print("sx_gs",sx_gs)
                    print("sy_es",sy_es)
                    print("sy_gs",sy_gs)
                    print("sz_es",sz_es)
                    print("sz_gs",sz_gs)
                    print("lx_es",lx_es)
                    print("lx_gs",lx_gs)
                    print("ly_es",ly_es)
                    print("ly_gs",ly_gs)
                    print("lz_es",lz_es)
                    print("lz_gs",lz_gs)

                    

    if args.splits > 0:

        np.save(os.path.join(folder, f'matrix_spin_{args.potential}_j_{args.J}_Pth_{args.Ptheta}_Pph_{args.Pphi}_a_{args.alpha}_m_{args.M_2}_Ad_nsg_split_{args.split_idx}.npy'), Ad_nsg)
        np.save(os.path.join(folder, f'matrix_spin_{args.potential}_j_{args.J}_Pth_{args.Ptheta}_Pph_{args.Pphi}_a_{args.alpha}_m_{args.M_2}_Ad_nse_split_{args.split_idx}.npy'), Ad_nse)
        np.save(os.path.join(folder, f'matrix_spin_{args.potential}_j_{args.J}_Pth_{args.Ptheta}_Pph_{args.Pphi}_a_{args.alpha}_m_{args.M_2}_EPSg_split_{args.split_idx}.npy'), EPSg)
        np.save(os.path.join(folder, f'matrix_spin_{args.potential}_j_{args.J}_Pth_{args.Ptheta}_Pph_{args.Pphi}_a_{args.alpha}_m_{args.M_2}_EPSe_split_{args.split_idx}.npy'), EPSe)
        np.save(os.path.join(folder, f'matrix_spin_{args.potential}_j_{args.J}_Pth_{args.Ptheta}_Pph_{args.Pphi}_a_{args.alpha}_m_{args.M_2}_exp_RxGamma_y_gs_split_{args.split_idx}.npy'), exp_RxGamma_y_gs)
        np.save(os.path.join(folder, f'matrix_spin_{args.potential}_j_{args.J}_Pth_{args.Ptheta}_Pph_{args.Pphi}_a_{args.alpha}_m_{args.M_2}_exp_RxGamma_y_es_split_{args.split_idx}.npy'), exp_RxGamma_y_es)
        np.save(os.path.join(folder, f'matrix_spin_{args.potential}_j_{args.J}_Pth_{args.Ptheta}_Pph_{args.Pphi}_a_{args.alpha}_m_{args.M_2}_exp_RxGamma_z_gs_split_{args.split_idx}.npy'), exp_RxGamma_Z_gs)
        np.save(os.path.join(folder, f'matrix_spin_{args.potential}_j_{args.J}_Pth_{args.Ptheta}_Pph_{args.Pphi}_a_{args.alpha}_m_{args.M_2}_exp_RxGamma_z_es_split_{args.split_idx}.npy'), exp_RxGamma_Z_es)

    else:
        
        np.save(os.path.join(folder, f'matrix3D_{args.potential}_j_{args.J}_Pth_{args.Ptheta}_Pph_{args.Pphi}_a_{args.alpha}_m_{args.M_2}_EPSg_split_{args.split_idx}.npy'), EPSg)
        np.save(os.path.join(folder, f'matrix3D_{args.potential}_j_{args.J}_Pth_{args.Ptheta}_Pph_{args.Pphi}_a_{args.alpha}_m_{args.M_2}_EPSe_split_{args.split_idx}.npy'), EPSe)
        np.save(os.path.join(folder, f'matrix_spin_{args.potential}_j_{args.J}_Pth_{args.Ptheta}_Pph_{args.Pphi}_a_{args.alpha}_m_{args.M_2}_exp_RxGamma_y_gs_split_{args.split_idx}.npy'), exp_RxGamma_y_gs)
        np.save(os.path.join(folder, f'matrix_spin_{args.potential}_j_{args.J}_Pth_{args.Ptheta}_Pph_{args.Pphi}_a_{args.alpha}_m_{args.M_2}_exp_RxGamma_y_es_split_{args.split_idx}.npy'), exp_RxGamma_y_es)
        np.save(os.path.join(folder, f'matrix_spin_{args.potential}_j_{args.J}_Pth_{args.Ptheta}_Pph_{args.Pphi}_a_{args.alpha}_m_{args.M_2}_exp_RxGamma_z_gs_split_{args.split_idx}.npy'), exp_RxGamma_z_gs)
        np.save(os.path.join(folder, f'matrix_spin_{args.potential}_j_{args.J}_Pth_{args.Ptheta}_Pph_{args.Pphi}_a_{args.alpha}_m_{args.M_2}_exp_RxGamma_z_es_split_{args.split_idx}.npy'), exp_RxGamma_z_es)

        Hbo_g = +1/(2*H.mu12)*(-H.ddR2 + xp.diag(H.Pphi**2/H.R**2)+ xp.diag(H.Ptheta**2/H.R**2)+xp.diag(1/(2*H.R)**2)) +xp.diag(Ad_nsg)
        Ad_vn_g = batch_eigvalsh(Hbo_g)
        e_bo_g = xp.sort(Ad_vn_g.flatten())
        print("e_bo_new g.s.",e_bo_g[0:10])
        bo_vib_ggap = e_bo_g[1] - e_bo_g[0]
        print("BO new vib gap g.s.",bo_vib_ggap,flush=True)

        Hbo_e = +1/(2*H.mu12)*(-H.ddR2 + xp.diag(H.Pphi**2/H.R**2)+ xp.diag(H.Ptheta**2/H.R**2)+xp.diag(1/(2*H.R)**2)) +xp.diag(Ad_nse)
        Ad_vn_e = batch_eigvalsh(Hbo_e)
        e_bo_e = xp.sort(Ad_vn_e.flatten())
        print("e_bo_new e.s.",e_bo_e[0:10])
        bo_vib_egap = e_bo_e[1] - e_bo_e[0]
        print("BO new vib gap e.s.",bo_vib_egap,flush=True)
#
        EPSg += 1/(2*H.mu12)*(Pval**2+H.Pphi**2/Rval**2+H.Ptheta**2/Rval**2+1/(2*Rval)**2)
        HPSg = inverse_weyl_transform(EPSg, H.shape[0], H.R, H.P_R)
        EPSvg, evecs_vg = xp.linalg.eigh(HPSg)
        print("EPSv g.s.",xp.sort(EPSvg.flatten())[:10])
        print("PS vib gap g.s.",EPSvg[1]-EPSvg[0],flush=True)

        EPSe += 1/(2*H.mu12)*(Pval**2+H.Pphi**2/Rval**2+H.Ptheta**2/Rval**2+1/(2*Rval)**2)
        HPSe = inverse_weyl_transform(EPSe, H.shape[0], H.R, H.P_R)
        EPSve,evecs_ve = xp.linalg.eigh(HPSe)
        print("EPSv e.s.",xp.sort(EPSve.flatten())[:10])
        print("PS vib gap e.s.",EPSve[1]-EPSve[0],flush=True)
       
        print("norm HPSg",xp.linalg.norm(HPSg+HPSg.conj().T))
        print("HPSg",HPSg[0:6,0:6])
        print("norm HPSe",xp.linalg.norm(HPSe+HPSe.conj().T))
        print("HPSe",HPSe[0:6,0:6])


        HGamma_RRy_gs = inverse_weyl_transform(exp_RxGamma_y_gs,NR,H.R,H.P_R)
        HGamma_RRy_es = inverse_weyl_transform(exp_RxGamma_y_es,NR,H.R,H.P_R)
        HGamma_RRz_gs = inverse_weyl_transform(exp_RxGamma_z_gs,NR,H.R,H.P_R)
        HGamma_RRz_es = inverse_weyl_transform(exp_RxGamma_z_es,NR,H.R,H.P_R)
        Hsx_gs = inverse_weyl_transform(exp_sx_gs,NR,H.R,H.P_R)
        Hsx_es = inverse_weyl_transform(exp_sx_es,NR,H.R,H.P_R)
        Hsy_gs = inverse_weyl_transform(exp_sy_gs,NR,H.R,H.P_R)
        Hsy_es = inverse_weyl_transform(exp_sy_es,NR,H.R,H.P_R)
        Hsz_gs = inverse_weyl_transform(exp_sz_gs,NR,H.R,H.P_R)
        Hsz_es = inverse_weyl_transform(exp_sz_es,NR,H.R,H.P_R)
        Hlx_gs = inverse_weyl_transform(exp_lx_gs,NR,H.R,H.P_R)
        Hlx_es = inverse_weyl_transform(exp_lx_es,NR,H.R,H.P_R)
        Hly_gs = inverse_weyl_transform(exp_ly_gs,NR,H.R,H.P_R)
        Hly_es = inverse_weyl_transform(exp_ly_es,NR,H.R,H.P_R)
        Hlz_gs = inverse_weyl_transform(exp_lz_gs,NR,H.R,H.P_R)
        Hlz_es = inverse_weyl_transform(exp_lz_es,NR,H.R,H.P_R)

        print("Hsx_gs Hermitian",xp.linalg.norm(Hsx_gs-Hsx_gs.conj().T))
        print("Hsx_es Hermitian",xp.linalg.norm(Hsx_es-Hsx_es.conj().T))
        print("Hsy_gs Hermitian",xp.linalg.norm(Hsy_gs-Hsy_gs.conj().T))
        print("Hsy_es Hermitian",xp.linalg.norm(Hsy_es-Hsy_es.conj().T))
        print("Hsz_gs Hermitian",xp.linalg.norm(Hsz_gs-Hsz_gs.conj().T))
        print("Hsz_es Hermitian",xp.linalg.norm(Hsz_es-Hsz_es.conj().T))
        print("Hlx_gs Hermitian",xp.linalg.norm(Hlx_gs-Hlx_gs.conj().T))
        print("Hlx_es Hermitian",xp.linalg.norm(Hlx_es-Hlx_es.conj().T))
        print("Hly_gs Hermitian",xp.linalg.norm(Hly_gs-Hly_gs.conj().T))
        print("Hly_es Hermitian",xp.linalg.norm(Hly_es-Hly_es.conj().T))
        print("Hlz_gs Hermitian",xp.linalg.norm(Hlz_gs-Hlz_gs.conj().T))
        print("Hlz_es Hermitian",xp.linalg.norm(Hlz_es-Hlz_es.conj().T))



        EPSg += 1/(2*mu12) * (Pval**2 + H.Pphi**2/Rval**2 + H.Ptheta**2/Rval**2 + 1/(2*Rval)**2)
        EPSe += 1/(2*mu12) * (Pval**2 + H.Pphi**2/Rval**2 + H.Ptheta**2/Rval**2 + 1/(2*Rval)**2)

        HPSg = inverse_weyl_transform(EPSg, NR, H.R, H.P_R)
        HPSe = inverse_weyl_transform(EPSe, NR, H.R, H.P_R)

        EPSvg, evecs_vg = xp.linalg.eigh(HPSg)
        EPSve, evecs_ve = xp.linalg.eigh(HPSe)
        print('EPSv g.s.', xp.sort(EPSvg.flatten())[:10])
        print('EPSv e.s.', xp.sort(EPSve.flatten())[:10])

        print("norm HPSg",xp.linalg.norm(HPSg-HPSg.conj().T))
        #print("HPSg",HPSg[0:6,0:6])
        print("norm HPSe",xp.linalg.norm(HPSe-HPSe.conj().T))
        #print("HPSe",HPSe[0:6,0:6])


        RxGamma_y_gs = xp.conj(evecs_vg[:,0]).T @ (HGamma_RRy_gs @ evecs_vg[:,0])
        RxGamma_y_es = xp.conj(evecs_ve[:,1]).T @ (HGamma_RRy_es @ evecs_ve[:,1])
        RxGamma_z_gs = xp.conj(evecs_vg[:,0]).T @ (HGamma_RRz_gs @ evecs_vg[:,0])
        RxGamma_z_es = xp.conj(evecs_ve[:,1]).T @ (HGamma_RRz_es @ evecs_ve[:,1])

        v_sx_gs = xp.conj(evecs_vg[:,0]).T @ (Hsx_gs @ evecs_vg[:,0])
        v_sx_es = xp.conj(evecs_ve[:,1]).T @ (Hsx_es @ evecs_ve[:,1])
        v_sy_gs = xp.conj(evecs_vg[:,0]).T @ (Hsy_gs @ evecs_vg[:,0])
        v_sy_es = xp.conj(evecs_ve[:,1]).T @ (Hsy_es @ evecs_ve[:,1])
        v_sz_gs = xp.conj(evecs_vg[:,0]).T @ (Hsz_gs @ evecs_vg[:,0])
        v_sz_es = xp.conj(evecs_ve[:,1]).T @ (Hsz_es @ evecs_ve[:,1])

        v_lx_gs = xp.conj(evecs_vg[:,0]).T @ (Hlx_gs @ evecs_vg[:,0])
        v_lx_es = xp.conj(evecs_ve[:,1]).T @ (Hlx_es @ evecs_ve[:,1])
        v_ly_gs = xp.conj(evecs_vg[:,0]).T @ (Hly_gs @ evecs_vg[:,0])
        v_ly_es = xp.conj(evecs_ve[:,1]).T @ (Hly_es @ evecs_ve[:,1])
        v_lz_gs = xp.conj(evecs_vg[:,0]).T @ (Hlz_gs @ evecs_vg[:,0])
        v_lz_es = xp.conj(evecs_ve[:,1]).T @ (Hlz_es @ evecs_ve[:,1])

        check_gamma_y_gs = RxGamma_y_gs + v_sx_gs + v_ly_gs
        check_gamma_y_es = RxGamma_y_es + v_sx_es + v_ly_es
        check_gamma_z_gs = RxGamma_z_gs + v_sz_gs + v_lx_gs
        check_gamma_z_es = RxGamma_z_es + v_sz_es + v_lx_es
        check_x_gs = v_sx_gs + v_lx_gs
        check_x_es = v_sx_es + v_lx_es

        def _s(x):
            return float(xp.asarray(x).real) if xp.size(x) == 1 else float(x)

        fmt = "  {:>12.7f}"
        print("gs")
        print("         gamma          l          s           sum")
        print("  x " + fmt.format(0.0)           + fmt.format(_s(v_lx_gs)) + fmt.format(_s(v_sx_gs)) + fmt.format(_s(v_lx_gs + v_sx_gs)))
        print("  y " + fmt.format(_s(RxGamma_y_gs)) + fmt.format(_s(v_ly_gs)) + fmt.format(_s(v_sy_gs)) + fmt.format(_s(check_gamma_y_gs)))
        print("  z " + fmt.format(_s(RxGamma_z_gs)) + fmt.format(_s(v_lz_gs)) + fmt.format(_s(v_sz_gs)) + fmt.format(_s(check_gamma_z_gs)))
        print("es")
        print("         gamma          l          s           sum")
        print("  x " + fmt.format(0.0)           + fmt.format(_s(v_lx_es)) + fmt.format(_s(v_sx_es)) + fmt.format(_s(check_x_es)))
        print("  y " + fmt.format(_s(RxGamma_y_es)) + fmt.format(_s(v_ly_es)) + fmt.format(_s(v_sy_es)) + fmt.format(_s(check_gamma_y_es)))
        print("  z " + fmt.format(_s(RxGamma_z_es)) + fmt.format(_s(v_lz_es)) + fmt.format(_s(v_sz_es)) + fmt.format(_s(check_gamma_z_es)))

        print("check_gamma_y_gs",check_gamma_y_gs.real)
        print("check_gamma_y_es",check_gamma_y_es.real)
        print("check_gamma_z_gs",check_gamma_z_gs.real)
        print("check_gamma_z_es",check_gamma_z_es.real)
        print("check_x_gs",check_x_gs.real)
        print("check_x_es",check_x_es.real)

        print("gs:S(S+1)= <sx^2> + <sy^2> + <sz^2>", v_sx_gs**2 + v_sy_gs**2 + v_sz_gs**2)
        print("es:S(S+1)= <sx^2> + <sy^2> + <sz^2>", v_sx_es**2 + v_sy_es**2 + v_sz_es**2)
        print("gs:L(L+1)= <lx^2> + <ly^2> + <lz^2>", v_lx_gs**2 + v_ly_gs**2 + v_lz_gs**2)
        print("es:L(L+1)= <lx^2> + <ly^2> + <lz^2>", v_lx_es**2 + v_ly_es**2 + v_lz_es**2)


        EPS_bog = xp.zeros((H.shape[0], H.shape[0]))
        Helmatg = xp.repeat(ivalg,H.shape[0],axis=1)
        EPS_bog += Helmatg   
        EPS_bog += 1/(2*H.mu12)*(Pval**2+H.Pphi**2/Rval**2+H.Ptheta**2/Rval**2+1/(2*Rval)**2)
        HPS_bog = inverse_weyl_transform(EPS_bog, H.shape[0], H.R, H.P_R)
        EPSv_bog = batch_eigvalsh(HPS_bog)
        print("EPSv_bo",EPSv_bog[0:10])
    
        EPS_boe = xp.zeros((H.shape[0], H.shape[0]))
        Helmate = xp.repeat(ivale,H.shape[0],axis=1)
        EPS_boe += Helmate   
        EPS_boe += 1/(2*H.mu12)*(Pval**2+H.Pphi**2/Rval**2+H.Ptheta**2/Rval**2+1/(2*Rval)**2)
        HPS_boe = inverse_weyl_transform(EPS_boe, H.shape[0], H.R, H.P_R)
        EPSv_boe = batch_eigvalsh(HPS_boe)
        print("EPSv_bo",EPSv_boe[0:10])