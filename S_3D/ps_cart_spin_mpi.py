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
        self.sx = xp.array([[0,1],[1,0]])
        self.sy = xp.array([[0,-1j],[1j,0]])
        self.sz = xp.array([[1,0],[0,-1]])
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
            xp.einsum('sS,ij,BSjkl->Bsikl',self.si,self.ddx2,xdav,optimize=True)
            +xp.einsum('sS,ij,BSkjl->Bskil',self.si,self.ddy2,xdav,optimize=True)
            +xp.einsum('sS,ij,BSklj->Bskli',self.si,self.ddz2,xdav,optimize=True)
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
                Vx = xp.einsum('sS,xyz,Bsxyz->BSxyz',self.si,self.Vgrid[Ri],x,optimize=True)
                Hpsdav = (
                    Vx + self.Tx(x) + self.soc_full(x, soc_data_i)
                    +xp.einsum('sS,xayz,BSxyz->Bsayz', self.si, term1, x, optimize=True) 
                    +xp.einsum('sS,xybz,BSxyz->Bsxbz', self.si, term2, x, optimize=True) 
                    +xp.einsum('sS,xyzc,BSxyz->Bsxyc', self.si, term3, x, optimize=True)
                    +xp.einsum('sS,xyz,BSxyz->Bsxyz',self.sy,coeffgammaerfy,x,optimize=True)
                    +xp.einsum('sS,xyz,BSxyz->Bsxyz',self.sx,coeffgammaerfz,x,optimize=True)
                )
                
            elif self.soc =='no_spin_erf':
                Vx = xp.einsum('sS,xyz,BSxyz->Bsxyz',self.si,self.Vgrid[Ri],x,optimize=True)
                Hpsdav = (
                    Vx + self.Tx(x) + self.soc_full(x, soc_data_i)
                    +xp.einsum('sS,xayz,BSxyz->Bsayz', self.si, term1, x, optimize=True) 
                    +xp.einsum('sS,xybz,BSxyz->Bsxbz', self.si, term2, x, optimize=True) 
                    +xp.einsum('sS,xyzc,BSxyz->Bsxyc', self.si, term3, x, optimize=True)
                )
            elif self.soc =='no_soc':
                Vx = xp.einsum('sS,xyz,BSxyz->Bsxyz',self.si,self.Vgrid[Ri],x,optimize=True)
                Hpsdav = (
                    Vx + self.Tx(x)
                    +xp.einsum('sS,xayz,BSxyz->Bsayz', self.si, term1, x, optimize=True) 
                    +xp.einsum('sS,xybz,BSxyz->Bsxbz', self.si, term2, x, optimize=True) 
                    +xp.einsum('sS,xyzc,BSxyz->Bsxyc', self.si, term3, x, optimize=True)
                    +xp.einsum('sS,xyz,BSxyz->Bsxyz',self.sy,coeffgammaerfy,x,optimize=True)
                    +xp.einsum('sS,xyz,BSxyz->Bsxyz',self.sx,coeffgammaerfz,x,optimize=True)
                )
            return Hpsdav.reshape(xdav.shape)

        return Hx_ps

    def Hbo_dav(self,Ri,soc_data_i):

        def Hxbo(xdav):
            #print("xdav shape:", xdav.shape)
            x = xdav.reshape((-1,)+self.bospinshape)               
            if self.soc =='full':  
                #print("Ri",Ri)              
                Vx = xp.einsum('sS,xyz,BSxyz->Bsxyz',self.si,self.Vgrid[Ri],x,optimize=True)                
                Hbodav = (
                    Vx + self.Tx(x) + self.soc_full(x, soc_data_i)
                )               
            elif self.soc == 'no_soc':
                #print("Ri",i)
                Vx = xp.einsum('sS,xyz,BSxyz->Bsxyz',self.si,self.Vgrid[Ri],x,optimize=True)
                Hbodav = (
                    Vx + self.Tx(x)
                ) 
            elif self.soc =='no_spin_erf':
                Vx = xp.einsum('sS,xyz,BSxyz->Bsxyz',self.si,self.Vgrid[Ri],x,optimize=True)
                Hbodav = (
                    Vx + self.Tx(x) + self.soc_full(x, soc_data_i)
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
            
            E1, E2 = H.Efield(H.R[i], self.x_grid, self.y_grid, self.z_grid)
            c1 = 0.5 * (1/137)**2 * E1 * self.alpha / (self.m_e**2)
            c2 = 0.5 * (1/137)**2 * E2 * self.alpha / (self.m_e**2)
            coef12 = c1 + c2
            coef1, coef2 = c1, c2
            xi, yi, zi = self.x, self.y, self.z
            mu12, M1, M2 = H.mu12, H.M_1, H.M_2
            w_y_coef12 = yi[None, :, None] * coef12
            w_z_coef12 = zi[None, None, :] * coef12
            w_x_coef1  = (xi - H.R[i]*mu12/M1)[:, None, None] * coef1
            w_x_coef2  = (xi + H.R[i]*mu12/M2)[:, None, None] * coef2
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
            exit()

        return Ad_nsg, Ad_nse, ivalg, ivale  
  
def Gamma_etf(R, ddx, ddy, ddz, t1):
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


def Gamma_erf_spin(R, M1, M2, t1, t2):
    J1xs = -0.5j*t1
    J1ys = -0.5j*t1
    J2xs = -0.5j*t2
    J2ys = -0.5j*t2
    gammaerf1ys = -1/R*(-J1ys-J2ys)
    gammaerf1zs = -1/R*(J1xs+J2xs)
    gammaerf2ys = -gammaerf1ys
    gammaerf2zs = -gammaerf1zs
    Gammaerfys = (M2*gammaerf1ys-M1*gammaerf2ys)/(M1+M2)
    Gammaerfzs = (M2*gammaerf1zs-M1*gammaerf2zs)/(M1+M2)
    #gammaspinerfs = (gammaerf1ys,gammaerf1zs,gammaerf2ys,gammaerf2zs)
    return Gammaerfys, Gammaerfzs


def Gamma_erf_orb(R, rx, ry, rz, M1, M2, mu12, gammaetf, t1, t2):
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
    Gammaerfya = (M2*gammaerf1ya-M1*gammaerf2ya)/(M1+M2)
    Gammaerfyb = (M2*gammaerf1yb-M1*gammaerf2yb)/(M1+M2)
    Gammaerfyc = (M2*gammaerf1yc-M1*gammaerf2yc)/(M1+M2)
    Gammaerfza = (M2*gammaerf1za-M1*gammaerf2za)/(M1+M2)
    Gammaerfzb = (M2*gammaerf1zb-M1*gammaerf2zb)/(M1+M2)
    return Gammaerfya, Gammaerfyb, Gammaerfyc, Gammaerfza, Gammaerfzb

def R_Gamma_exp(R,evecs,gammavals,si,sy,sx):

    (gammaetfy, gammaerfya, gammaerfyb, gammaerfyc, gammaerfza, gammaerfzb, gammaerfsy, gammaerfsz) = gammavals

    evecs_gs = evecs_save[0,:].reshape(2, Nx, Ny, Nz)
    evecs_conj_gs = xp.conj(evecs_gs)
    evecs_es = evecs_save[1,:].reshape(2, Nx, Ny, Nz)
    evecs_conj_es = xp.conj(evecs_es)

    gammaerfybc = gammaerfyb + gammaerfyc
    Gamma_y_gs_etf = xp.einsum('sxyz,sS,xybz,Sxbz->', evecs_conj_gs, si, gammaetfy, evecs_gs, optimize=True) 
    Gamma_y_gs_erf_a = xp.einsum('sxyz,sS,xayz,Sayz->', evecs_conj_gs, si, gammaerfya, evecs_gs, optimize=True)
    Gamma_y_gs_erf_bc = xp.einsum('sxyz,sS,xyzc,Sxyc->', evecs_conj_gs,si, gammaerfybc, evecs_gs, optimize=True)
    Gamma_y_gs_erf_s = xp.einsum('sxyz,sS,xyz,Sxyz->', evecs_conj_gs,sy, gammaerfsy, evecs_gs, optimize=True) 

    Gamma_y_es_etf = xp.einsum('sxyz,sS,xybz,Sxbz->', evecs_conj_es, si, gammaetfy, evecs_es, optimize=True) 
    Gamma_y_es_erf_a = xp.einsum('sxyz,sS,xayz,Sayz->', evecs_conj_es, si, gammaerfya, evecs_es, optimize=True)
    Gamma_y_es_erf_bc = xp.einsum('sxyz,sS,xyzc,Sxyc->', evecs_conj_es,si, gammaerfybc, evecs_es, optimize=True)
    Gamma_y_es_erf_s = xp.einsum('sxyz,sS,xyz,Sxyz->', evecs_conj_es,sy, gammaerfsy, evecs_es, optimize=True) 

    Gamma_z_gs_etf = xp.einsum('sxyz,sS,xyzc,Sxyc->', evecs_conj_gs, si, gammaetfz, evecs_gs, optimize=True) 
    Gamma_z_gs_erf_a = xp.einsum('sxyz,sS,xyzc,Sayc->', evecs_conj_gs, si, gammaerfza, evecs_gs, optimize=True)
    Gamma_z_gs_erf_b = xp.einsum('sxyz,sS,xybz,Sxbc->', evecs_conj_gs,si, gammaerfzb, evecs_gs, optimize=True)
    Gamma_z_gs_erf_s = xp.einsum('sxyz,sS,xyz,Sxyz->', evecs_conj_gs,sx, gammaerfsz, evecs_gs, optimize=True) 

    Gamma_z_es_etf = xp.einsum('sxyz,sS,xyzc,Sxyc->', evecs_conj_es, si, gammaetfz, evecs_es, optimize=True) 
    Gamma_z_es_erf_a = xp.einsum('sxyz,sS,xyzc,Sayc->', evecs_conj_es, si, gammaerfza, evecs_es, optimize=True)
    Gamma_z_es_erf_b = xp.einsum('sxyz,sS,xybz,Sxbc->', evecs_conj_es,si, gammaerfzb, evecs_es, optimize=True)
    Gamma_z_es_erf_s = xp.einsum('sxyz,sS,xyz,Sxyz->', evecs_conj_es,sx, gammaerfsz, evecs_es, optimize=True) 

    Gamma_y_gs = Gamma_y_gs_etf + Gamma_y_gs_erf_a + Gamma_y_gs_erf_bc + Gamma_y_gs_erf_s
    Gamma_y_es = Gamma_y_es_etf + Gamma_y_es_erf_a + Gamma_y_es_erf_bc + Gamma_y_es_erf_s
    Gamma_z_gs = Gamma_z_gs_etf + Gamma_z_gs_erf_a + Gamma_z_gs_erf_b + Gamma_z_gs_erf_s
    Gamma_z_es = Gamma_z_es_etf + Gamma_z_es_erf_a + Gamma_z_es_erf_b + Gamma_z_es_erf_s

    R_Gamma_y_gs = -R*Gamma_z_gs
    R_Gamma_y_es = -R*Gamma_z_es
    R_Gamma_z_gs = R*Gamma_y_gs
    R_Gamma_z_es = R*Gamma_y_es
    

    return R_Gamma_y_gs, R_Gamma_y_es, R_Gamma_z_gs, R_Gamma_z_es

def parse_args():
    parser = ap.ArgumentParser(
        prog='3body-3D',
        description="computes the lowest k eigenvalues of a 3-body potential in 3D")

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
    parser.add_argument('-R', dest="NR", metavar="NR", default=101, type=int)
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
    exp_RxGamma_y_gs = xp.zeros((NR, NR),dtype=xp.complex128)
    exp_RxGamma_y_es = xp.zeros((NR, NR),dtype=xp.complex128)
    exp_RxGamma_z_gs = xp.zeros((NR, NR),dtype=xp.complex128)
    exp_RxGamma_z_es = xp.zeros((NR, NR),dtype=xp.complex128)
    
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

        np.save(os.path.join(folder, f'matrix_{arge.potential}_j_{args.J}_m_{args.Pphi}_a_{args.alpha}_m_{args.M_2}_Ad_nsg_split_{args.split_idx}.npy'), Ad_nsg)
        np.save(os.path.join(folder, f'matrix_{arge.potential}_j_{args.J}_m_{args.Pphi}_a_{args.alpha}_m_{args.M_2}_Ad_nse_split_{args.split_idx}.npy'), Ad_nse)
        exit()

    

    with timer_ctx(f"R for loop"):
        for i in sequence:
            print("Atom Ri idx",i, "Atom Ri",H.R[i],flush=True)
            diag = H.buildDiag(i)               
            if evecs_prev == True:
                guess_bo = guess_spin
                print("I ini idx",i)
            else:
                guess_bo = evecs
                print("I idx",i)
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
            gammaetf1x, gammaetf1y, gammaetf1z = Gamma_etf(H.R[i], H.ddx1, H.ddy1, H.ddz1, t1)
            gammaetf2x, gammaetf2y, gammaetf2z = Gamma_etf(H.R[i], H.ddx1, H.ddy1, H.ddz1, t2)
            gammaetf = (gammaetf1x, gammaetf1y, gammaetf1z, gammaetf2x, gammaetf2y, gammaetf2z)
            gammaerfya, gammaerfyb, gammaerfyc, gammaerfza, gammaerfzb = Gamma_erf_orb(H.R[i], H.x, H.y, H.z, H.M_1, H.M_2, H.mu12, gammaetf, t1, t2)
            gammaerfsy, gammaerfsz = Gamma_erf_spin(H.R[i], H.M_1, H.M_2, t1, t2)
            gammaetfx = (H.M_2*gammaetf1x-H.M_1*gammaetf2x)/(H.M_1+H.M_2)
            gammaetfy = (H.M_2*gammaetf1y-H.M_1*gammaetf2y)/(H.M_1+H.M_2)
            gammaetfz = (H.M_2*gammaetf1z-H.M_1*gammaetf2z)/(H.M_1+H.M_2)
            
            term2 = (
                    gammacoeff_phi[i] * gammaetfy +
                    gammacoeff_theta[i] * gammaerfzb
                )

            term3 = (
                    gammacoeff_phi[i] * (gammaerfyb + gammaerfyc) +
                    gammacoeff_theta[i] * (gammaetfz + gammaerfza)
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
                        print("J ini idx",j)
                    else:
                        guess_ps = evecs_save
                        print("J idx",j)
                    term1 = (
                        gammacoeff_R[i,j] * gammaetfx +
                        gammacoeff_phi[i] * gammaerfya
                    )
                    
                    with timer_ctx(f"Davidson of size {H.size}"):
                        conv, e_ps_approx, evecs_save = lib.davidson1(
                            H.ps_ham(term1,term2,term3,coeffgammaerfy,coeffgammaerfz,i,soc_data_i),
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

                    gammavals = (gammaetfy, gammaerfya, gammaerfyb, gammaerfyc, gammaerfza, gammaerfzb, gammaerfsy, gammaerfsz)

                    R_Gamma_y_gs, R_Gamma_y_es, R_Gamma_z_gs, R_Gamma_z_es = R_Gamma_exp(H.R[i],evecs,gammavals,H.si,H.sy,H.sx)
                    exp_RxGamma_y_gs[i,j] = R_Gamma_y_gs
                    exp_RxGamma_y_es[i,j] = R_Gamma_y_es
                    exp_RxGamma_z_gs[i,j] = R_Gamma_z_gs
                    exp_RxGamma_z_es[i,j] = R_Gamma_z_es

                    
    np.save(os.path.join(folder, f'matrix_{args.potential}_j_{args.J}_m_{args.Pphi}_a_{args.alpha}_m_{args.M_2}_Ad_nsg_split_{args.split_idx}.npy'), Ad_nsg)
    np.save(os.path.join(folder, f'matrix_{args.potential}_j_{args.J}_m_{args.Pphi}_a_{args.alpha}_m_{args.M_2}_Ad_nse_split_{args.split_idx}.npy'), Ad_nse)
    np.save(os.path.join(folder, f'matrix_{args.potential}_j_{args.J}_Pth_{args.Ptheta}_Pph_{args.Pphi}_a_{args.alpha}_m_{args.M_2}_EPSg_split_{args.split_idx}.npy'), EPSg)
    np.save(os.path.join(folder, f'matrix_{args.potential}_j_{args.J}_Pth_{args.Ptheta}_Pph_{args.Pphi}_a_{args.alpha}_m_{args.M_2}_EPSe_split_{args.split_idx}.npy'), EPSe)
    np.save(os.path.join(folder, f'matrix_{args.potential}_j_{args.J}_Pth_{args.Ptheta}_Pph_{args.Pphi}_a_{args.alpha}_m_{args.M_2}_exp_RxGamma_y_gs_split_{args.split_idx}.npy'), exp_RxGamma_y_gs)
    np.save(os.path.join(folder, f'matrix_{args.potential}_j_{args.J}_Pth_{args.Ptheta}_Pph_{args.Pphi}_a_{args.alpha}_m_{args.M_2}_exp_RxGamma_y_es_split_{args.split_idx}.npy'), exp_RxGamma_y_es)
    np.save(os.path.join(folder, f'matrix_{args.potential}_j_{args.J}_Pth_{args.Ptheta}_Pph_{args.Pphi}_a_{args.alpha}_m_{args.M_2}_exp_RxGamma_z_gs_split_{args.split_idx}.npy'), exp_RxGamma_z_gs)
    np.save(os.path.join(folder, f'matrix_{args.potential}_j_{args.J}_Pth_{args.Ptheta}_Pph_{args.Pphi}_a_{args.alpha}_m_{args.M_2}_exp_RxGamma_z_es_split_{args.split_idx}.npy'), exp_RxGamma_z_es)

