from numpy.fft import fft, fftshift

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
        'R', 'P_R', 'R_grid', 'RP_grid','_Efunc',
        'x', 'y', 'z','x_grid','y_grid','z_grid', 'xb_grid','yb_grid','zb_grid',
        'ddR2', 'ddx2','ddx1','ddy2','ddy1','ddz2','ddz1',
        'axes','Vgrid', '_preconditioner_data','Pg','Pphi','Ptheta',
        'shape','boshape','bospinshape','size','guess','k','mu12','_Vfunc',
        '_locked','max_threads','alpha','soc','sx','sy','sz','E1','E2'
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

        self.mu   = xp.sqrt(self.M_1*self.M_2*self.m_e/(self.M_1+self.M_2+self.m_e))
        self.mur  = (self.M_1+self.M_2)*self.m_e/(self.M_1+self.M_2+self.m_e)
        self.mu12 = self.M_1*self.M_2/(self.M_1+self.M_2)
        self._Vfunc, extent_func, self._Efunc = {
            'soft_coulomb': (potentials.soft_coulomb, potentials.extents_soft_coulomb, None),
            'borgis': (partial(potentials.borgis, asymmetry_param=1), potentials.extents_borgis, potentials.Efield_borgis),
            'erf_coulomb':(potentials.erf_coulomb, potentials.extents_erf_coulomb, potentials.Efield_coulomb)
            }[args.potential]

        extent = extent_func(self.mu12)
        print("alpha=",self.alpha,"  soc_const=",1/2*(1/137)**2*self.alpha)

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
        self.ddR2, _ = KE_Borisov_3D(self.R, bare=True)
        #self.ddR2  = KE(args.NR, dR, bare=True, cyclic=False)
        #self.ddR2  = KE_FFT(args.NR, P, R)
    
        self.ddx2 = KE(args.Nx, dx, bare=True, cyclic=False)
        self.ddx1 = KE(args.Nx, dx, bare=True, cyclic=False, order=1) 

        self.ddy2 = KE(args.Ny, dy, bare=True, cyclic=False)
        self.ddy1 = KE(args.Ny, dy, bare=True, cyclic=False, order=1)

        self.ddz2 = KE(args.Nz, dz, bare=True, cyclic=False)
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

    def BO_energies(self,args,iR,sequence):
        
        NR,Nx,Ny,Nz = self.shape
    
        Ad_nsg = xp.zeros(NR)
        Ad_nse = xp.zeros(NR)
        ivalg = xp.zeros([NR,1])
        ivale = xp.zeros([NR,1])

        for i in sequence:
            print("Atom Ri idx",i, "Atom Ri",self.R[i],flush=True)
            diag = self.buildDiag(i)   

            guess_ns = xp.exp(-(self.Vgrid[i] - xp.min(self.Vgrid[i]))**2/27.211**2).ravel()
            guess_zeros = xp.zeros(len(guess_ns))
            guess_spin = xp.array([xp.append(guess_ns, guess_zeros),xp.append(guess_zeros, guess_ns)])
            if i==iR:
                guess_bo = guess_spin
            else:
                guess_bo = evecs

            conv, e_approx, evecs = lib.davidson1(
                self.Hbo_dav(i),
                guess_bo,
                lambda dx, e, x0: dx/(diag-e+1e-5),
                nroots=args.k,
                max_cycle=args.iterations,
                verbose=args.verbosity,
                max_space=args.subspace,
                max_memory=get_davidson_mem(0.75),
                #tol=1e-12, #FIXME:DEBUG
                tol=1e-8
            )
            print("Davidson:", e_approx)
            print(conv)
            Ad_nsg[i] = e_approx[0]
            Ad_nse[i] = e_approx[1]
            ivalg[i,0] = e_approx[0]
            ivale[i,0] = e_approx[1]
            #exit()
        #print("Ad_nsg",Ad_nsg)
        #exit()

        return Ad_nsg, Ad_nse, ivalg, ivale        
                
    def Tx(self,xdav):
        Hel_dav = -1/(2*self.mur)*(
            xp.einsum('sS,ij,Bsjkl->BSikl',self.si,self.ddx2,xdav,optimize=True)
            +xp.einsum('sS,ij,Bskjl->BSkil',self.si,self.ddy2,xdav,optimize=True)
            +xp.einsum('sS,ij,Bsklj->BSkli',self.si,self.ddz2,xdav,optimize=True)
            )
        return Hel_dav.reshape(xdav.shape)

    def soc_full(self,xdav,R):

        x = xdav.reshape((-1,) + H.bospinshape)    
        E1, E2 = self.Efield(R, self.x_grid, self.y_grid, self.z_grid)
        c1 = 1/2*(1/137)**2*E1*self.alpha*(1/self.m_e**2)
        c2 = 1/2*(1/137)**2*E2*self.alpha*(1/self.m_e**2)
        coef12 = (c1 + c2)        
        coef1  = c1
        coef2  = c2
        xi,yi,zi = self.x, self.y, self.z
        sx,sy,sz = self.sx, self.sy, self.sz
        mu12,M1,M2 = self.mu12,self.M_1,self.M_2

        Hsocdav = 0.5j*( 
            -xp.einsum('sS,y,zc,Bsxyz,xyz->BSxyc', sx, yi, self.ddz1,x,coef12,optimize=True) 
            +xp.einsum('sS,z,yb,Bsxyz,xyz->BSxbz', sx, zi, self.ddy1,x,coef12,optimize=True) 
            -xp.einsum('sS,z,xa,Bsxyz,xyz->BSayz', sy, zi, self.ddx1,x,coef12,optimize=True) 
            +xp.einsum('sS,x,zc,Bsxyz,xyz->BSxyc', sy, xi-(R*mu12/M1), self.ddz1,x,coef1,optimize=True) 
            +xp.einsum('sS,x,zc,Bsxyz,xyz->BSxyc', sy, xi+(R*mu12/M2), self.ddz1,x,coef2,optimize=True) 
            -xp.einsum('sS,x,yb,Bsxyz,xyz->BSxbz', sz, xi-(R*mu12/M1), self.ddy1,x,coef1,optimize=True) 
            -xp.einsum('sS,x,yb,Bsxyz,xyz->BSxbz', sz, xi+(R*mu12/M2), self.ddy1,x,coef2,optimize=True) 
            +xp.einsum('sS,y,xa,Bsxyz,xyz->BSayz', sz, yi, self.ddx1,x,coef12,optimize=True) 
            )
        #Hsocdav_check = 0.5j*( 
        #    -xp.einsum('sS,y,zc,xyz->sSxyc', sx, yi, self.ddz1,coef12,optimize=True) 
        #    +xp.einsum('sS,z,yb,xyz->sSxbz', sx, zi, self.ddy1,coef12,optimize=True) 
        #    #-xp.einsum('sS,z,xa,xyz->sSayz', sy, zi, self.ddx1,coef12,optimize=True) 
        #    #+xp.einsum('sS,x,zc,xyz->sSxyc', sy, xi-(R*mu12/M1), self.ddz1,coef1,optimize=True) 
        #    #+xp.einsum('sS,x,zc,xyz->sSxyc', sy, xi+(R*mu12/M2), self.ddz1,coef2,optimize=True) 
        #    -xp.einsum('sS,x,yb,xyz->sSxbz', sz, xi-(R*mu12/M1), self.ddy1,coef1,optimize=True) 
        #    -xp.einsum('sS,x,yb,xyz->sSxbz', sz, xi+(R*mu12/M2), self.ddy1,coef2,optimize=True) 
        #    +xp.einsum('sS,y,xa,xyz->sSayz', sz, yi, self.ddx1,coef12,optimize=True) 
        #    )
        #print("Hsocdav_check",xp.transpose(xp.conj(Hsocdav_check[0,1,:,:,:])))
        #print("Hsocdav_checkdown",(Hsocdav_check[1,0,:,:,:]))

        return Hsocdav.reshape(xdav.shape)

    def soc_naive(self,xdav,R):

        x = xdav.reshape((-1,) + self.bospinshape) 
        rR1 = (self.x_grid**2 + self.y_grid**2 +self.z_grid**2)**(1.5) 
        rR2 = (self.x_grid**2 + self.y_grid**2 +self.z_grid**2)**(1.5) 
        c1 = 1/2*(1/137)**2*self.g_1/rR1*self.alpha 
        c2 = 1/2*(1/137)**2*self.g_2/rR2*self.alpha 
        coef12 = (c1 + c2)       
        coef1  = c1
        coef2  = c2
        xi,yi,zi = self.x, self.y, self.z
        sx,sy,sz = self.sx, self.sy, self.sz
        mu12,M1,M2 = self.mu12,self.M_1,self.M_2

        Hsocdav = 0.5*( 
            -xp.einsum('sS,y,zc,Bsxyz,xyz->BSxyc', sx, yi, self.ddz1,x,coef12,optimize=True) 
            +xp.einsum('sS,z,yb,Bsxyz,xyz->BSxbz', sx, zi, self.ddy1,x,coef12,optimize=True) 
            -xp.einsum('sS,z,xa,Bsxyz,xyz->BSayz', sy, zi, self.ddx1,x,coef12,optimize=True) 
            +xp.einsum('sS,x,zc,Bsxyz,xyz->BSxyc', sy, xi-(R*mu12/M1), self.ddz1,x,coef1,optimize=True) 
            +xp.einsum('sS,x,zc,Bsxyz,xyz->BSxyc', sy, xi+(R*mu12/M2), self.ddz1,x,coef2,optimize=True) 
            -xp.einsum('sS,x,yb,Bsxyz,xyz->BSxbz', sz, xi-(R*mu12/M1), self.ddy1,x,coef1,optimize=True) 
            -xp.einsum('sS,x,yb,Bsxyz,xyz->BSxbz', sz, xi+(R*mu12/M2), self.ddy1,x,coef2,optimize=True) 
            +xp.einsum('sS,y,xa,Bsxyz,xyz->BSayz', sz, yi, self.ddx1,x,coef12,optimize=True) )
    
        return Hsocdav.reshape(xdav.shape)


    def ps_ham(self,term1,term2,term3,coeffgammaerfy,coeffgammaerfz,Ri):

        #print("term1 shape:", term1.shape)
        #print("term2 shape:", term2.shape)
        #print("term3 shape:", term3.shape)
        #
        #print("coeffgammaerfy shape",coeffgammaerfy.shape)
        #print("H.sy shape",(H.sy).shape)

        def Hx_ps(xdav):
            x = xdav.reshape((-1,)+self.bospinshape).astype(complex) 
            #print("x shape:", x.shape)
            if H.soc =='lazy':
                Vx = xp.einsum('sS,xyz,Bsxyz->BSxyz',self.si,self.Vgrid[Ri],x,optimize=True)
                Hpsdav = (
                    Vx + self.Tx(x) + self.soc_naive(x,Ri)
                    +xp.einsum('sS,xayz,Bsxyz->BSayz', self.si, term1, x, optimize=True) 
                    +xp.einsum('sS,xybz,Bsxyz->BSxbz', self.si, term2, x, optimize=True) 
                    +xp.einsum('sS,xyzc,Bsxyz->BSxyc', self.si, term3, x, optimize=True)
                    +xp.einsum('sS,xyz,Bsxyz->BSxyz',self.sy,coeffgammaerfy,x,optimize=True)
                    +xp.einsum('sS,xyz,Bsxyz->BSxyz',self.sx,coeffgammaerfz,x,optimize=True)
                )
            elif H.soc =='full':
                Vx = xp.einsum('sS,xyz,Bsxyz->BSxyz',self.si,self.Vgrid[Ri],x,optimize=True)
                Hpsdav = (
                    Vx + self.Tx(x) + self.soc_full(x,Ri)
                    +xp.einsum('sS,xayz,Bsxyz->BSayz', self.si, term1, x, optimize=True) 
                    +xp.einsum('sS,xybz,Bsxyz->BSxbz', self.si, term2, x, optimize=True) 
                    +xp.einsum('sS,xyzc,Bsxyz->BSxyc', self.si, term3, x, optimize=True)
                    +xp.einsum('sS,xyz,Bsxyz->BSxyz',self.sy,coeffgammaerfy,x,optimize=True)
                    +xp.einsum('sS,xyz,Bsxyz->BSxyz',self.sx,coeffgammaerfz,x,optimize=True)
                )
            elif H.soc =='no_spin_erf':
                Vx = xp.einsum('sS,xyz,Bsxyz->BSxyz',self.si,self.Vgrid[Ri],x,optimize=True)
                Hpsdav = (
                    Vx + self.Tx(x) + self.soc_full(x,Ri)
                    +xp.einsum('sS,xayz,Bsxyz->BSayz', self.si, term1, x, optimize=True) 
                    +xp.einsum('sS,xybz,Bsxyz->BSxbz', self.si, term2, x, optimize=True) 
                    +xp.einsum('sS,xyzc,Bsxyz->BSxyc', self.si, term3, x, optimize=True)
                )
            elif H.soc =='no_soc':
                Vx = xp.einsum('sS,xyz,Bsxyz->BSxyz',self.si,self.Vgrid[Ri],x,optimize=True)
                Hpsdav = (
                    Vx + self.Tx(x)
                    +xp.einsum('sS,xayz,Bsxyz->BSayz', self.si, term1, x, optimize=True) 
                    +xp.einsum('sS,xybz,Bsxyz->BSxbz', self.si, term2, x, optimize=True) 
                    +xp.einsum('sS,xyzc,Bsxyz->BSxyc', self.si, term3, x, optimize=True)
                    +xp.einsum('sS,xyz,Bsxyz->BSxyz',self.sy,coeffgammaerfy,x,optimize=True)
                    +xp.einsum('sS,xyz,Bsxyz->BSxyz',self.sx,coeffgammaerfz,x,optimize=True)
                )
            return Hpsdav.reshape(xdav.shape)

        return Hx_ps

    def Hbo_dav(self,Ri):

        def Hxbo(xdav):
            #print("xdav shape:", xdav.shape)
            x = xdav.reshape((-1,)+self.bospinshape)
            if self.soc =='lazy':
                Hbodav = (
                    self.Vgrid[Ri]*x + self.Tx(x) + self.soc_naive(x,Ri)
                )                
            elif self.soc =='full':
                
                Vx = xp.einsum('sS,xyz,Bsxyz->BSxyz',self.si,self.Vgrid[Ri],x,optimize=True)                
                Hbodav = (
                    Vx + self.Tx(x) + self.soc_full(x,Ri)
                )               
            elif self.soc == 'no_soc':
                Vx = xp.einsum('sS,xyz,Bsxyz->BSxyz',self.si,self.Vgrid[Ri],x,optimize=True)
                Hbodav = (
                    self.Vgrid[Ri]*x + self.Tx(x)
                ) 

            return Hbodav.reshape(xdav.shape)
        return Hxbo

    def buildDiag(self,Ri):
        NR,Nx,Ny,Nz = self.shape
        ke  = xp.zeros([Nx,Ny,Nz])
        ke += xp.diag(self.ddx2)[:,None,None]
        ke += xp.diag(self.ddy2)[None,:,None]
        ke += xp.diag(self.ddz2)[None,None,:]
        ke *= -1 / (2*self.mur)
        diag = self.Vgrid[Ri] + ke #XXXXXFix Vgrid
        diagravel = diag.ravel()
        diagspin = xp.append(diagravel,diagravel)
        return diagspin

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
 
 
    Gammaerfya = (M2*gammaerf1ya-M1*gammaerf2ya)/(M1+M2)
    Gammaerfyb = (M2*gammaerf1yb-M1*gammaerf2yb)/(M1+M2)
    Gammaerfyc = (M2*gammaerf1yc-M1*gammaerf2yc)/(M1+M2)
 
    Gammaerfza = (M2*gammaerf1za-M1*gammaerf2za)/(M1+M2)
    Gammaerfzb = (M2*gammaerf1zb-M1*gammaerf2zb)/(M1+M2)

    return Gammaerfya, Gammaerfyb, Gammaerfyc, Gammaerfza, Gammaerfzb


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
    parser.add_argument('-Pphi', default=0, type=float)
    parser.add_argument('-Ptheta', default=0, type=float)
    parser.add_argument('-alpha', default=0, type=float)
    parser.add_argument('-R', dest="NR", metavar="NR", default=101, type=int)
    parser.add_argument('-x', dest="Nx", metavar="Nx", default=250, type=int)
    parser.add_argument('-y', dest="Ny", metavar="Ny", default=250, type=int)
    parser.add_argument('-z', dest="Nz", metavar="Nz", default=250, type=int)
    parser.add_argument('--bo_spectrum', metavar='bo_spectrum', default=False, type=bool)
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

    #kwargs = dict(optimize=True)
    #if xp.backend == 'torch':
    #    kwargs = {}
    #xp.einsum(..., **kwargs)  

    H = Hamiltonian(args)
    start_script = perf_counter()   
    NR,Nx,Ny,Nz = H.shape
    Nelec = 2*Nx*Ny*Nz 
    
    Ad_nsg = xp.zeros(NR)
    Ad_nse = xp.zeros(NR)

    Rval, Pval = H.RP_grid

    EPSg = xp.zeros((NR, NR))
    EPSe = xp.zeros((NR, NR))
    gammacoeff_R = -1j*(Pval-1/Rval)/H.mu12 
    gammacoeff_phi = +1j*(H.Pphi/H.R)/H.mu12
    gammacoeff_theta = +1j*(H.Ptheta/H.R-1/H.R)/H.mu12

    ivalg = xp.zeros([NR,1])
    ivale = xp.zeros([NR,1])

    energy_bo = xp.zeros([NR,args.k])
    #evecs_bo = xp.zeros([NR,Nelec],dtype=complex)
    #print("evecs",evecs_bo.shape)

    ## Start the loops from the middle of the bond, for optimal guesses
    # iR = int(NR/2)
    iR = NR//2
    #iR = 0
    print("iR",iR)
    sequence = chain(
        [iR],
        range(iR - 1, -1, -1),
        range(iR + 1, NR)
    )
    gammacoeff = (gammacoeff_R, gammacoeff_phi, gammacoeff_theta)

    if (args.bo_spectrum==True):
        Ad_nsg, Ad_nse, ivalg, ivale = H.BO_energies(args,iR,sequence)

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
        print("EPSv_bo g.s.",EPSv_bog[0:10])

        EPS_boe = xp.zeros((H.shape[0], H.shape[0]))
        Helmate = xp.repeat(ivale,H.shape[0],axis=1)
        EPS_boe += Helmate   
        EPS_boe += 1/(2*H.mu12)*(Pval**2+H.Pphi**2/Rval**2+H.Ptheta**2/Rval**2+1/(2*Rval)**2)
        HPS_boe = inverse_weyl_transform(EPS_boe, H.shape[0], H.R, H.P_R)
        EPSv_boe = batch_eigvalsh(HPS_boe)
        print("EPSv_bo e.s.",EPSv_boe[0:10])

        if args.evecs:
            Hbo_g = +1/(2*H.mu12)*(-H.ddR2 + xp.diag(H.Pphi**2/H.R**2)+ xp.diag(H.Ptheta**2/H.R**2)+xp.diag(1/(2*H.R)**2)) +xp.diag(Ad_nsg)
            Ad_vn_g, evecsvib_g = xp.linalg.eigh(Hbo_g)

            numpy.savez_compressed(args.evecs, evecs_bo_g=evecsvib_g, e_bo_g=Ad_nsg, Rval=H.R)
            #numpy.savez_compressed(args.evecs, guess=evecsvib_g, e_approx=Ad_vn_g, Rval=H.R)
            print("Wrote eigenvectors to", args.evecs)

        exit()

    
    with timer_ctx(f"R for loop"):
        for i in sequence:
            print("Atom Ri idx",i, "Atom Ri",H.R[i],flush=True)
            diag = H.buildDiag(i)   

            guess_ns = xp.exp(-(H.Vgrid[i] - xp.min(H.Vgrid[i]))**2/27.211**2).ravel()
            guess_zeros = xp.zeros(len(guess_ns))
            #guess_spin = xp.repeat(guess_ns, 2)
            #guess_spin = xp.append(guess_ns, guess_ns)
            guess_spin = xp.array([xp.append(guess_ns, guess_zeros),xp.append(guess_zeros, guess_ns)])
            if i==iR:
                guess_bo = guess_spin
            else:
                guess_bo = evecs

            #guess_spin = xp.append(guess_ns, guess_ns)
            conv, e_approx, evecs = lib.davidson1(
                H.Hbo_dav(i),
                guess_bo,
                lambda dx, e, x0: dx/(diag-e+1e-5),
                nroots=args.k,
                max_cycle=args.iterations,
                verbose=args.verbosity,
                max_space=args.subspace,
                max_memory=get_davidson_mem(0.75),
                #tol=1e-12, #FIXME:DEBUG
                tol=1e-8,
            )
            print("Davidson:", e_approx)
            print(conv)
            #if not xp.all(conv):
            #    print("Davidson failed for atom Ri",i)
            #    exit()
            Ad_nsg[i] = e_approx[0]
            Ad_nse[i] = e_approx[1]
            ivalg[i,0] = e_approx[0]
            ivale[i,0] = e_approx[1]
            #energy_bo[i,:] = e_approx
            #print("evecs",evecs.shape)
            #evecs_bo[i,:] = evecs[0,:]
    
            r1e2, r2e2 = H.V(H.R[i], H.x_grid, H.y_grid, H.z_grid, spitvals=True)
            theta1 = xp.exp(-r1e2)
            theta2 = xp.exp(-r2e2)
            partition = theta1 + theta2
    
            t1 = theta1/partition
            t2 = theta2/partition
    
            gammaetf1x,gammaetf1y,gammaetf1z = Gamma_etf(H.R[i],H.ddx1,H.ddy1,H.ddz1,t1)
            gammaetf2x,gammaetf2y,gammaetf2z = Gamma_etf(H.R[i],H.ddx1,H.ddy1,H.ddz1,t2)
            gammaetf = (gammaetf1x, gammaetf1y, gammaetf1z, gammaetf2x, gammaetf2y, gammaetf2z)
            gammaerfya, gammaerfyb, gammaerfyc, gammaerfza, gammaerfzb = Gamma_erf_orb(H.R[i],H.x,H.y,H.z,H.M_1,H.M_2,H.mu12,gammaetf,t1,t2)
            gammaerfsy, gammaerfsz = Gamma_erf_spin(H.R[i],H.M_1,H.M_2,t1,t2)

            gammaetfx = (H.M_2*gammaetf1x-H.M_1*gammaetf2x)/(H.M_1+H.M_2)
            gammaetfy = (H.M_2*gammaetf1y-H.M_1*gammaetf2y)/(H.M_1+H.M_2)
            gammaetfz = (H.M_2*gammaetf1z-H.M_1*gammaetf2z)/(H.M_1+H.M_2)
                
            
            term2 = (
                    gammacoeff_phi[i] * gammaetfy +
                    gammacoeff_theta[i] * gammaerfzb
                )
#
            term3 = (
                    gammacoeff_phi[i] * (gammaerfyb + gammaerfyc) +
                    gammacoeff_theta[i] * (gammaetfz + gammaerfza)
                )

            coeffgammaerfy = gammacoeff_phi[i]*gammaerfsy
            coeffgammaerfz = gammacoeff_theta[i]*gammaerfsz
            
            with timer_ctx(f"P for loop"):
                Pseq = [NR//2 -i for i in range(NR//2+1)] + [NR//2+i+1 for i in range(NR//2-1)]
                print("Pseq", Pseq)
                for j in Pseq:
                    #j=0
                    print("Atom Ri",i,"Atom Pj",j,flush=True)

                    term1 = (
                        gammacoeff_R[i,j] * gammaetfx +
                        gammacoeff_phi[i] * gammaerfya
                    )
                    if i==iR and j==NR//2:
                        guess_ps = evecs
                    else:
                        guess_ps = evecs_save
                    #guess_ps = evecs
                    with timer_ctx(f"Davidson of size {H.size}"):
                        conv, e_ps_approx, evecs_save = lib.davidson1(
                            H.ps_ham(term1,term2,term3,coeffgammaerfy,coeffgammaerfz,i),
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
                    
                    
    #EPS = xp.loadtxt("rij_matrix.txt")
    #ivalload = xp.loadtxt("ri_values.txt")
    #ival = ivalload.reshape([NR,1])
    #Ad_n= ivalload

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
    EPSvg = batch_eigvalsh(HPSg)
    print("EPSv g.s.",EPSvg[0:10])
    print("PS vib gap g.s.",EPSvg[1]-EPSvg[0],flush=True)

    EPSe += 1/(2*H.mu12)*(Pval**2+H.Pphi**2/Rval**2+H.Ptheta**2/Rval**2+1/(2*Rval)**2)
    HPSe = inverse_weyl_transform(EPSe, H.shape[0], H.R, H.P_R)
    EPSve = batch_eigvalsh(HPSe)
    print("EPSv e.s.",EPSve[0:10])
    print("PS vib gap e.s.",EPSve[1]-EPSve[0],flush=True)

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
    #print("Weyl BO vib gap",EPSv_bo[1]-EPSv_bo[0],flush=True)

    if args.evecs:
        Hbo_g = +1/(2*H.mu12)*(-H.ddR2 + xp.diag(H.Pphi**2/H.R**2)+ xp.diag(H.Ptheta**2/H.R**2)+xp.diag(1/(2*H.R)**2)) +xp.diag(Ad_nsg)
        Ad_vn_g, evecsvib_g = xp.linalg.eigh(Hbo_g)

        numpy.savez_compressed(args.evecs, evecs_bo_g=evecsvib_g, e_bo_g=Ad_nsg, Rval=H.R)
        #numpy.savez_compressed(args.evecs, guess=evecsvib_g, e_approx=Ad_vn_g, Rval=H.R)
        print("Wrote eigenvectors to", args.evecs)

