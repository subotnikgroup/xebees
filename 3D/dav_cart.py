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
            'borgis': (potentials.borgis, potentials.extents_borgis),
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

def Gamma_etf_old(R,rx,ry,rz,ddx,ddy,ddz,M_1,M_2,mu12,r1e2,r2e2):
    
    Nx = len(ddx)
    Ny = len(ddy)
    Nz = len(ddz)
    
    theta1 = xp.exp(-r1e2)
    theta2 = xp.exp(-r2e2)
    partition = theta1 + theta2

    t1_old = xp.diag((theta1/partition).ravel())
    t2_old = xp.diag((theta2/partition).ravel())

    px_old =  xp.kron(xp.kron(ddx, xp.eye(Ny)), xp.eye(Nz))
    py_old =  xp.kron(xp.kron(xp.eye(Nx), ddy), xp.eye(Nz))
    pz_old =  xp.kron(xp.kron(xp.eye(Nx), xp.eye(Ny)), ddz)
    
    t1px_old = xp.dot(t1_old,px_old)
    pxt1_old = xp.dot(px_old,t1_old)
    t2px_old = xp.dot(t2_old,px_old)
    pxt2_old = xp.dot(px_old,t2_old)

    t1py_old = xp.dot(t1_old,py_old)
    pyt1_old = xp.dot(py_old,t1_old)
    t2py_old = xp.dot(t2_old,py_old)
    pyt2_old = xp.dot(py_old,t2_old)

    t1pz_old = xp.dot(t1_old,pz_old)
    pzt1_old = xp.dot(pz_old,t1_old)
    t2pz_old = xp.dot(t2_old,pz_old)
    pzt2_old = xp.dot(pz_old,t2_old)

    gammaetf1x_old = -0.5*(t1px_old + pxt1_old)
    gammaetf1y_old = -0.5*(t1py_old + pyt1_old)
    gammaetf1z_old = -0.5*(t1pz_old + pzt1_old)

    gammaetf2x_old = -0.5*(t2px_old + pxt2_old)   
    gammaetf2y_old = -0.5*(t2py_old + pyt2_old)
    gammaetf2z_old = -0.5*(t2pz_old + pzt2_old)

    return gammaetf1x_old, gammaetf1y_old, gammaetf1z_old, gammaetf2x_old, gammaetf2y_old, gammaetf2z_old

    
def Gamma_erf_old(R,rx,ry,rz,ddx,ddy,ddz,M_1,M_2,mu12,r1e2,r2e2):
    
    Nx = len(ddx)
    Ny = len(ddy)
    Nz = len(ddz)
    
    theta1 = xp.exp(-r1e2)
    theta2 = xp.exp(-r2e2)
    partition = theta1 + theta2

    t1_old = xp.diag((theta1/partition).ravel())
    t2_old = xp.diag((theta2/partition).ravel())
    
    rx = rx[:,0,0]
    ry = ry[0,:,0]
    rz = rz[0,0,:]

    def kron3(Ox,Oy,Oz):
        return xp.kron(xp.kron(Ox,Oy),Oz)
    
    J1xa = -0.5*(kron3(xp.eye(Nx),xp.diag(ry),xp.eye(Nz))@((t1_old@kron3(xp.eye(Nx),xp.eye(Ny),ddz))+(kron3(xp.eye(Nx),xp.eye(Ny),ddz)@t1_old)))
    J1xb = -0.5*(kron3(xp.eye(Nx),xp.eye(Ny),xp.diag(rz))@((t1_old@kron3(xp.eye(Nx),ddy,xp.eye(Nz)))+(kron3(xp.eye(Nx),ddy,xp.eye(Nz))@t1_old)))

    J1ya = -0.5*(kron3(xp.eye(Nx),xp.eye(Ny),xp.diag(rz))@((t1_old@kron3(ddx,xp.eye(Ny),xp.eye(Nz)))+(kron3(ddx,xp.eye(Ny),xp.eye(Nz))@t1_old)))
    J1yb = -0.5*((kron3(xp.diag(rx),xp.eye(Ny),xp.eye(Nz))-(R*mu12)/M_1*xp.eye(Nx*Ny*Nz))@((t1_old@kron3(xp.eye(Nx),xp.eye(Ny),ddz))+(kron3(xp.eye(Nx),xp.eye(Ny),ddz)@t1_old)))

    J1za = -0.5*((kron3(xp.diag(rx),xp.eye(Ny),xp.eye(Nz))-(R*mu12)/M_1*xp.eye(Nx*Ny*Nz))@((t1_old@kron3(xp.eye(Nx),ddy,xp.eye(Nz)))+(kron3(xp.eye(Nx),ddy,xp.eye(Nz))@t1_old)))
    J1zb = -0.5*(kron3(xp.eye(Nx),xp.diag(ry),xp.eye(Nz))@((t1_old@kron3(xp.eye(Nx),ddy,xp.eye(Nz)))+(kron3(xp.eye(Nx),ddy,xp.eye(Nz))@t1_old)))

    J2xa = -0.5*(kron3(xp.eye(Nx),xp.diag(ry),xp.eye(Nz))@((t2_old@kron3(xp.eye(Nx),xp.eye(Ny),ddz))+(kron3(xp.eye(Nx),xp.eye(Ny),ddz)@t2_old)))
    J2xb = -0.5*(kron3(xp.eye(Nx),xp.eye(Ny),xp.diag(rz))@((t2_old@kron3(xp.eye(Nx),ddy,xp.eye(Nz)))+(kron3(xp.eye(Nx),ddy,xp.eye(Nz))@t2_old)))

    J2ya = -0.5*(kron3(xp.eye(Nx),xp.eye(Ny),xp.diag(rz))@((t2_old@kron3(ddx,xp.eye(Ny),xp.eye(Nz)))+(kron3(ddx,xp.eye(Ny),xp.eye(Nz))@t2_old)))
    #J2ya = -0.5*((t2_old@kron3(ddx,xp.eye(Ny),xp.eye(Nz)))+(kron3(ddx,xp.eye(Ny),xp.eye(Nz))@t2_old))
    J2yb = -0.5*((kron3(xp.diag(rx),xp.eye(Ny),xp.eye(Nz))-(R*mu12)/M_2*xp.eye(Nx*Ny*Nz))@((t2_old@kron3(xp.eye(Nx),xp.eye(Ny),ddz))+(kron3(xp.eye(Nx),xp.eye(Ny),ddz)@t2_old)))

    J2za = -0.5*((kron3(xp.diag(rx),xp.eye(Ny),xp.eye(Nz))-(R*mu12)/M_2*xp.eye(Nx*Ny*Nz))@((t2_old@kron3(xp.eye(Nx),ddy,xp.eye(Nz)))+(kron3(xp.eye(Nx),ddy,xp.eye(Nz))@t2_old)))
    J2zb = -0.5*(kron3(xp.eye(Nx),xp.diag(ry),xp.eye(Nz))@((t2_old@kron3(xp.eye(Nx),ddy,xp.eye(Nz)))+(kron3(xp.eye(Nx),ddy,xp.eye(Nz))@t2_old)))

    J1x = J1xa-J1xb
    J1y = J1ya-J1yb
    J1z = J1za-J1zb

    J2x = J2xa-J2xb
    J2y = J2ya-J2yb
    J2z = J2za-J2zb
        
    gammaerf1x_old_tmp = xp.zeros([Nx*Ny*Nz,Nx*Ny*Nz])
    gammaerf2x_old_tmp = xp.zeros([Nx*Ny*Nz,Nx*Ny*Nz])
    gammaerf1y_old_tmp = -1/R*(-J1y-J2y)
    gammaerf1z_old_tmp = -1/R*(J1x+J2x)
    gammaerf2y_old_tmp = 1/R*(-J1y-J2y)
    gammaerf2z_old_tmp = 1/R*(J1x+J2x)

    return gammaerf1x_old_tmp, gammaerf1y_old_tmp, gammaerf1z_old_tmp, gammaerf2x_old_tmp, gammaerf2y_old_tmp, gammaerf2z_old_tmp 


def Gamma_erf_old2(R,rx,ry,rz,ddx,ddy,ddz,M_1,M_2,mu12,r1e2,r2e2):
    
    Nx = len(ddx)
    Ny = len(ddy)
    Nz = len(ddz)
    
    theta1 = xp.exp(-r1e2)
    theta2 = xp.exp(-r2e2)
    partition = theta1 + theta2

    t1_old = xp.diag((theta1/partition).ravel())
    t2_old = xp.diag((theta2/partition).ravel())
    
    rx = rx[:,0,0]
    ry = ry[0,:,0]
    rz = rz[0,0,:]

    def kron3(Ox,Oy,Oz):
        return xp.kron(xp.kron(Ox,Oy),Oz)
    
    J1xa = -0.5*(kron3(xp.eye(Nx),xp.diag(ry),xp.eye(Nz))@((t1_old@kron3(xp.eye(Nx),xp.eye(Ny),ddz))+(kron3(xp.eye(Nx),xp.eye(Ny),ddz)@t1_old)))
    J1xb = -0.5*(kron3(xp.eye(Nx),xp.eye(Ny),xp.diag(rz))@((t1_old@kron3(xp.eye(Nx),ddy,xp.eye(Nz)))+(kron3(xp.eye(Nx),ddy,xp.eye(Nz))@t1_old)))

    J1ya = -0.5*(kron3(xp.eye(Nx),xp.eye(Ny),xp.diag(rz))@((t1_old@kron3(ddx,xp.eye(Ny),xp.eye(Nz)))+(kron3(ddx,xp.eye(Ny),xp.eye(Nz))@t1_old)))
    J1yb = -0.5*((kron3(xp.diag(rx),xp.eye(Ny),xp.eye(Nz))-(R*mu12)/M_1*xp.eye(Nx*Ny*Nz))@((t1_old@kron3(xp.eye(Nx),xp.eye(Ny),ddz))+(kron3(xp.eye(Nx),xp.eye(Ny),ddz)@t1_old)))

    J1za = -0.5*((kron3(xp.diag(rx),xp.eye(Ny),xp.eye(Nz))-(R*mu12)/M_1*xp.eye(Nx*Ny*Nz))@((t1_old@kron3(xp.eye(Nx),ddy,xp.eye(Nz)))+(kron3(xp.eye(Nx),ddy,xp.eye(Nz))@t1_old)))
    J1zb = -0.5*(kron3(xp.eye(Nx),xp.diag(ry),xp.eye(Nz))@((t1_old@kron3(xp.eye(Nx),ddy,xp.eye(Nz)))+(kron3(xp.eye(Nx),ddy,xp.eye(Nz))@t1_old)))

    J2xa = -0.5*(kron3(xp.eye(Nx),xp.diag(ry),xp.eye(Nz))@((t2_old@kron3(xp.eye(Nx),xp.eye(Ny),ddz))+(kron3(xp.eye(Nx),xp.eye(Ny),ddz)@t2_old)))
    J2xb = -0.5*(kron3(xp.eye(Nx),xp.eye(Ny),xp.diag(rz))@((t2_old@kron3(xp.eye(Nx),ddy,xp.eye(Nz)))+(kron3(xp.eye(Nx),ddy,xp.eye(Nz))@t2_old)))

    J2ya = -0.5*(kron3(xp.eye(Nx),xp.eye(Ny),xp.diag(rz))@((t2_old@kron3(ddx,xp.eye(Ny),xp.eye(Nz)))+(kron3(ddx,xp.eye(Ny),xp.eye(Nz))@t2_old)))
    J2yb = -0.5*((kron3(xp.diag(rx),xp.eye(Ny),xp.eye(Nz))-(R*mu12)/M_2*xp.eye(Nx*Ny*Nz))@((t2_old@kron3(xp.eye(Nx),xp.eye(Ny),ddz))+(kron3(xp.eye(Nx),xp.eye(Ny),ddz)@t2_old)))

    J2za = -0.5*((kron3(xp.diag(rx),xp.eye(Ny),xp.eye(Nz))-(R*mu12)/M_2*xp.eye(Nx*Ny*Nz))@((t2_old@kron3(xp.eye(Nx),ddy,xp.eye(Nz)))+(kron3(xp.eye(Nx),ddy,xp.eye(Nz))@t2_old)))
    J2zb = -0.5*(kron3(xp.eye(Nx),xp.diag(ry),xp.eye(Nz))@((t2_old@kron3(xp.eye(Nx),ddy,xp.eye(Nz)))+(kron3(xp.eye(Nx),ddy,xp.eye(Nz))@t2_old)))

    J1x = J1xa-J1xb
    J1y = J1ya-J1yb
    J1z = J1za-J1zb

    J2x = J2xa-J2xb
    J2y = J2ya-J2yb
    J2z = J2za-J2zb
        
    gammaerf1x_old_tmp = xp.zeros([Nx*Ny*Nz,Nx*Ny*Nz])
    gammaerf2x_old_tmp = xp.zeros([Nx*Ny*Nz,Nx*Ny*Nz])
    gammaerf1y_old_tmp = t1_old@kron3(xp.eye(Nx),xp.eye(Ny),ddz)
    gammaerf1z_old_tmp = -1/R*(J1x+J2x)
    gammaerf2y_old_tmp = 1/R*(-J1y-J2y)
    gammaerf2z_old_tmp = 1/R*(J1x+J2x)

    return gammaerf1y_old_tmp


def Gamma_etf_diag_old(R,rx,ry,rz,ddx,ddy,ddz,M_1,M_2,mu12,r1e2,r2e2):
    
    Nx = len(ddx)
    Ny = len(ddy)
    Nz = len(ddz)
    
    theta1 = xp.exp(-r1e2)
    theta2 = xp.exp(-r2e2)
    partition = theta1 + theta2

    t1_old = xp.diag((theta1/partition).ravel())
    t2_old = xp.diag((theta2/partition).ravel())
    
    px_old =  xp.kron(xp.kron(ddx, xp.eye(Ny)), xp.eye(Nz))
    py_old =  xp.kron(xp.kron(xp.eye(Nx), ddy), xp.eye(Nz))
    pz_old =  xp.kron(xp.kron(xp.eye(Nx), xp.eye(Ny)), ddz)
    
    t1px_old = xp.dot(t1_old,px_old)
    pxt1_old = xp.dot(px_old,t1_old)
    t2px_old = xp.dot(t2_old,px_old)
    pxt2_old = xp.dot(px_old,t2_old)

    t1py_old = xp.dot(t1_old,py_old)
    pyt1_old = xp.dot(py_old,t1_old)
    t2py_old = xp.dot(t2_old,py_old)
    pyt2_old = xp.dot(py_old,t2_old)

    t1pz_old = xp.dot(t1_old,pz_old)
    pzt1_old = xp.dot(pz_old,t1_old)
    t2pz_old = xp.dot(t2_old,pz_old)
    pzt2_old = xp.dot(pz_old,t2_old)

    gamma1x_old = -0.5*(t1px_old + pxt1_old)
    gamma1y_old = -0.5*(t1py_old + pyt1_old)
    gamma1z_old = -0.5*(t1pz_old + pzt1_old)

    gamma2x_old = -0.5*(t2px_old + pxt2_old)   
    gamma2y_old = -0.5*(t2py_old + pyt2_old)
    gamma2z_old = -0.5*(t2pz_old + pzt2_old)

    gammasq1x_old = xp.dot(gamma1x_old,gamma1x_old)
    gammasq2x_old = xp.dot(gamma2x_old,gamma2x_old)
    gamma1x2x_old = xp.dot(gamma1x_old,gamma2x_old)
    gamma2x1x_old = xp.dot(gamma2x_old,gamma1x_old)

    gammasq1y_old = xp.dot(gamma1y_old,gamma1y_old)
    gammasq2y_old = xp.dot(gamma2y_old,gamma2y_old)       
    gamma1y2y_old = xp.dot(gamma1y_old,gamma2y_old)
    gamma2y1y_old = xp.dot(gamma2y_old,gamma1y_old)

    gammasq1z_old = xp.dot(gamma1z_old,gamma1z_old)
    gammasq2z_old = xp.dot(gamma2z_old,gamma2z_old)       
    gamma1z2z_old = xp.dot(gamma1z_old,gamma2z_old)
    gamma2z1z_old = xp.dot(gamma2z_old,gamma1z_old)

    output = (xp.diag(gammasq1x_old), xp.diag(gammasq2x_old), xp.diag(gamma1x2x_old), xp.diag(gamma2x1x_old), 
              xp.diag(gammasq1y_old), xp.diag(gammasq2y_old), xp.diag(gamma1y2y_old), xp.diag(gamma2y1y_old),
              xp.diag(gammasq1z_old), xp.diag(gammasq2z_old), xp.diag(gamma1z2z_old), xp.diag(gamma2z1z_old))


    return output

    
    

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

def Gamma_erf(R,rx,ry,rz,ddx,ddy,ddz,M1,M2,mu12,r1e2,r2e2,*xdav):

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

    t1px_y = xp.einsum('ijk,il,Bljk->Bijk',t1,ddx,xdavy1)
    pxt1_y = xp.einsum('il,ljk,Bljk->Bijk',ddx,t1,xdavy1)
    t2px_y = xp.einsum('ijk,il,Bljk->Bijk',t2,ddx,xdavy2)
    pxt2_y = xp.einsum('il,ljk,Bljk->Bijk',ddx,t2,xdavy2)

    t1px_z = xp.einsum('ijk,il,Bljk->Bijk',t1,ddx,xdavz1)
    pxt1_z = xp.einsum('il,ljk,Bljk->Bijk',ddx,t1,xdavz1)
    t2px_z = xp.einsum('ijk,il,Bljk->Bijk',t2,ddx,xdavz2)
    pxt2_z = xp.einsum('il,ljk,Bljk->Bijk',ddx,t2,xdavz2)

    t1py_x = xp.einsum('ijk,jl,Bilk->Bijk',t1,ddy,xdavx1)
    pyt1_x = xp.einsum('il,jlk,Bjlk->Bjik',ddy,t1,xdavx1)
    t2py_x = xp.einsum('ijk,jl,Bilk->Bijk',t2,ddy,xdavx2)
    pyt2_x = xp.einsum('il,jlk,Bjlk->Bjik',ddy,t2,xdavx2)

    t1py_z = xp.einsum('ijk,jl,Bilk->Bijk',t1,ddy,xdavz1)
    pyt1_z = xp.einsum('il,jlk,Bjlk->Bjik',ddy,t1,xdavz1)
    t2py_z = xp.einsum('ijk,jl,Bilk->Bijk',t2,ddy,xdavz2)
    pyt2_z = xp.einsum('il,jlk,Bjlk->Bjik',ddy,t2,xdavz2)

    t1pz_x = xp.einsum('ijk,kl,Bijl->Bijk',t1,ddz,xdavx1)
    pzt1_x = xp.einsum('ij,klj,Bklj->Bkli',ddz,t1,xdavx1)
    t2pz_x = xp.einsum('ijk,kl,Bijl->Bijk',t2,ddz,xdavx2)
    pzt2_x = xp.einsum('ij,klj,Bklj->Bkli',ddz,t2,xdavx2)

    t1pz_y = xp.einsum('ijk,kl,Bijl->Bijk',t1,ddz,xdavy1)
    pzt1_y = xp.einsum('ij,klj,Bklj->Bkli',ddz,t1,xdavy1)
    t2pz_y = xp.einsum('ijk,kl,Bijl->Bijk',t2,ddz,xdavy2)
    pzt2_y = xp.einsum('ij,klj,Bklj->Bkli',ddz,t2,xdavy2)


    gammaetf1x_y = -0.5*(t1px_y + pxt1_y)
    gammaetf1x_z = -0.5*(t1px_z + pxt1_z)
    gammaetf1y_x = -0.5*(t1py_x + pyt1_x)
    gammaetf1y_z = -0.5*(t1py_z + pyt1_z)
    gammaetf1z_x = -0.5*(t1pz_x + pzt1_x)
    gammaetf1z_y = -0.5*(t1pz_y + pzt1_y)

    gammaetf2x_y = -0.5*(t2px_y + pxt2_y)
    gammaetf2x_z = -0.5*(t2px_z + pxt2_z)
    gammaetf2y_x = -0.5*(t2py_x + pyt2_x)
    gammaetf2y_z = -0.5*(t2py_z + pyt2_z)
    gammaetf2z_x = -0.5*(t2pz_x + pzt2_x)
    gammaetf2z_y = -0.5*(t2pz_y + pzt2_y)
    
    J1x = xp.einsum('j,Bkjl->Bkjl',ry[0,:,0],gammaetf1z_x)-xp.einsum('j,Bklj->Bklj',rz[0,0,:],gammaetf1y_x)
    J1y = xp.einsum('j,Bklj->Bklj',rz[0,0,:],gammaetf1x_y)-xp.einsum('j,Bjkl->Bjkl',(rx[:,0,0]),gammaetf1z_y)+(R*mu12/M1)*gammaetf1z_y
    J1z = xp.einsum('j,Bjkl->Bjkl',(rx[:,0,0]),gammaetf1y_z)-(R*mu12/M1)*gammaetf1y_z-xp.einsum('j,Bkjl->Bkjl',ry[0,:,0],gammaetf1x_z)

    J2x = xp.einsum('j,Bkjl->Bkjl',ry[0,:,0],gammaetf2z_x)-xp.einsum('j,Bklj->Bklj',rz[0,0,:],gammaetf2y_x)
    J2y = xp.einsum('j,Bklj->Bklj',rz[0,0,:],gammaetf2x_y)-xp.einsum('j,Bjkl->Bjkl',(rx[:,0,0]),gammaetf2z_y)+(R*mu12/M2)*gammaetf2z_y
    J2z = xp.einsum('j,Bjkl->Bjkl',(rx[:,0,0]),gammaetf2y_z)-(R*mu12/M2)*gammaetf2y_z-xp.einsum('j,Bkjl->Bkjl',ry[0,:,0],gammaetf2x_z)

    #gammaerf1x = xp.zeros([Nx,Ny,Nz])
    #gammaerf2x = xp.zeros([Nx,Ny,Nz])
    
    gammaerf1x = xp.zeros(xdavx1.shape)
    gammaerf2x = xp.zeros(xdavx1.shape)
    gammaerf1y = -1/R*(-J1y-J2y)
    gammaerf1z = -1/R*(J1x+J2x)
    gammaerf2y = 1/R*(-J1y-J2y)
    gammaerf2z = 1/R*(J1x+J2x)

    return gammaerf1x, gammaerf1y, gammaerf1z, gammaerf2x, gammaerf2y, gammaerf2z


def Gamma_erf2(R,rx,ry,rz,ddx,ddy,ddz,M1,M2,mu12,r1e2,r2e2,*xdav):

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

    t1px_y = xp.einsum('ijk,il,Bljk->Bijk',t1,ddx,xdavy1)
    pxt1_y = xp.einsum('il,ljk,Bljk->Bijk',ddx,t1,xdavy1)
    t2px_y = xp.einsum('ijk,il,Bljk->Bijk',t2,ddx,xdavy2)
    pxt2_y = xp.einsum('il,ljk,Bljk->Bijk',ddx,t2,xdavy2)

    t1px_z = xp.einsum('ijk,il,Bljk->Bijk',t1,ddx,xdavz1)
    pxt1_z = xp.einsum('il,ljk,Bljk->Bijk',ddx,t1,xdavz1)
    t2px_z = xp.einsum('ijk,il,Bljk->Bijk',t2,ddx,xdavz2)
    pxt2_z = xp.einsum('il,ljk,Bljk->Bijk',ddx,t2,xdavz2)

    t1py_x = xp.einsum('ijk,jl,Bilk->Bijk',t1,ddy,xdavx1)
    pyt1_x = xp.einsum('il,jlk,Bjlk->Bjik',ddy,t1,xdavx1)
    t2py_x = xp.einsum('ijk,jl,Bilk->Bijk',t2,ddy,xdavx2)
    pyt2_x = xp.einsum('il,jlk,Bjlk->Bjik',ddy,t2,xdavx2)

    t1py_z = xp.einsum('ijk,jl,Bilk->Bijk',t1,ddy,xdavz1)
    pyt1_z = xp.einsum('il,jlk,Bjlk->Bjik',ddy,t1,xdavz1)
    t2py_z = xp.einsum('ijk,jl,Bilk->Bijk',t2,ddy,xdavz2)
    pyt2_z = xp.einsum('il,jlk,Bjlk->Bjik',ddy,t2,xdavz2)

    t1pz_x = xp.einsum('ijk,kl,Bijl->Bijk',t1,ddz,xdavx1)
    pzt1_x = xp.einsum('ij,klj,Bklj->Bkli',ddz,t1,xdavx1)
    t2pz_x = xp.einsum('ijk,kl,Bijl->Bijk',t2,ddz,xdavx2)
    pzt2_x = xp.einsum('ij,klj,Bklj->Bkli',ddz,t2,xdavx2)

    t1pz_1y = xp.einsum('ijk,kl,Bijl->Bijk',t1,ddz,xdavy1)
    pzt1_1y = xp.einsum('ij,klj,Bklj->Bkli',ddz,t1,xdavy1)
    t1pz_2y = xp.einsum('ijk,kl,Bijl->Bijk',t1,ddz,xdavy2)
    pzt1_2y = xp.einsum('ij,klj,Bklj->Bkli',ddz,t1,xdavy2)
    t2pz_1y = xp.einsum('ijk,kl,Bijl->Bijk',t2,ddz,xdavy1)
    pzt2_1y = xp.einsum('ij,klj,Bklj->Bkli',ddz,t2,xdavy1)
    t2pz_2y = xp.einsum('ijk,kl,Bijl->Bijk',t2,ddz,xdavy2)
    pzt2_2y = xp.einsum('ij,klj,Bklj->Bkli',ddz,t2,xdavy2)


    gammaetf1x_1y = -0.5*(t1px_1y + pxt1_1y)
    gammaetf1x_2y = -0.5*(t1px_2y + pxt1_2y)
    gammaetf1x_z = -0.5*(t1px_z + pxt1_z)
    gammaetf1y_x = -0.5*(t1py_x + pyt1_x)
    gammaetf1y_z = -0.5*(t1py_z + pyt1_z)
    gammaetf1z_x = -0.5*(t1pz_x + pzt1_x)
    gammaetf1z_y = -0.5*(t1pz_y + pzt1_y)

    gammaetf2x_y = -0.5*(t2px_y + pxt2_y)
    gammaetf2x_z = -0.5*(t2px_z + pxt2_z)
    gammaetf2y_x = -0.5*(t2py_x + pyt2_x)
    gammaetf2y_z = -0.5*(t2py_z + pyt2_z)
    gammaetf2z_x = -0.5*(t2pz_x + pzt2_x)
    gammaetf2z_y = -0.5*(t2pz_y + pzt2_y)
    
    J1x = xp.einsum('j,Bkjl->Bkjl',ry[0,:,0],gammaetf1z_x)-xp.einsum('j,Bklj->Bklj',rz[0,0,:],gammaetf1y_x)
    J1y = xp.einsum('j,Bklj->Bklj',rz[0,0,:],gammaetf1x_1y)-xp.einsum('j,Bjkl->Bjkl',(rx[:,0,0]),gammaetf1z_1y)+(R*mu12/M1)*gammaetf1z_1y
    J1z = xp.einsum('j,Bjkl->Bjkl',(rx[:,0,0]),gammaetf1y_z)-(R*mu12/M1)*gammaetf1y_z-xp.einsum('j,Bkjl->Bkjl',ry[0,:,0],gammaetf1x_z)

    J2x = xp.einsum('j,Bkjl->Bkjl',ry[0,:,0],gammaetf2z_x)-xp.einsum('j,Bklj->Bklj',rz[0,0,:],gammaetf2y_x)
    J2y = xp.einsum('j,Bklj->Bklj',rz[0,0,:],gammaetf2x_2y)-xp.einsum('j,Bjkl->Bjkl',(rx[:,0,0]),gammaetf2z_2y)+(R*mu12/M2)*gammaetf2z_2y
    J2z = xp.einsum('j,Bjkl->Bjkl',(rx[:,0,0]),gammaetf2y_z)-(R*mu12/M2)*gammaetf2y_z-xp.einsum('j,Bkjl->Bkjl',ry[0,:,0],gammaetf2x_z)

    #gammaerf1x = xp.zeros([Nx,Ny,Nz])
    #gammaerf2x = xp.zeros([Nx,Ny,Nz])
    
    gammaerf1x = xp.zeros(xdavx1.shape)
    gammaerf2x = xp.zeros(xdavx1.shape)
    gammaerf1y = t1pz_y
    gammaerf1z = -1/R*(J1x+J2x)
    gammaerf2y = 1/R*(-J1y-J2y)
    gammaerf2z = 1/R*(J1x+J2x)

    return gammaerf1y


def Gamma_etf_diag_new(R,rx,ry,rz,ddx,ddy,ddz,M_1,M_2,mu12,r1e2,r2e2):

    Nx = len(ddx)
    Ny = len(ddy)
    Nz = len(ddz)

    theta1 = xp.exp(-r1e2)
    theta2 = xp.exp(-r2e2)
    partition = theta1 + theta2

    t1 = (theta1/partition)
    t2 = (theta2/partition)

    t1x = xp.diag(t1[:,0,0])
    t1y = xp.diag(t1[0,:,0])
    t1z = xp.diag(t1[0,0,:])

    t2x = xp.diag(t2[:,0,0])
    t2y = xp.diag(t2[0,:,0])
    t2z = xp.diag(t2[0,0,:])

    t1px_t1px = xp.einsum('ij,jk,kl,li->i', t1x, ddx, t1x, ddx)
    t1px_pxt1 = xp.einsum('ij,jk,kl,li->i', t1x, ddx, ddx, t1x)
    pxt1_pxt1 = xp.einsum('ij,jk,kl,li->i', ddx, t1x, ddx, t1x)
    pxt1_t1px = xp.einsum('ij,jk,kl,li->i', ddx, t1x, t1x, ddx)

    diag_gammasq1x = 0.25*xp.kron(xp.kron(t1px_t1px+t1px_pxt1+pxt1_pxt1+pxt1_t1px,xp.ones(Ny)),xp.ones(Nz))

    t1px_t2px = xp.einsum('ij,jk,kl,li->i', t1x, ddx, t2x, ddx)
    t1px_pxt2 = xp.einsum('ij,jk,kl,li->i', t1x, ddx, ddx, t2x)
    pxt1_pxt2 = xp.einsum('ij,jk,kl,li->i', ddx, t1x, ddx, t2x)
    pxt1_t2px = xp.einsum('ij,jk,kl,li->i', ddx, t1x, t2x, ddx)

    diag_gamma1x2x = 0.25*xp.kron(xp.kron(t1px_t2px+t1px_pxt2+pxt1_pxt2+pxt1_t2px,xp.ones(Ny)),xp.ones(Nz))

    t2px_t1px = xp.einsum('ij,jk,kl,li->i', t2x, ddx, t1x, ddx)
    t2px_pxt1 = xp.einsum('ij,jk,kl,li->i', t2x, ddx, ddx, t1x)
    pxt2_pxt1 = xp.einsum('ij,jk,kl,li->i', ddx, t2x, ddx, t1x)
    pxt2_t1px = xp.einsum('ij,jk,kl,li->i', ddx, t2x, t1x, ddx)

    diag_gamma2x1x = 0.25*xp.kron(xp.kron(t2px_t1px+t2px_pxt1+pxt2_pxt1+pxt2_t1px,xp.ones(Ny)),xp.ones(Nz))
    
    t2px_t2px = xp.einsum('ij,jk,kl,li->i', t2x, ddx, t2x, ddx)
    t2px_pxt2 = xp.einsum('ij,jk,kl,li->i', t2x, ddx, ddx, t2x)
    pxt2_pxt2 = xp.einsum('ij,jk,kl,li->i', ddx, t2x, ddx, t2x)
    pxt2_t2px = xp.einsum('ij,jk,kl,li->i', ddx, t2x, t2x, ddx)

    diag_gammasq2x = 0.25*xp.kron(xp.kron(t2px_t2px+t2px_pxt2+pxt2_pxt2+pxt2_t2px,xp.ones(Ny)),xp.ones(Nz)) 

    t1py_t1py = xp.einsum('ij,jk,kl,li->i', t1y, ddy, t1y, ddy)
    t1py_pyt1 = xp.einsum('ij,jk,kl,li->i', t1y, ddy, ddy, t1y)
    pyt1_pyt1 = xp.einsum('ij,jk,kl,li->i', ddy, t1y, ddy, t1y)
    pyt1_t1py = xp.einsum('ij,jk,kl,li->i', ddy, t1y, t1y, ddy)

    diag_gammasq1y = 0.25*xp.kron(xp.kron(xp.ones(Nx),t1py_t1py+t1py_pyt1+pyt1_pyt1+pyt1_t1py),xp.ones(Nz)) 

    t1py_t2py = xp.einsum('ij,jk,kl,li->i', t1y, ddy, t2y, ddy)
    t1py_pyt2 = xp.einsum('ij,jk,kl,li->i', t1y, ddy, ddy, t2y)
    pyt1_pyt2 = xp.einsum('ij,jk,kl,li->i', ddy, t1y, ddy, t2y)
    pyt1_t2py = xp.einsum('ij,jk,kl,li->i', ddy, t1y, t2y, ddy)

    diag_gamma1y2y = 0.25*xp.kron(xp.kron(xp.ones(Nx),t1py_t2py+t1py_pyt2+pyt1_pyt2+pyt1_t2py),xp.ones(Nz)) 

    t2py_t1py = xp.einsum('ij,jk,kl,li->i', t2y, ddy, t1y, ddy)
    t2py_pyt1 = xp.einsum('ij,jk,kl,li->i', t2y, ddy, ddy, t1y)
    pyt2_pyt1 = xp.einsum('ij,jk,kl,li->i', ddy, t2y, ddy, t1y)
    pyt2_t1py = xp.einsum('ij,jk,kl,li->i', ddy, t2y, t1y, ddy)

    diag_gamma2y1y = 0.25*xp.kron(xp.kron(xp.ones(Nx),t2py_t1py+t2py_pyt1+pyt2_pyt1+pyt2_t1py),xp.ones(Nz))
    
    t2py_t2py = xp.einsum('ij,jk,kl,li->i', t2y, ddy, t2y, ddy)
    t2py_pyt2 = xp.einsum('ij,jk,kl,li->i', t2y, ddy, ddy, t2y)
    pyt2_pyt2 = xp.einsum('ij,jk,kl,li->i', ddy, t2y, ddy, t2y)
    pyt2_t2py = xp.einsum('ij,jk,kl,li->i', ddy, t2y, t2y, ddy)

    diag_gammasq2y = 0.25*xp.kron(xp.kron(xp.ones(Nx),t2py_t2py+t2py_pyt2+pyt2_pyt2+pyt2_t2py),xp.ones(Nz)) 

    t1pz_t1pz = xp.einsum('ij,jk,kl,li->i', t1z, ddz, t1z, ddz)
    t1pz_pzt1 = xp.einsum('ij,jk,kl,li->i', t1z, ddz, ddz, t1z)
    pzt1_pzt1 = xp.einsum('ij,jk,kl,li->i', ddz, t1z, ddz, t1z)
    pzt1_t1pz = xp.einsum('ij,jk,kl,li->i', ddz, t1z, t1z, ddz)

    diag_gammasq1z = 0.25*xp.kron(xp.kron(xp.ones(Nx),xp.ones(Ny)),t1pz_t1pz+t1pz_pzt1+pzt1_pzt1+pzt1_t1pz) 

    t1pz_t2pz = xp.einsum('ij,jk,kl,li->i', t1z, ddz, t2z, ddz)
    t1pz_pzt2 = xp.einsum('ij,jk,kl,li->i', t1z, ddz, ddz, t2z)
    pzt1_pzt2 = xp.einsum('ij,jk,kl,li->i', ddz, t1z, ddz, t2z)
    pzt1_t2pz = xp.einsum('ij,jk,kl,li->i', ddz, t1z, t2z, ddz)

    diag_gamma1z2z = 0.25*xp.kron(xp.kron(xp.ones(Nx),xp.ones(Ny)),t1pz_t2pz+t1pz_pzt2+pzt1_pzt2+pzt1_t2pz) 

    t2pz_t1pz = xp.einsum('ij,jk,kl,li->i', t2z, ddz, t1z, ddz)
    t2pz_pzt1 = xp.einsum('ij,jk,kl,li->i', t2z, ddz, ddz, t1z)
    pzt2_pzt1 = xp.einsum('ij,jk,kl,li->i', ddz, t2z, ddz, t1z)
    pzt2_t1pz = xp.einsum('ij,jk,kl,li->i', ddz, t2z, t1z, ddz)

    diag_gamma2z1z = 0.25*xp.kron(xp.kron(xp.ones(Nx),xp.ones(Ny)),t2pz_t1pz+t2pz_pzt1+pzt2_pzt1+pzt2_t1pz)

    t2pz_t2pz = xp.einsum('ij,jk,kl,li->i', t2z, ddz, t2z, ddz)
    t2pz_pzt2 = xp.einsum('ij,jk,kl,li->i', t2z, ddz, ddz, t2z)
    pzt2_pzt2 = xp.einsum('ij,jk,kl,li->i', ddz, t2z, ddz, t2z)
    pzt2_t2pz = xp.einsum('ij,jk,kl,li->i', ddz, t2z, t2z, ddz)

    diag_gammasq2z = 0.25*xp.kron(xp.kron(xp.ones(Nx),xp.ones(Ny)),t2pz_t2pz+t2pz_pzt2+pzt2_pzt2+pzt2_t2pz)


    output = (diag_gammasq1x, diag_gammasq2x, diag_gamma1x2x, diag_gamma2x1x, 
              diag_gammasq1y, diag_gammasq2y, diag_gamma1y2y, diag_gamma2y1y,
              diag_gammasq1z, diag_gammasq2z, diag_gamma1z2z, diag_gamma2z1z)

    
    return output



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

    #t1pyt1py = xp.einsum('iajb,ibjc->iajc', t1py,t1py)
    #diag_t1pyt1py = xp.einsum('iaja->iaj', t1pyt1py)
#
    #t1py_t1py = xp.einsum('ij,jk,kl,li->i',xp.diag(t1[0,:,0]) , ddy, xp.diag(t1[0,:,0]), ddy)
    #t1px_t1px = xp.einsum('ij,jk,kl,li->i',xp.diag(t1[:,0,0]) ,xp.eye(Nx), xp.diag(t1[:,0,0]), xp.eye(Nx))
    #t1pz_t1pz = xp.einsum('ij,jk,kl,li->i',xp.diag(t1[0,0,:]) ,xp.eye(Nz), xp.diag(t1[0,0,:]), xp.eye(Nz))
#
    #diag_gammasq1y = xp.kron(xp.kron(t1px_t1px,t1py_t1py),t1pz_t1pz)
    #print("first",(diag_t1pyt1py)[:,0,0])
    #print("first",(diag_t1pyt1py)[0,0,:])
    #print("t1",t1px_t1px)
    #print("t1",t1pz_t1pz)
    ##print("t1",t1[0,0,:])
    ##print("second",diag_gammasq1y)
    #print("diffx", xp.sum((diag_gammasq1y-diag_t1pyt1py.flatten())**2))
    #exit()
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




def compute_EPS(info):

    Rval, Pval, Htot_bo, gammacoeff_R, gammacoeff_phi, gammacoeff_theta, \
    Gammatotr, Gammatotp, Gammatott, Gammasqtotr, Gammasqtotp, Gammasqtott, mu12 = info
    
    #print("i,j",Rval,Pval,gammacoeff_R[Rval,Pval],flush=True)           
    
    Htot = Htot_bo[Rval]+(gammacoeff_R[Pval]*Gammatotr)+(gammacoeff_phi[Rval]*Gammatotp)+(gammacoeff_theta[Rval]*Gammatott)
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
    xdot = xdav.flatten()
    xdavnew = xdav.reshape(1,Nx,Ny,Nz)



    
    
    #for i in range(NR):
    #    print(i)
    #    r1e2, r2e2 = H.V(H.R[i], H.x_grid, H.y_grid, H.z_grid, spitvals=True)
#
    #    gammaerf1x_old, gammaerf1y_old, gammaerf1z_old,\
    #    gammaerf2x_old, gammaerf2y_old, gammaerf2z_old = Gamma_erf_old(H.R[i],H.x_grid,H.y_grid,H.z_grid,H.ddx1,H.ddy1,H.ddz1,H.M_1,H.M_2,H.mu12,r1e2,r2e2)
    #    
    #    gammaerf1x, gammaerf1y, gammaerf1z,\
    #    gammaerf2x, gammaerf2y, gammaerf2z = Gamma_erf(H.R[i],H.x_grid,H.y_grid,H.z_grid,H.ddx1,H.ddy1,H.ddz1,H.M_1,H.M_2,H.mu12,r1e2,r2e2,xdavnew)
#
    #    #print("gammasq2z_old",gammasq2z_old)
    #    #print("gammasq2z",gammasq2z)
#
    #    print("diff",xp.sum((gammaerf1x_old@xdavnew.flatten()-gammaerf1x.flatten())**2))
    #    print("diff",xp.sum((gammaerf1y_old@xdavnew.flatten()-gammaerf1y.flatten())**2))
    #    print("diff",xp.sum((gammaerf1z_old@xdavnew.flatten()-gammaerf1z.flatten())**2))
    #    print("diff",xp.sum((gammaerf2x_old@xdavnew.flatten()-gammaerf2x.flatten())**2))
    #    print("diff",xp.sum((gammaerf2y_old@xdavnew.flatten()-gammaerf2y.flatten())**2))
    #    print("diff",xp.sum((gammaerf2z_old@xdavnew.flatten()-gammaerf2z.flatten())**2))
#
    #    
    #exit()
    '''
    for i in range(NR):
        print("i",i)

        xnew = xdav.reshape((-1,)+H.boshape)
        #print("xnew",xnew.shape)
        r1e2, r2e2 = H.V(H.R[i], H.x_grid, H.y_grid, H.z_grid, spitvals=True)
        gammaetf1x_old, gammaetf1y_old, gammaetf1z_old, gammaetf2x_old, gammaetf2y_old, gammaetf2z_old, t1px_old = Gamma_etf_polar_old(H.R[i],H.x_grid,H.y_grid,H.z_grid,H.ddx1,H.ddy1,H.ddz1,H.M_1,H.M_2,H.mu12,r1e2,r2e2)
        gammaetf1x, gammaetf1y, gammaetf1z, gammaetf2x, gammaetf2y, gammaetf2z = Gamma_etf(H.R[i],H.x_grid,H.y_grid,H.z_grid,H.ddx1,H.ddy1,H.ddz1,H.M_1,H.M_2,H.mu12,r1e2,r2e2,xdavnew)
        
        
        gamma1x_old = gammaetf1x_old
        gamma2x_old = gammaetf2x_old
        gamma1y_old = gammaetf1y_old
        gamma2y_old = gammaetf2y_old
        gamma1z_old = gammaetf1z_old
        gamma2z_old = gammaetf2z_old

        Gammatotx_old = (H.M_2*gamma1x_old-H.M_1*gamma2x_old)/(H.M_1+H.M_2)
        Gammatoty_old = (H.M_2*gamma1y_old-H.M_1*gamma2y_old)/(H.M_1+H.M_2)
        Gammatotz_old = (H.M_2*gamma1z_old-H.M_1*gamma2z_old)/(H.M_1+H.M_2)

        
        Gammasqtotx_old = ((H.M_2**2*gammasq1x_old)+(H.M_1**2*gammasq2x_old)-(H.M_1*H.M_2*gamma1x2x_old)-(H.M_1*H.M_2*gamma2x1x_old))/(H.M_1+H.M_2)**2
        Gammasqtoty_old = ((H.M_2**2*gammasq1y_old)+(H.M_1**2*gammasq2y_old)-(H.M_1*H.M_2*gamma1y2y_old)-(H.M_1*H.M_2*gamma2y1y_old))/(H.M_1+H.M_2)**2
        Gammasqtotz_old = ((H.M_2**2*gammasq1z_old)+(H.M_1**2*gammasq2z_old)-(H.M_1*H.M_2*gamma1z2z_old)-(H.M_1*H.M_2*gamma2z1z_old))/(H.M_1+H.M_2)**2 
            
        
        gamma1x = gammaetf1x
        gamma2x = gammaetf2x
        gamma1y = gammaetf1y
        gamma2y = gammaetf2y
        gamma1z = gammaetf1z
        gamma2z = gammaetf2z
        Gammatotx = (H.M_2*gamma1x-H.M_1*gamma2x)/(H.M_1+H.M_2)
        Gammatoty = (H.M_2*gamma1y-H.M_1*gamma2y)/(H.M_1+H.M_2)
        Gammatotz = (H.M_2*gamma1z-H.M_1*gamma2z)/(H.M_1+H.M_2)

        gammasqetf1x, gammasqetf1y, gammasqetf1z, gammasqetf2x, gammasqetf2y, gammasqetf2z = Gamma_etf(H.R[i],H.x_grid,H.y_grid,H.z_grid,H.ddx1,H.ddy1,H.ddz1,H.M_1,H.M_2,H.mu12,r1e2,r2e2, gammaetf1x, gammaetf1y, gammaetf1z, gammaetf2x, gammaetf2y, gammaetf2z)
        gamma1x2x, gamma1y2y, gamma1z2z, gamma2x1x, gamma2y1y, gamma2z1z = Gamma_etf(H.R[i],H.x_grid,H.y_grid,H.z_grid,H.ddx1,H.ddy1,H.ddz1,H.M_1,H.M_2,H.mu12,r1e2,r2e2, gammaetf2x, gammaetf2y, gammaetf2z, gammaetf1x, gammaetf1y, gammaetf1z)
        
        Gammasqtotx = ((H.M_2**2*gammasqetf1x)+(H.M_1**2*gammasqetf2x)-(H.M_1*H.M_2*gamma1x2x)-(H.M_1*H.M_2*gamma2x1x))/(H.M_1+H.M_2)**2
        Gammasqtoty = ((H.M_2**2*gammasqetf1y)+(H.M_1**2*gammasqetf2y)-(H.M_1*H.M_2*gamma1y2y)-(H.M_1*H.M_2*gamma2y1y))/(H.M_1+H.M_2)**2
        Gammasqtotz = ((H.M_2**2*gammasqetf1z)+(H.M_1**2*gammasqetf2z)-(H.M_1*H.M_2*gamma1z2z)-(H.M_1*H.M_2*gamma2z1z))/(H.M_1+H.M_2)**2

        #print("x",xp.sum((gamma1x2x_old@xdavnew.flatten()-gamma1x2x.flatten())**2))
        #print("y",xp.sum((gamma1y2y_old@xdavnew.flatten()-gamma1y2y.flatten())**2))
        #print("z",xp.sum((gamma1z2z_old@xdavnew.flatten()-gamma1z2z.flatten())**2))

        #print("x",xp.sum((Gammasqtotx_old@xdavnew.flatten()-Gammasqtotx.flatten())**2))
        #print("y",xp.sum((Gammasqtoty_old@xdavnew.flatten()-Gammasqtoty.flatten())**2))
        #print("z",xp.sum((Gammasqtotz_old@xdavnew.flatten()-Gammasqtotz.flatten())**2))

        
        gammasq1x_old,gammasq2x_old,gamma1x2x_old,gamma2x1x_old,gammasq1y_old,gammasq2y_old,gamma1y2y_old,gamma2y1y_old,gammasq1z_old,gammasq2z_old,gamma1z2z_old,gamma2z1z_old = Gamma_etf_diag_old(H.R[i],H.x_grid,H.y_grid,H.z_grid,H.ddx1,H.ddy1,H.ddz1,H.M_1,H.M_2,H.mu12,r1e2,r2e2)
        
        
        gammaetf1x_diag = Gamma_etf_diag(H.R[i],H.x_grid,H.y_grid,H.z_grid,H.ddx1,H.ddy1,H.ddz1,H.M_1,H.M_2,H.mu12,r1e2,r2e2)
        
        print("diagx",xp.sum((gammaetf1x_diag-xp.diag(t1px_old))**2))

    '''
    

    #orig = Htot_bo[i]@xdav.flatten()
    #print("orig",orig.shape)
    ##print("orig",orig)
    #new = Hbo_dav(xdav)
    ##print("new",new)
    #print("diff",xp.linalg.norm(orig-new.flatten()))

    
    
    #testdav = xp.einsum('ij,kl,mn,jln->ikm',xp.diag(H.rinv2), xp.diag(xp.sin(H.g+xp.pi/2)**2),H.ddp2,xdav)
    #testnorm = xp.kron(xp.kron(xp.diag(H.rinv2), xp.diag(xp.sin(H.g+xp.pi/2)**2)),H.ddp2)@xdot


    #(xdav.reshape((-1,) + H.shape) * H.Vgrid).reshape(x.shape)

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
  
    Hel = -1/(2*H.mur)*(
        xp.kron(xp.kron(H.ddx2, xp.eye(Ny)), xp.eye(Nz)) \
        +xp.kron(xp.kron(xp.eye(Nx), H.ddy2), xp.eye(Nz))\
        +xp.kron(xp.kron(xp.eye(Nx), xp.eye(Ny)), H.ddz2)\
        )

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

    Htot_bo = xp.zeros([NR,Nelec,Nelec])
    Htot_bo[:] = Hel
    Htot_bo[:,xp.arange(Nelec),xp.arange(Nelec)] += xp.reshape(H.Vgrid[:],(NR,Nelec))#XXXXXcheck this 
    ival = xp.zeros([NR,1])
    Ad_n = xp.zeros(NR)


    '''        

    for i in range(NR):
        print("Atom Ri",i)
        diag = buildDiag(H,i)
        def Hbo_dav(xdav):
            x = xdav.reshape((-1,)+H.boshape)
            
            Hbodav = H.Vgrid[i]*x + Tx(x)
            return Hbodav.reshape(xdav.shape)

        #xdav = xp.random.rand(Nx,Ny,Nz)
        #orig = Htot_bo[i]@xdav.flatten()
        #print("orig",orig.shape)
        ##print("orig",orig)
        #new = Hbo_dav(xdav)
        ##print("new",new)
        #print("diff",xp.linalg.norm(orig-new.flatten()))
        #Vgridef = H.V(H.R[i], H.xb_grid, H.yb_grid, H.zb_grid, spitvals=False)
        guess_bo = xp.exp(-(H.Vgrid[i] - xp.min(H.Vgrid[i]))**2/27.211**2).ravel()
        #guess_bo = xp.random.random(H.boshape).ravel()
       # with timer_ctx(f"Davidson of size {H.size}"):
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
        #eigvals = xp.linalg.eigvalsh(Htot_bo[i])
        #print("eigvals",eigvals[0:5]) 
        #print("diff",e_approx[0]-eigvals[0])

    Rval, Pval = H.RP_grid
    
    EPS = xp.zeros((H.shape[0], H.shape[0]))
    EPSsq = xp.zeros((H.shape[0], H.shape[0]))

    

    #Gammasqtotr = xp.zeros([Nelec,Nelec,Nelec],dtype=complex)
    #Gammasqtott = xp.zeros([Nelec,Nelec,Nelec],dtype=complex)
    #Gammatotr = xp.zeros([Nelec,Nelec,Nelec],dtype=complex)
    #Gammatott = xp.zeros([Nelec,Nelec,Nelec],dtype=complex)

    '''

    #def buildDiagps(H,Ri):
    #    ke  = xp.zeros([Nx,Ny,Nz])
    #    ke += xp.diag(H.ddx2)[:,None,None]
    #    ke += xp.diag(H.ddy2)[None,:,None]
    #    ke += xp.diag(H.ddz2)[None,None,:]
    #    ke *= -1 / (2*H.mur)
    #    
    #    diag = H.Vgrid[Ri] + ke #XXXXXFix Vgrid
    #    return diag.ravel()

    #acoeff_R = -1j*H.P_R/H.mu12 
    #acoeff_phi = -1j*(H.Pphi/H.R)/H.mu12
    #acoeff_theta = +1j*(H.Ptheta/H.R)/H.mu12


    #buildDiagpssq(H,Ri):

    #(diag_gammasq1x, diag_gammasq2x, diag_gamma1x2x, diag_gamma2x1x,\
    # diag_gammasq1y, diag_gammasq2y, diag_gamma1y2y, diag_gamma2y1y,\
    # diag_gammasq1z, diag_gammasq2z, diag_gamma1z2z, diag_gamma2z1z) = Gamma_etf_diag(H.R[i],H.x_grid,H.y_grid,H.z_grid,H.ddx1,H.ddy1,H.ddz1,H.M_1,H.M_2,H.mu12,r1e2,r2e2)


    #ke  = xp.zeros([Nx,Ny,Nz])
    #ke += xp.diag(H.ddx2)[:,None,None]
    #ke += xp.diag(H.ddy2)[None,:,None]
    #ke += xp.diag(H.ddz2)[None,None,:]
    #ke += (((H.M_2**2*diag_gammasq1x)+(H.M_1**2*diag_gammasq2x)-(H.M_1*H.M_2*diag_gamma1x2x)-(H.M_1*H.M_2*diag_gamma2x1x))/(H.M_1+H.M_2)**2)
    #ke += (((H.M_2**2*diag_gammasq1y)+(H.M_1**2*diag_gammasq2y)-(H.M_1*H.M_2*diag_gamma1y2y)-(H.M_1*H.M_2*diag_gamma2y1y))/(H.M_1+H.M_2)**2)
    #ke += (((H.M_2**2*diag_gammasq1z)+(H.M_1**2*diag_gammasq2z)-(H.M_1*H.M_2*diag_gamma1z2z)-(H.M_1*H.M_2*diag_gamma2z1z))/(H.M_1+H.M_2)**2) 
    #ke *= -1 / (2*H.mur)
    #
    #diag = H.Vgrid[Ri] + ke #XXXXXFix Vgrid
    #return diag.ravel()

gammacoeff_R = -1j*H.P_R/H.mu12 
gammacoeff_phi = -1j*(H.Pphi/H.R)/H.mu12
gammacoeff_theta = +1j*(H.Ptheta/H.R)/H.mu12


for i in range(H.shape[0]):
    print("i",i,flush=True)
    r1e2, r2e2 = H.V(H.R[i], H.x_grid, H.y_grid, H.z_grid, spitvals=True)
    #diag = buildDiagpssq(H,i)
    gammaetf1x_old, gammaetf1y_old, gammaetf1z_old, gammaetf2x_old, gammaetf2y_old, gammaetf2z_old = Gamma_etf_old(H.R[i],H.x_grid,H.y_grid,H.z_grid,H.ddx1,H.ddy1,H.ddz1,H.M_1,H.M_2,H.mu12,r1e2,r2e2)
    gammaerf1x_old, gammaerf1y_old, gammaerf1z_old, gammaerf2x_old, gammaerf2y_old, gammaerf2z_old = Gamma_erf_old(H.R[i],H.x_grid,H.y_grid,H.z_grid,H.ddx1,H.ddy1,H.ddz1,H.M_1,H.M_2,H.mu12,r1e2,r2e2)
    
    for j in range(H.shape[0]):
        print("j",j)
        gamma1x_old = gammaetf1x_old+gammaerf1x_old
        gamma2x_old = gammaetf2x_old+gammaerf2x_old
        gamma1y_old = gammaetf1y_old+gammaerf1y_old
        gamma2y_old = gammaetf2y_old+gammaerf2y_old
        gamma1z_old = gammaetf1z_old+gammaerf1z_old
        gamma2z_old = gammaetf2z_old+gammaerf2z_old

        Gammatotx_old = (H.M_2*gamma1x_old-H.M_1*gamma2x_old)/(H.M_1+H.M_2)
        Gammatoty_old = (H.M_2*gamma1y_old-H.M_1*gamma2y_old)/(H.M_1+H.M_2)
        Gammatotz_old = (H.M_2*gamma1z_old-H.M_1*gamma2z_old)/(H.M_1+H.M_2)

        gammasq1x_old = xp.dot(gamma1x_old,gamma1x_old)
        gammasq2x_old = xp.dot(gamma2x_old,gamma2x_old)
        gamma1x2x_old = xp.dot(gamma1x_old,gamma2x_old)
        gamma2x1x_old = xp.dot(gamma2x_old,gamma1x_old)
        gammasq1y_old = xp.dot(gamma1y_old,gamma1y_old)
        gammasq2y_old = xp.dot(gamma2y_old,gamma2y_old)       
        gamma1y2y_old = xp.dot(gamma1y_old,gamma2y_old)
        gamma2y1y_old = xp.dot(gamma2y_old,gamma1y_old)
        gammasq1z_old = xp.dot(gamma1z_old,gamma1z_old)
        gammasq2z_old = xp.dot(gamma2z_old,gamma2z_old)       
        gamma1z2z_old = xp.dot(gamma1z_old,gamma2z_old)
        gamma2z1z_old = xp.dot(gamma2z_old,gamma1z_old)

        Gammasqtotx_old = ((H.M_2**2*gammasq1x_old)+(H.M_1**2*gammasq2x_old)-(H.M_1*H.M_2*gamma1x2x_old)-(H.M_1*H.M_2*gamma2x1x_old))/(H.M_1+H.M_2)**2
        Gammasqtoty_old = ((H.M_2**2*gammasq1y_old)+(H.M_1**2*gammasq2y_old)-(H.M_1*H.M_2*gamma1y2y_old)-(H.M_1*H.M_2*gamma2y1y_old))/(H.M_1+H.M_2)**2
        Gammasqtotz_old = ((H.M_2**2*gammasq1z_old)+(H.M_1**2*gammasq2z_old)-(H.M_1*H.M_2*gamma1z2z_old)-(H.M_1*H.M_2*gamma2z1z_old))/(H.M_1+H.M_2)**2 
        Htot_old = Htot_bo[i]+(gammacoeff_R[j]*Gammatotx_old)+(gammacoeff_phi[i]*Gammatoty_old)+(gammacoeff_theta[i]*Gammatotz_old)
        Htotsq = Htot_old - (Gammasqtotx_old +Gammasqtoty_old + Gammasqtotz_old)/(2*H.mu12)
        #e_approxsq = xp.linalg.eigvalsh(Htotsq)
    
        #print("e_approx",e_approxsq[0:5])
        
        gammaerfsq1y_old = xp.dot(gammaerf1y_old,gammaerf1y_old)
        def ps_ham(xdav):
            x = xdav.reshape((-1,)+H.boshape)
            Tx = -1/(2*H.mur)*(
                xp.einsum('ij,Bjkl->Bikl',H.ddx2,x)\
                +xp.einsum('ij,Bkjl->Bkil',H.ddy2,x)\
                +xp.einsum('ij,Bklj->Bkli',H.ddz2,x)\
                )
            gammaetf1x, gammaetf1y, gammaetf1z, gammaetf2x, gammaetf2y, gammaetf2z = Gamma_etf(H.R[i],H.x_grid,H.y_grid,H.z_grid,H.ddx1,H.ddy1,H.ddz1,H.M_1,H.M_2,H.mu12,r1e2,r2e2,x)
            gammaerf1x, gammaerf1y, gammaerf1z, gammaerf2x, gammaerf2y, gammaerf2z = Gamma_erf(H.R[i],H.x_grid,H.y_grid,H.z_grid,H.ddx1,H.ddy1,H.ddz1,H.M_1,H.M_2,H.mu12,r1e2,r2e2,x)
            
            gamma1x = gammaetf1x+gammaerf1x
            gamma2x = gammaetf2x+gammaerf2x
            gamma1y = gammaetf1y+gammaerf1y
            gamma2y = gammaetf2y+gammaerf2y
            gamma1z = gammaetf1z+gammaerf1z
            gamma2z = gammaetf2z+gammaerf2z
            Gammatotx = (H.M_2*gamma1x-H.M_1*gamma2x)/(H.M_1+H.M_2)
            Gammatoty = (H.M_2*gamma1y-H.M_1*gamma2y)/(H.M_1+H.M_2)
            Gammatotz = (H.M_2*gamma1z-H.M_1*gamma2z)/(H.M_1+H.M_2)

            gammasq1x_etf, gammasq1y_etf, gammasq1z_etf, gammasq2x_etf, gammasq2y_etf, gammasq2z_etf = Gamma_etf(H.R[i],H.x_grid,H.y_grid,H.z_grid,H.ddx1,H.ddy1,H.ddz1,H.M_1,H.M_2,H.mu12,r1e2,r2e2, gammaetf1x, gammaetf1y, gammaetf1z, gammaetf2x, gammaetf2y, gammaetf2z)
            gamma1x2x_etf, gamma1y2y_etf, gamma1z2z_etf, gamma2x1x_etf, gamma2y1y_etf, gamma2z1z_etf = Gamma_etf(H.R[i],H.x_grid,H.y_grid,H.z_grid,H.ddx1,H.ddy1,H.ddz1,H.M_1,H.M_2,H.mu12,r1e2,r2e2, gammaetf2x, gammaetf2y, gammaetf2z, gammaetf1x, gammaetf1y, gammaetf1z)
            
            gammasq1x_erf, gammasq1y_erf, gammasq1z_erf, gammasq2x_erf, gammasq2y_erf, gammasq2z_erf = Gamma_erf(H.R[i],H.x_grid,H.y_grid,H.z_grid,H.ddx1,H.ddy1,H.ddz1,H.M_1,H.M_2,H.mu12,r1e2,r2e2, gammaerf1x, gammaerf1y, gammaerf1z, gammaerf2x, gammaerf2y, gammaerf2z)
            gamma1x2x_erf, gamma1y2y_erf, gamma1z2z_erf, gamma2x1x_erf, gamma2y1y_erf, gamma2z1z_erf = Gamma_erf(H.R[i],H.x_grid,H.y_grid,H.z_grid,H.ddx1,H.ddy1,H.ddz1,H.M_1,H.M_2,H.mu12,r1e2,r2e2, gammaerf2x, gammaerf2y, gammaerf2z, gammaerf1x, gammaerf1y, gammaerf1z)
            
            gamma_etf_erf_1x, gamma_etf_erf_1y, gamma_etf_erf_1z, gamma_etf_erf_2x, gamma_etf_erf_2y, gamma_etf_erf_2z = Gamma_etf(H.R[i],H.x_grid,H.y_grid,H.z_grid,H.ddx1,H.ddy1,H.ddz1,H.M_1,H.M_2,H.mu12,r1e2,r2e2, gammaerf1x, gammaerf1y, gammaerf1z, gammaerf2x, gammaerf2y, gammaerf2z)
            gamma_erf_etf_1x, gamma_erf_etf_1y, gamma_erf_etf_1z, gamma_erf_etf_2x, gamma_erf_etf_2y, gamma_erf_etf_2z = Gamma_erf(H.R[i],H.x_grid,H.y_grid,H.z_grid,H.ddx1,H.ddy1,H.ddz1,H.M_1,H.M_2,H.mu12,r1e2,r2e2, gammaetf1x, gammaetf1y, gammaetf1z, gammaetf2x, gammaetf2y, gammaetf2z)
            
            gamma1x_etf_2x_erf, gamma1y_etf_2y_erf, gamma1z_etf_2z_erf, gamma2x_etf_1x_erf, gamma2y_etf_1y_erf, gamma2z_etf_1z_erf = Gamma_etf(H.R[i],H.x_grid,H.y_grid,H.z_grid,H.ddx1,H.ddy1,H.ddz1,H.M_1,H.M_2,H.mu12,r1e2,r2e2, gammaerf2x, gammaerf2y, gammaerf2z, gammaerf1x, gammaerf1y, gammaerf1z)
            gamma1x_erf_2x_etf, gamma1y_erf_2y_etf, gamma1z_erf_2z_etf, gamma2x_erf_1x_etf, gamma2y_erf_1y_etf, gamma2z_erf_1z_etf = Gamma_erf(H.R[i],H.x_grid,H.y_grid,H.z_grid,H.ddx1,H.ddy1,H.ddz1,H.M_1,H.M_2,H.mu12,r1e2,r2e2, gammaetf2x, gammaetf2y, gammaetf2z, gammaetf1x, gammaetf1y, gammaetf1z)
            
            gammasq1x = gammasq1x_etf + gammasq1x_erf + gamma_etf_erf_1x + gamma_erf_etf_1x
            gammasq1y = gammasq1y_etf + gammasq1y_erf + gamma_etf_erf_1y + gamma_erf_etf_1y
            gammasq1z = gammasq1z_etf + gammasq1z_erf + gamma_etf_erf_1z + gamma_erf_etf_1z
            gammasq2x = gammasq2x_etf + gammasq2x_erf + gamma_etf_erf_2x + gamma_erf_etf_2x 
            gammasq2y = gammasq2y_etf + gammasq2y_erf + gamma_etf_erf_2y + gamma_erf_etf_2y
            gammasq2z = gammasq2z_etf + gammasq2z_erf + gamma_etf_erf_2z + gamma_erf_etf_2z
            gamma1x2x = gamma1x2x_etf + gamma1x2x_erf + gamma1x_etf_2x_erf + gamma1x_erf_2x_etf
            gamma1y2y = gamma1y2y_etf + gamma1y2y_erf + gamma1y_etf_2y_erf + gamma1y_erf_2y_etf
            gamma1z2z = gamma1z2z_etf + gamma1z2z_erf + gamma1z_etf_2z_erf + gamma1z_erf_2z_etf
            gamma2x1x = gamma2x1x_etf + gamma2x1x_erf + gamma2x_etf_1x_erf + gamma2x_erf_1x_etf 
            gamma2y1y = gamma2y1y_etf + gamma2y1y_erf + gamma2y_etf_1y_erf + gamma2y_erf_1y_etf
            gamma2z1z = gamma2z1z_etf + gamma2z1z_erf + gamma2z_etf_1z_erf + gamma2z_erf_1z_etf

            gammasq1y_erf_new = Gamma_erf2(H.R[i],H.x_grid,H.y_grid,H.z_grid,H.ddx1,H.ddy1,H.ddz1,H.M_1,H.M_2,H.mu12,r1e2,r2e2, gammaerf1x, gammaerf1y, gammaerf1z, gammaerf2x, gammaerf2y, gammaerf2z)

            gammaerf1x_old, gammaerf1y_old, gammaerf1z_old, gammaerf2x_old, gammaerf2y_old, gammaerf2z_old = Gamma_erf_old(H.R[i],H.x_grid,H.y_grid,H.z_grid,H.ddx1,H.ddy1,H.ddz1,H.M_1,H.M_2,H.mu12,r1e2,r2e2)
            J1 = Gamma_erf_old2(H.R[i],H.x_grid,H.y_grid,H.z_grid,H.ddx1,H.ddy1,H.ddz1,H.M_1,H.M_2,H.mu12,r1e2,r2e2)
            gammaerfsq1y_old = xp.dot(J1,gammaerf1y_old)

            print("hello",gammasq1y_erf_new.flatten())

            print("hello2",gammaerfsq1y_old@x.flatten())

            
            print("diff2",xp.linalg.norm(gammasq1y_erf_new.flatten()-gammaerfsq1y_old@x.flatten()))
            #print("diff2",xp.sum((gammaerf1y.flatten()-gammaerf1y_old@x.flatten())**2))
            #print("diff2",xp.linalg.norm(gammasq1x.flatten()-gammasq1x_old@xdav.flatten()))
            #print("diff2",xp.linalg.norm(gammasq1y.flatten()-gammasq1y_old@xdav.flatten()))
            #print("diff2",xp.linalg.norm(gammasq1z.flatten()-gammasq1z_old@xdav.flatten()))
            #print("diff2",xp.linalg.norm(gammasq2x.flatten()-gammasq2x_old@xdav.flatten()))
            #print("diff2",xp.linalg.norm(gammasq2y.flatten()-gammasq2y_old@xdav.flatten()))
            #print("diff2",xp.linalg.norm(gammasq2z.flatten()-gammasq2z_old@xdav.flatten()))
            #print("diff2",xp.linalg.norm(gamma1x2x.flatten()-gamma1x2x_old@xdav.flatten()))
            #print("diff2",xp.linalg.norm(gamma1y2y.flatten()-gamma1y2y_old@xdav.flatten()))
            #print("diff2",xp.linalg.norm(gamma1z2z.flatten()-gamma1z2z_old@xdav.flatten()))
            #print("diff2",xp.linalg.norm(gamma2x1x.flatten()-gamma2x1x_old@xdav.flatten()))
            #print("diff2",xp.linalg.norm(gamma2y1y.flatten()-gamma2y1y_old@xdav.flatten()))
            #print("diff2",xp.linalg.norm(gamma2z1z.flatten()-gamma2z1z_old@xdav.flatten()))

            exit()
            
            Gammasqtotx = ((H.M_2**2*gammasq1x)+(H.M_1**2*gammasq2x)-(H.M_1*H.M_2*gamma1x2x)-(H.M_1*H.M_2*gamma2x1x))/(H.M_1+H.M_2)**2
            Gammasqtoty = ((H.M_2**2*gammasq1y)+(H.M_1**2*gammasq2y)-(H.M_1*H.M_2*gamma1y2y)-(H.M_1*H.M_2*gamma2y1y))/(H.M_1+H.M_2)**2
            Gammasqtotz = ((H.M_2**2*gammasq1z)+(H.M_1**2*gammasq2z)-(H.M_1*H.M_2*gamma1z2z)-(H.M_1*H.M_2*gamma2z1z))/(H.M_1+H.M_2)**2
            Hbodav = H.Vgrid[i]*x + Tx + (gammacoeff_R[j]*Gammatotx)+(gammacoeff_phi[i]*Gammatoty)+(gammacoeff_theta[i]*Gammatotz)
            Htotsq = Hbodav - (Gammasqtotx +Gammasqtoty + Gammasqtotz)/(2*H.mu12) 
            return Htotsq.reshape(xdav.shape)

        xdav = xp.random.rand(H.shape[1],H.shape[2],H.shape[3])
        xdot = xdav.flatten()
        xdavnew = xdav.reshape(1,Nx,Ny,Nz)
        print("diff",xp.linalg.norm(Htotsq@xdavnew.flatten()-ps_ham(xdavnew).flatten()))

            

            #guess_ps = xp.exp(-(H.Vgrid[i] - xp.min(H.Vgrid[i]))**2/27.211**2).ravel()
            #with timer_ctx(f"Davidson of size {H.size}"):
            #    conv, e_approx, evecs = lib.davidson1(
            #        ps_ham,
            #        guess_ps,
            #        #H.diag,
            #        #_preconditioner_naive(H, dx, e, x0,i),
            #        lambda dx, e, x0: dx/(diag-e+1e-5),
            #        nroots=args.k,
            #        max_cycle=args.iterations,
            #        verbose=args.verbosity,
            #        max_space=args.subspace,
            #        max_memory=get_davidson_mem(0.75),
            #        #tol=1e-12, #FIXME:DEBUG
            #        tol=1e-10,
            #    )
#
            #print("Davidson:", e_approx)
            #print(conv)#        


    

    '''

    for i in range(H.shape[0]):
        print("i",i,flush=True)

        r1e2, r2e2 = H.V(H.R[i], H.x_grid, H.y_grid, H.z_grid, spitvals=True)
        with timer_ctx("build gamma"):
            gammaetf1x, gammaetf1y, gammaetf1z, gammaetf2x, gammaetf2y, gammaetf2z, gammaerf1y, gammaerf1z, gammaerf2y, gammaerf2z = Gamma_etf_polar(H.R[i],H.x_grid,H.y_grid,H.z_grid,H.ddx1,H.ddy1,H.ddz1,H.M_1,H.M_2,H.mu12,r1e2,r2e2)
            
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
'''

    
    
    

    
