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

from scipy.special import sph_harm_y, lpmv, factorial
## TO-DO: make sph_harm_y also work from cupyx backend


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
import potentials
from constants import *
from hamiltonian import  KE, KE_FFT, KE_Borisov
from davidson import phase_match, phase_match_mem_constrained, get_interpolated_guess, get_davidson_mem, solve_exact_gen, eye_lazy
from debug import prms, timer, timer_ctx
from threadpoolctl import ThreadpoolController

if __name__ == '__main__':
    from tqdm import tqdm
else:  # mock this out for use in Jupyter Notebooks etc
    def tqdm(iterator, **kwargs):
        print(f"Mock call to tqdm({kwargs})")
        return iterator


class Hamiltonian:
    __slots__ = ( # any new members must be added here
        'm_e', 'M_1', 'M_2', 'mu', 'mu12', 'mur', 'aa', 'g_1', 'g_2', 'J',
        'R', 'P', 'R_grid', 'r', 'p', 'r_grid', 'g', 'pg', 'j', 'g_grid','Om','Om_grid', 'j_grid',
        'R_rgrid','r_rgrid','g_rgrid',
        'axes', 'dtype', 'args','NOm',
        'max_threads',
        'preconditioner', 'make_guess', '_Vfunc',
        'Vgrid', 'Vsph','VOm', 'ddR2', 'ddr2', 'ddg2', 'ddg1',
        'Rinv2', 'rinv2', 'diag', '_preconditioner_data',
        'shape', 'size',
        '_locked', '_hash', 'r_lab', 'R_lab', 'ddr_lab2', 'ddR_lab2'
    )

    def __init__(self, args):
        # save number of threads for preconditioner
        self.max_threads = getattr(args, "t", 1)
        self.args = args

        self.m_e = 1
        self.M_1 = args.M_1
        self.M_2 = args.M_2

        self.g_1 = args.g_1
        self.g_2 = args.g_2

        self.J   = args.J
        self.NOm = 2*self.J+1
        self.dtype = xp.float64 #if self.J == 0 else xp.complex128

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

        R_range_lab = extent[:2]
        r_max_lab   = extent[-1]

        if r_max_lab < R_range_lab[-1]/2:
            raise RuntimeError("r_max should be at least R_max/2")

        R_range = R_range_lab * self.aa
        r_max   = r_max_lab   / self.aa

        print("extent in unscaled coords:", R_range_lab, r_max_lab)
        print("extent in   scaled coords:", R_range, r_max)

        # N.B.: We are careful not to include 0 in the range of r by
        # starting 1 "step" away from 0. It might be more consistent
        # to have Nr-1 points, but the confusion this would cause
        # would be intolerable. This behavior is required because we
        # have terms that go like 1/r.
        self.r     = xp.linspace(r_max    /args.Nr, r_max, args.Nr)
        self.r_lab = xp.linspace(r_max_lab/args.Nr, r_max_lab, args.Nr)
        self.R     = xp.linspace(*R_range,     args.NR)
        self.R_lab = xp.linspace(*R_range_lab, args.NR)

        # require Ng to be even
        if args.Ng % 2 != 0:
            raise RuntimeError(f"Ng must be even!")

        # N.B.: It is essential that we not include the endpoint in
        # gamma lest our cyclic grid be ill-formed and 2nd derivatives
        # all over the place
        # N.B : in 3D gamma must exist only on the interval (0,pi)!

        #self.g = xp.asarray([i*xp.pi/args.Ng for i in range(args.Ng)])
        self.g = xp.linspace(0, xp.pi, args.Ng, endpoint=True)
        
        # self.j = xp.fft.fftfreq(args.Ng)*args.Ng
        self.j = xp.arange(0,args.Ng)
        self.j = self.j.astype(int)

        self.Om = xp.fft.fftfreq(2*self.J +1)*(2*self.J+1)
        self.Om = self.Om.astype(int)

        # self.psi = xp.asarray([i*2*xp.pi/(2*self.J+1) for i in range(0,self.Npsi)])
        #self.psi = xp.linspace(0,2*xp.pi, 2*self.J+1, endpoint=False)

        self.axes = (self.R, self.r, self.j, self.Om)

        self.R_grid, self.r_grid, self.j_grid, self.Om_grid = xp.meshgrid(self.R, self.r, self.j, self.Om, indexing='ij')
        self.R_rgrid, self.r_rgrid, self.g_rgrid = xp.meshgrid(self.R, self.r, self.g, indexing='ij')
        self.Vgrid = self.V(self.R_rgrid, self.r_rgrid, self.g_rgrid) # Vgrid in real space \propto cos(psi)cos(g)
        assert not xp.any(self.Vgrid)==xp.nan

        with timer_ctx("Build Vsph from Vgrid"):
            self.Vsph = self.buildVsph()
            # xp.savez("test_Vsph",Vsph=self.Vsph,Vgrid=self.Vgrid, Rgrid=self.R, rgrid=self.r, g_grid=self.g, psi_grid=self.psi)

        self.VOm = self.buildVOm()

        self.shape = self.Vgrid.shape + (self.NOm,)
        self.size = xp.prod(xp.asarray(self.shape))

        dR = self.R[1] - self.R[0]
        dr = self.r[1] - self.r[0]
        dg = self.g[1] - self.g[0]

        self.P  = xp.fft.fftshift(xp.fft.fftfreq(args.NR, dR)) * 2 * xp.pi
        self.p  = xp.fft.fftshift(xp.fft.fftfreq(args.Nr, dr)) * 2 * xp.pi
        self.pg = xp.fft.fftshift(xp.fft.fftfreq(args.Ng, dg)) * 2 * xp.pi

        # FIXME: the representations of the operators we build are
        # 'dumb' in the sense that they do not know how to apply
        # themselves to vectors in our |Rrɣ> space. Rather, that logic
        # is encoded in Hx() and duplicated wherever needed. It would
        # be nicer if we could encode it in the operators themselves.
        # Then we could do something like self.ddR2 @ x and get the
        # correct behavior for free. We also wouldn't have to
        # duplicate it in H.build_diag() jupyter notebooks. Fixing
        # this would also let us make the Hamiltonian class more
        # generic: simply defining the axes and the operators.

        # N.B.: These all lack the factor of -1/(2 * mu)
        # We also are throwing away the returned jacobian of R/r
        #self.ddR2, _ = KE_Borisov(self.R, bare=True)
        
        # needed for testing on tiny systems in 3D
        stencil_R = min(11, args.NR)
        if stencil_R%2==0: stencil_R -= 1
        stencil_g = min(11,args.Ng)
        if stencil_g%2==0: stencil_g -= 1

        self.ddR2    = KE(args.NR, dR, bare=True, cyclic=False, stencil_size = stencil_R) 
        self.ddr2, _ = KE_Borisov(self.r, bare=True)

        self.ddr_lab2, _ = KE_Borisov(self.r_lab, bare=True)
        self.ddR_lab2    = KE(args.NR, self.R_lab[1]-self.R_lab[0], bare=True, cyclic=False, stencil_size=stencil_R)


        # Part of the reason for using a cyclic *stencil* for gamma
        # rather than KE_FFT is that it wasn't immediately obvious how
        # I would represent ∂/∂γ. (∂²/∂γ² was clear.)  N.B.: The
        # default stencil degree is 11
        self.ddg2 = KE(args.Ng, dg, bare=True, cyclic=True, stencil_size=stencil_g)
        self.ddg1 = KE(args.Ng, dg, bare=True, cyclic=True, order=1, stencil_size= stencil_g)

        # since we need these in Hx; maybe fine to compute on the fly?
        self.Rinv2 = 1.0/(self.R_grid)**2
        self.rinv2 = 1.0/(self.r_grid)**2

        self.diag = self.buildDiag()

        xa = xp.zeros(self.size,dtype=xp.complex128)
        xa[0] = 1
        xa = self.Hx(xa)

        if not hasattr(args, "preconditioner"):
            args.preconditioner = 'naive'

        self.args = args

        builder, self.preconditioner, self.make_guess = {
#            'BO':     (self._build_preconditioner_BO,        self._preconditioner_BO,        self._make_guess_BO),
            'naive':  (lambda: (self.diag,),                 self._preconditioner_naive,     self._make_guess_naive),
            None:     (lambda: (self.diag,),                 self._preconditioner_naive,     self._make_guess_naive),
            }[args.preconditioner]

        with timer_ctx(f"Build preconditioner {args.preconditioner}"):
            self._preconditioner_data = builder()
            size = sum([x.nbytes for x in self._preconditioner_data]) / 1024**2
            print(f"Preconditioner requires {int(size)}MB.")


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

    def V(self, R, r, gamma):
        mu12 = self.mu12
        aa = self.aa
        M_1 = self.M_1
        M_2 = self.M_2

        kappa2 = r*R*xp.cos(gamma)

        r1e2 = (aa*r)**2 + (R/aa)**2*(mu12/M_1)**2 - 2*kappa2*mu12/M_1
        r2e2 = (aa*r)**2 + (R/aa)**2*(mu12/M_2)**2 + 2*kappa2*mu12/M_2

        r1e = xp.sqrt(xp.where(r1e2 < 0, 0, r1e2))
        r2e = xp.sqrt(xp.where(r2e2 < 0, 0, r2e2))

        return self._Vfunc(R/aa, r1e, r2e, (self.g_1, self.g_2))

    def sph_transform(self, Vgrid, j1, j2, Om1, Om2):
        ''' returns (int dγ dψ sin(γ)
                        conj(Y1(j1,Ω1,γ,ψ)) V(r,R,γ, ψ=0) Y2(j2,Ω2, γ,ψ) )'''
        args = self.args
        Y1 = xp.zeros((args.Ng), dtype=xp.float64)
        Y2 = xp.zeros((args.Ng), dtype=xp.float64)
        V_jjOmOm = xp.zeros((args.NR, args.Nr),dtype=xp.float64)

        dg = self.g[1]-self.g[0]
        
        if Om1==0 and j1==0:
            Y1[:] = 1/2/xp.pi**0.5
        elif j1 < xp.abs(Om1):
            Y1[:] = 0 # exclude j < |Om|
        else:
            c1 = ((2*xp.abs(Om1)+1)/4/xp.pi*factorial(j2-xp.abs(Om1))/factorial(xp.abs(Om1)+j1))**0.5
            if Om1 > 0: c1 *= (-1)**Om1
            Y1 = c1*lpmv(xp.abs(Om1),j1, xp.cos(self.g))
        
        if Om2==0 and j2==0:
            Y2[:] = 1/2/xp.pi**0.5
        elif j2 < xp.abs(Om2):
            Y2[:] = 0 # exclude j < |Om|
        else:
            c2 = ((2*xp.abs(Om2)+1)/4/xp.pi*factorial(j2-xp.abs(Om2))/factorial(xp.abs(Om2)+j2))**0.5
            if Om1 > 0: c2 *= (-1)**Om2
            Y2 = c2*lpmv(xp.abs(Om2),j2, xp.cos(self.g))

        
        if xp.any(xp.isnan(Y1)):
            print("Y1 error", j1, Om1 )
        if xp.any(xp.isnan(Y2)):
            print("Y2 error", j2, Om2 )
        # for igam in range(args.Ng):
            # for ipsi in range(2*self.J+1)
            # Y1[igam] = c1*lpmv[j1,Om1,self.g[igam]]
            # Y2[igam] = c2*lpmv[j2,Om2,self.g[igam]]
                # Y1[igam, ipsi] = sph_harm_y(j1,Om1,self.g[igam], self.psi[ipsi])
                # Y2[igam, ipsi] = sph_harm_y(j2,Om2, self.g[igam], self.psi[ipsi])
        sin_gam = xp.sin(self.g)

        for iR in range(args.NR):
            for ir in range(args.Nr):
                integrand = (xp.conj(Y1)*Y2)*sin_gam[:]*Vgrid[iR,ir,:]
                V_jjOmOm[iR,ir] = xp.sum(integrand)/4.*dg # 2 for psi being 0...2pi instead of 0...pi
        return V_jjOmOm

    def buildVsph(self):
        ''' V(R,r,j,j',Ω=Ω') '''
        args = self.args
        Vsph = xp.zeros((args.NR, args.Nr, args.Ng, args.Ng, 2*self.J+1), dtype=xp.float64)
        for iOm in range(2*self.J+1):
                for ij1 in range(args.Ng):
                    for ij2 in range(args.Ng):
                        a = self.sph_transform(self.Vgrid, self.j[ij1], self.j[ij2], self.Om[iOm], self.Om[iOm])
                        Vsph[:,:,ij1,ij2,iOm] = a 
        
        assert not xp.any(xp.isnan(Vsph))
        return Vsph

    def buildVOm(self):
        ''' Contains the Clebsch-Gordon Coeffs between Om values
                √(J(J+1)-Ω(Ω±1))√(j(j+1)-Ω(Ω±1)),
                shape: Ng x (2J+1)^2 '''
        args = self.args
        VOm = xp.zeros((args.Ng, 2*self.J+1, 2*self.J+1), dtype=xp.float64)

        #NB: recall self.Om = [-J, -J+1 ...0...J-1,J]
        # will not appear tridiagonal with this matrix element ordering!
        for i in range(2*self.J+1):
            for j in range(2*self.J+1):
                if self.Om[i]== self.Om[j]+1:
                    a = self.j*(self.j+1) - self.Om[i]*(self.Om[i]+1)
                    VOm[:,i,j] = xp.sqrt(xp.maximum(xp.zeros(args.Ng),a))
                    VOm[:,i,j] *= xp.sqrt(self.J*(self.J+1) - self.Om[i]*(self.Om[i]+1))
                elif self.Om[i] == self.Om[j]-1:
                    a = self.j*(self.j+1) - self.Om[i]*(self.Om[i]-1)
                    VOm[:,i,j] = xp.sqrt(xp.maximum(xp.zeros(args.Ng),a))
                    VOm[:,i,j] *= xp.sqrt(self.J*(self.J+1) - self.Om[i]*(self.Om[i]-1))

        assert not xp.any(xp.isnan(VOm))
        return VOm

    # allows H @ x
    def __matmul__(self, other):
        return self.Hx(other).reshape(other.shape)

    #@partial(jax.jit, static_argnums=0)
    def Hx(self, x):
        return self.Tx(x) + self.Vx(x)

    def Vx(self,x):
        if xp.backend == 'torch':
            xa = x.reshape((-1,) + self.shape).type(self.dtype)
        else:
            xa = x.reshape((-1,) + self.shape).astype(self.dtype)
        vout = xp.einsum('BRrjO, RrjkO-> BRrkO', xa, self.Vsph)
        return vout.reshape(x.shape)


    #@partial(jax.jit, static_argnums=0)
    def Tx(self, x):
        if xp.backend == 'torch':
            xa = x.reshape((-1,) + self.shape).type(self.dtype)
        else:
            xa = x.reshape((-1,) + self.shape).astype(self.dtype)
        ke = xp.zeros_like(xa)

        # Radial Kinetic Energy terms, easy
        ke += xp.einsum('BRrjO,RS->BSrjO', xa, self.ddR2)  # ∂²/∂R²
        ke += xp.einsum('BRrjO,rs->BRsjO', xa, self.ddr2)  # ∂²/∂r²

        # Angular electronic ke terms: j(j+1)/r^2 + j(j+1)/R^2
        kej = xp.einsum('BRrjO, j-> BRrjO', xa, self.j*(self.j+1))
        # kej = xa*self.j_grid*(self.j_grid+1) # we don't have a j_grid defined yet?
        ke += (self.Rinv2 + self.rinv2)*kej


        # Angular Kinetic Energy J terms
        if self.J != 0:
            keJdiag  = xa*self.J*(self.J+1)
            keJdiag += -2*xp.einsum('BRrjO,O-> BRrjO', xa,self.Om**2)     #  J(J+1)-2Ω^2

            keJoffdiag = xp.einsum('BRrjO,jOP-> BRrjP', xa, self.VOm) #√(J(J+1)-Ω(Ω±1))√(j(j+1)-Ω(Ω±1))
            ke += self.Rinv2*keJdiag
            ke += self.Rinv2*keJoffdiag

        # mass portion of KE
        ke *= -1 / (2*self.mu)
        return ke.reshape(x.shape)


    # N.B. This section *must* be kept in sync with Hx above
    def buildDiag(self):
        ke  = xp.zeros(self.shape)
        ke += xp.diag(self.ddR2)[:, None, None, None]
        ke += xp.diag(self.ddr2)[None, :, None, None]
        ke += (self.Rinv2 + self.rinv2) * (self.j*(self.j+1))[None, None, :,None]

        # Angular Kinetic Energy J terms
        if self.J != 0:
            ke += self.Rinv2 * (
                self.J*(self.J+1) -2*self.Om**2)[None,None,None,:]

        # mass portion of KE
        ke *= -1 / (2*self.mu)

        Vdiag = xp.einsum('RrjjO-> RrjO',self.Vsph)
        # Potential terms
        diag = Vdiag + ke
        
        assert not xp.any(xp.isnan(diag))
        return diag.ravel()

    # FIXME: See concerns about jit-ing Hx. Currently jitting in the
    # @partial(jax.jit, static_argnums=0) fashion will break; not sure why.

    def _make_guess_naive(self, min_guess):
        tr_Vsph = xp.einsum('RrjjO->RrjO',self.Vsph)
        guesses = xp.exp(-(tr_Vsph - xp.min(tr_Vsph))**2/27.211**2).ravel()
        return guesses

    #@partial(jax.jit, static_argnums=0)
    def _preconditioner_naive(self, dx, e, x0):
        diagd = self.diag - (e - 1e-5)
        return dx/diagd

    #@partial(jax.jit, static_argnums=0)
    def _preconditioner_power(self, dx, e, x0):
        #vinv = 1/(self.Vgrid.ravel() - (e - 1e-5))
        vinv = 1/(self.diag - (e - 1e-5))
        vr = vinv * dx
        tvr = self.Tx(vr) - (self.diag - self.Vgrid.ravel()) * vr
        return vr - vinv * tvr

    def BO_spectrum(self, nroots=0, Hel_func=None):
        print("Building BO spectrum")
        NR, Nr, Ng, NOm = self.shape
        Nelec = Nr*Ng*NOm

        if Hel_func is None:
            Hel_func = self.build_Hel

        mem_thresh = 1e5
        memory_constrained = self.size > mem_thresh

        print(f"memory constraint threshold = {mem_thresh}, {memory_constrained}")

        if xp.backend == 'numpy':
            threadctl = ThreadpoolController()
            with threadctl.limit(limits=1), cf.ThreadPoolExecutor(max_workers=self.max_threads) as ex:
                result = list(tqdm(ex.map(lambda i: (i, xp.linalg.eigvalsh(Hel_func(i))), range(NR)), total=NR))
                Ad_n = xp.zeros((NR, Nelec))
                for i, a in result:
                    Ad_n[i] = a
        elif memory_constrained:
            Ad_n  = xp.zeros((NR, Nelec))
            for i in tqdm(range(NR)):
                Ad_n[i] = xp.linalg.eigvalsh(Hel_func(i))
        else:
            Ad_n = xp.linalg.eigvalsh(Hel_func())

        Hbo = xp.empty((Nelec, NR, NR))                # Hbo = -1/2/μ(∂²/∂R² + 1/4/R²) + V_n
        Hbo[:] = -1 / 2 / self.mu * self.ddR2          #       -1/2/μ(∂²/∂R² + 1/4/R²)
        Hbo[:, xp.arange(NR), xp.arange(NR)] += Ad_n.T # V_n

        Ad_vn = xp.linalg.eigvalsh(Hbo)  # xp.linalg.eigh(Hbo)
        Ad_vn = Ad_vn.T

        for i in range(nroots):
            with xp.printoptions(linewidth=xp.inf):
                print(f"BO state {i} spectrum:", Ad_vn[:nroots,i])
        return (Ad_vn, Ad_n)  # energies are Ad_vn[v,n]


    # NR x (NrNgNOm) x (NrNgNOm)
    def build_Hel(self, Ridx=None):
        NR, Nr, Nj, NOm = self.shape
        Nsph = Nj * NOm
        Nelec = Nr * Nsph

        if Ridx is None:
            Ridx = xp.arange(NR)
        else:
            Ridx = xp.atleast_1d(Ridx)
            NR,  = Ridx.shape

        def kron3(Or, Oj, OO):
            return xp.kron(Or, xp.kron(Oj, OO))

        # Hel = -1/2/μ · (Te + VOm) + V
        # Te  =  ∂²/∂r² + (1/r²)j(j+1) + (1/R²)(j(j+1) + J(J+1) - 2Ω²)
        # VOm = (1/R²)√(J(J+1) - Ω(Ω ± 1))√(j(j+1) - Ω(Ω ± 1)) ; (1/R²)*self.VOm
        # N.B. self.ddr2 = ∂²/∂r² + 1/4/r²
        Hel = xp.empty((NR, Nelec, Nelec), dtype=self.dtype)

        # build *bare* Te first
        # R-independent terms: ∂²/∂r² + (1/r²)j(j+1)
        Hel[:] = (
            xp.kron(self.ddr2, xp.eye(Nsph)) +   # ∂²/∂r²
            kron3(xp.diag(1 / self.r**2),        # +(1/r²)j(j+1)
                  xp.diag(self.j*(self.j+1)),
                  xp.eye(NOm))
        )


        # R-dependent terms: (1/R²)j(j+1)
        Rinv2 = (1 / self.R**2)[Ridx, None, None]  # (1/R²), ready for broadcasting
        Hel += Rinv2 * kron3(xp.eye(Nr),           # (1/R²) * j(j+1)
                             xp.diag(self.j*(self.j+1)),
                             xp.eye(NOm))[None]

        # J terms: (1/R²)(J(J+1) - 2Ω²)
        if self.J != 0:
            Hel[:, xp.arange(Nelec), xp.arange(Nelec)] += (
                Rinv2 * self.J * (self.J+1)  # +(1/R²) (J(J+1)
            )
            Hel -= 2 * kron3(xp.eye(Nr), xp.eye(Nj), self.Om**2) * Rinv2 # - 2Ω²(1/R²)

        # VOm term:
        VOm_big =  xp.einsum("jkOP,jOP->jkOP",
                             xp.kron(xp.eye(Nj), xp.eye(NOm)),
                             self.VOm).rehape(Nsph, Nsph)# Nj(NOm)^2 -> Nsph^2

        Hel += xp.kron(xp.eye(Nr), VOm_big) * R2inv
        Hel *= -1 / (2 * self.mu)  # -1/2/μ · (Te + VOm)


        Hel[:, xp.arange(Nelec), xp.arange(Nelec)] += xp.einsum("rs,OP,RrjkO->RrsjkOP",
                                                                xp.eye(Nr), xp.eye(NOm),
                                                                self.Vsph[Ridx])

        return xp.squeeze(Hel)

    # NR x (NrNj) x (NrNj)
    def build_Hel_j(self, Ridx=None, Nj=None):
        NR, Nr, Ng = self.shape

        if Nj is None:
            Nj = len(self.j)
            j = self.j
        elif Nj > len(self.j):
            raise RuntimeError("Cannot use more frequencies than Ng")
        else:
            j = xp.arange(Nj*2)-Nj

        Nelec = Nr*Nj

        if Ridx is None:
            Ridx = xp.arange(NR)
        else:
            Ridx = xp.atleast_1d(Ridx)
            NR,  = Ridx.shape

        # Hel = -1/2/μ · Tej + Vjj
        # Tej =  ∂²/∂r² + 1/4/r² - j²/r² - (J - j)²/R²
        # Vjj = <j|V(R,r)|j'> = 1/2/π ∫ dγ cos(|j-j'|γ) V(R,r,γ)

        # N.B. Tej is diagonal in j
        # N.B. Tn = ∂²/∂R² + 1/4/R²
        # N.B. self.ddr2 = ∂²/∂r² + 1/4/r²
        Hel = xp.empty((NR, Nelec, Nelec), dtype=self.dtype)

        # build *bare* Te first (no masses, hbar)
        # R-independent terms: ∂²/∂r² + 1/4/r² - j²/r²
        Hel[:] = (
            xp.kron(self.ddr2, xp.eye(Nj)) -                     # ∂²/∂r² + 1/4/r²
            xp.kron(xp.diag(1 / self.r**2), xp.diag(j**2))  # -j²/r²
        )

        # R-dependent terms: -(J - j)²/R²
        Rinv2 = (1 / self.R**2)[Ridx, None, None]  # (1/R²), ready for broadcasting
        Hel -= Rinv2 * xp.kron(
            xp.eye(Nr), xp.diag((self.J-j)**2))[None]  # -(J - j)²/R²

        Hel *= -1 / (2 * self.mu)  # -1/2/μ · Te

        # Vjj = <j|V(R,r)|j'> = 1/2/π ∫ dγ cos(|j-j'|γ) V(R,r,γ)
        COS = xp.cos(xp.abs(j[:, None] - j)[..., None] * self.g)  # cos(|j-j'|γ); shape: Nj × Nj × Ng
        # Vjj of shape: NR × Nr × Nj × Nj
        dg = self.g[1] - self.g[0]
        Vjj = xp.sum(
            COS[None,None,...] * self.Vgrid[Ridx,:,None,None,:] * dg,
            axis=-1)/numpy.pi/2  # 1/2/π Σ dγ cos(|j-j'|γ) V(R,r,γ)
        H_idx =  xp.arange(Nr)[:, None]*Nj + xp.arange(Nj)
        Hel[:, H_idx[:,:, None], H_idx[:, None, :]] += Vjj  # + Vjj

        ## Alternative, method for building Vjj via FFt and rolling;
        ## the above was generally faster sometimes considerably
        # Vj = xp.fft.ifft(self.Vgrid[Ridx], axis=2).real
        # H_idx =  xp.arange(Nr)[:, None]*Nj + xp.arange(Nj)
        # roll_idx = (xp.arange(Nj)[:, None] - xp.arange(Nj)[None, :])
        # Hel[:, H_idx[:,:, None], H_idx[:, None, :]] += Vj[:, :, roll_idx] # + Vjj

        return xp.squeeze(Hel)


    def _build_preconditioner_jfull(self):
        print("Building U_n")
        NR, Nr, Ng = self.shape
        Nj = Ng
        Nelec = Nr*Nj

        with timer_ctx("Build Hel"):
            Hel = self.build_Hel_j()

        with timer_ctx(f"Diag  Hel"):
            if xp.backend == 'numpy':
                threadctl = ThreadpoolController()
                with threadctl.limit(limits=1), cf.ThreadPoolExecutor(max_workers=self.max_threads) as ex:
                    result = ex.map(lambda i: (i, xp.linalg.eigh(self.build_Hel(i))), range(NR))
                    U_n   = xp.zeros((NR, Nr*Ng, Nelec), dtype=self.dtype)
                    Ad_n  = xp.zeros((NR, Nelec))
                    for i, (a, u) in result:
                        Ad_n[i] = a
                        U_n[i]  = u
            else:
                Ad_n, U_n = xp.linalg.eigh(Hel)

        Uj = xp.zeros((Ng, Ng), dtype=xp.complex128)
        for jidx, j in enumerate(self.j):
            Uj[jidx] = (2*numpy.pi)**(-1/2)*xp.exp((0+1j)*j*self.g)

        U_n4 = U_n.reshape((NR, Nr, Ng, Nr, Ng))
        U_n = xp.einsum("gj,Rrjsk,kl->Rrgsl", Uj, U_n4, Uj.T).reshape((NR, Nr*Ng, Nelec))

        with timer_ctx("Phase match U_n"):
            phase_match(U_n)

        NR, Nelec, _ = Hel.shape

        with timer_ctx("Build Hbo"):
            Hbo = xp.empty((Nelec, NR, NR))                # Hbo = -1/2/μ(∂²/∂R² + 1/4/R²) + V_n
            Hbo[:] = -1 / 2 / self.mu * self.ddR2          #       -1/2/μ(∂²/∂R² + 1/4/R²)
            Hbo[:, xp.arange(NR), xp.arange(NR)] += Ad_n.T # V_n

        with timer_ctx("Diag  Hbo"):
            Ad_vn, U_v = xp.linalg.eigh(Hbo)  # xp.linalg.eigh(Hbo)
            Ad_vn = Ad_vn.T

        with timer_ctx("Phase match U_v"):
            phase_match(U_v)

        pc = (Ad_vn, U_n, U_v, Ad_n)
        return pc

    def _make_guess_jfull(self, min_guess):
        guesses = xp.exp(-(self.Vgrid - xp.min(self.Vgrid))**2/27.211**2).ravel()
        return guesses


    def _preconditioner_jfull(self, dx, e, x0):
        Ad_vn, U_n, U_v, *_ = self._preconditioner_data
        diagd = Ad_vn - (e - 1e-5)
        NR, Nr, Ng = self.shape

        dx_ = dx.reshape((-1, NR, Nr*Ng))

        kwargs = dict(optimize=True)
        if xp.backend == 'torch':
            kwargs = {}

        tr_ = xp.einsum(
            'Rij,jRq,qj,jmq,mpj,Bmp->BRi',
            U_n, U_v, 1.0 / diagd, U_v, U_n, dx_, **kwargs
        ).real

        return tr_.reshape(dx.shape)


    # def _build_preconditioner_BO(self):
    #     print("Building U_n")
    #     NR, Nr, Ng = self.shape
    #     Nelec = Nr*Ng
    #
    #     # if xp.backend == 'cupy' or xp.backend == 'cupynumeric':
    #     #     from cupyx.profiler import time_range as timer_ctx
    #     # else:
    #     #     from debug import timer_ctx
    #
    #     with timer_ctx("Build Hel"):
    #         Hel = self.build_Hel()
    #
    #     #FIXME: something like this enhanced preconditioning in some cases, maybe?
    #     #Hel[:] += -xp.kron(xp.diag(1 / self.r**2), xp.eye(Ng)/4)/2/self.mu
    #     # with timer_ctx("Diag  Hel"):
    #     #     from cupyx.profiler import benchmark
    #     #     from cupy.cuda import memory_hooks
    #     #     # Profile the specific batch
    #     #     with memory_hooks.DebugPrintHook():
    #     #         result = benchmark(xp.linalg.eigh, (Hel,), n_repeat=5)
    #     #         print(result)
    #     #     Ad_n, U_n = xp.linalg.eigh(Hel)
    #
    #     with timer_ctx(f"Diag  Hel"):
    #         if xp.backend == 'numpy':
    #             threadctl = ThreadpoolController()
    #             with threadctl.limit(limits=1), cf.ThreadPoolExecutor(max_workers=self.max_threads) as ex:
    #                 result = ex.map(lambda i: (i, xp.linalg.eigh(self.build_Hel(i))), range(NR))
    #                 U_n   = xp.zeros((NR, Nr*Ng, Nelec), dtype=self.dtype)
    #                 Ad_n  = xp.zeros((NR, Nelec))
    #                 for i, (a, u) in result:
    #                     Ad_n[i] = a
    #                     U_n[i]  = u
    #         else:
    #             Ad_n, U_n = xp.linalg.eigh(Hel)
    #
    #     with timer_ctx("Phase match U_n"):
    #         phase_match(U_n)
    #
    #     NR, Nelec, _ = Hel.shape
    #
    #     with timer_ctx("Build Hbo"):
    #         Hbo = xp.empty((Nelec, NR, NR))                # Hbo = -1/2/μ(∂²/∂R² + 1/4/R²) + V_n
    #         Hbo[:] = -1 / 2 / self.mu * self.ddR2          #       -1/2/μ(∂²/∂R² + 1/4/R²)
    #         Hbo[:, xp.arange(NR), xp.arange(NR)] += Ad_n.T # V_n
    #
    #     with timer_ctx("Diag  Hbo"):
    #         Ad_vn, U_v = xp.linalg.eigh(Hbo)  # xp.linalg.eigh(Hbo)
    #         Ad_vn = Ad_vn.T
    #
    #     with timer_ctx("Phase match U_v"):
    #         phase_match(U_v)
    #
    #     pc = (Ad_vn, U_n, U_v, Ad_n)
    #     return pc
    #
    # def _make_guess_BO(self, min_guess):
    #     Ad_vn, U_n, U_v, *_ = self._preconditioner_data
    #     # BO states are like: U_n[:,:,n]
    #     # vib states are like: U_v[n,:,v]
    #     s = int(numpy.ceil(numpy.sqrt(min_guess)))
    #
    #     guesses = xp.stack([
    #         (U_n[:,:,n] * U_v[n,:,v,xp.newaxis]).ravel()
    #         for n in range(s) for v in range(s)
    #     ])
    #
    #     return guesses
    #
    # #@partial(jax.jit, static_argnums=0)
    # def _preconditioner_BO(self, dx, e, _):
    #     Ad_vn, U_n, U_v, *_ = self._preconditioner_data
    #     diagd = Ad_vn - (e - 1e-5)
    #     NR, Nr, Ng = self.shape
    #     dx_ = dx.reshape((-1, NR, Nr*Ng))
    #
    #     # YOLO: truncate it
    #     # Nelec=(3*Nr*Ng)//4
    #
    #     # Ad_vn_t = Ad_vn[:, :Nelec]
    #     # diagd_t = Ad_vn_t - (e - 1e-5)
    #     # U_n_t = U_n[:,:, :Nelec]
    #     # U_v_t = U_v[:Nelec, :, :]
    #     # tr_ = jnp.einsum(
    #     #     'Rij,jRq,qj,jmq,mpj,Bmp->BRi',
    #     #     U_n_t, U_v_t, 1.0 / diagd_t, U_v_t, U_n_t, dx_, optimize=True
    #     # )
    #
    #     #FIXME: precompute optimal einsum path and provide that
    #     kwargs = dict(optimize=True)
    #     if xp.backend == 'torch':
    #         kwargs = {}
    #
    #     tr_ = xp.einsum(
    #         'Rij,jRq,qj,jmq,mpj,Bmp->BRi',
    #         U_n, U_v, 1.0 / diagd, U_v, U_n, dx_, **kwargs
    #     )
    #
    #     return tr_.reshape(dx.shape)
    #
    # def _preconditioner_BO_interp(self, dx, e, x0):
    #     return
    #
    # def _build_preconditioner_BO_interp(self):
    #     args = self.args
    #     args.NR //= 2
    #     args.Nr //= 2
    #     args.Ng //= 2
    #     args.preconditioner='BO'
    #     print(args)
    #     H = Hamiltonian(args)
    #     Ad_vn, U_n, U_v, *_ = H._preconditioner_data
    #     print(Ad_vn.shape, U_n.shape, U_v.shape)
    #
    #     exit()
    #     # pseudo-code...
    #     # H_coarse =
    #
    #     return
    #
    # def _make_guess_BO_interp(self, min_guess):
    #     return
    #
    # def _build_preconditioner_V1(self, min_guess=4):
    #     NR, Nr, Ng = self.shape
    #     dg = self.g[1] - self.g[0]
    #
    #     threadctl = ThreadpoolController()
    #     threadctl.limit(limits=1)
    #
    #     j_full = fftshift(xp.arange(Ng)-Ng//2)
    #     COS = xp.cos(2*j_full[:,None] * self.g)
    #     V1 = simpson(self.Vgrid[:,:,:], dx=dg, axis=-1)/xp.pi/2
    #
    #     # V1 is V2(j==0)
    #     # IF debugging
    #     #V2 = simpson(self.Vgrid[:,:,None,:] * COS[None,None,:,:], dx=dg, axis=-1)/xp.pi/2
    #     #assert(xp.allclose(V1, V2[:,:,xp.squeeze(xp.where(j_full == 0)).item()]))
    #
    #     Ad = xp.zeros((NR, Nr, Ng))
    #     U  = xp.zeros((NR, Nr, Ng, Nr))
    #
    #     def diag_H0(args, Ad=Ad, U=U):
    #         Ri, (ji, j) = args
    #         H_el = -(
    #             self.ddr2 + xp.diag(j**2/self.r**2 + (self.J-j)**2/self.R[Ri]**2)
    #         )/2/self.mu + xp.diag(V1[Ri])
    #         Ad[Ri, :, ji], U[Ri, :, ji, :] = xp.linalg.eigh(H_el)
    #
    #
    #     # Presumably because of gated access, this tops out pretty fast at ~
    #     with cf.ThreadPoolExecutor(max_workers=self.max_threads) as ex:
    #         list(tqdm(
    #             ex.map(diag_H0, product(range(NR),enumerate(j_full))),
    #             total=NR*Ng, desc="Building Preconditioner", delay=3))
    #
    #     # Align phases by iterating over Nr eigen vectors at each Ri, ji
    #     for Ri, ji in tqdm(product(range(NR),
    #                                range(Ng)),
    #                        total=NR*Ng,
    #                        desc="Phase Matching",
    #                        delay=3, # only display if longer than 3 seconds
    #                        ):
    #         for n in range(Nr):
    #             current = (Ri, slice(None), ji, n)
    #             if Ri == 0 and ji == 0:   # consistent, arbitrary reference phase
    #                 reference = (0, 0, 0, 0)
    #             elif ji > 0:              # line up with previous j
    #                 reference = (Ri, slice(None), ji-1, n)
    #             elif ji == 0 and Ri > 0:  # line up with previous R
    #                 reference = (Ri-1, slice(None), ji, n)
    #             else:
    #                 raise RuntimeError("We should always have a reference!")
    #
    #             # be careful not to undo your work!! (Ri=0)-1=-1
    #             # indexes the last one and breaks things!
    #
    #             # if any(map(lambda x: x < 0 if type(x) is not slice else False,
    #             #             current + reference
    #             #             )):
    #             #     raise RuntimeError(f"invalid reference: {current(0)}, {reference(0)}")
    #
    #             # actually match the phase
    #             if xp.sum(U[current] * U[reference]) < 0:
    #                 U[current] *= -1
    #
    #     # yolo
    #     #oddjs = j_full%2 == 1
    #     #U[:, :, oddjs, :] = 0
    #     #U = fft(U, axis=2)
    #     #assert(xp.mean(xp.abs(U.imag)) < 1e-12)
    #     #U = U.real
    #
    #     U = xp.fft.ifft(U, axis=2)
    #     assert(xp.mean(xp.abs(U.imag)) < 1e-12)
    #     U = U.real
    #
    #     return (Ad, U)
    #
    # def _make_guess_V1(self, min_guess):
    #     Ad, U, *_ = self._preconditioner_data
    #     NR, Nr, Ng = self.shape
    #     # States are U[R, :, Ng//2 + j, n]
    #     s = int(numpy.ceil(numpy.sqrt(min_guess)))
    #     guesses = xp.stack([
    #         xp.copy(xp.broadcast_to(
    #             U[:, :, Ng//2 + j, i][:, :, xp.newaxis],
    #             self.shape
    #         )).ravel() for i in range(s) for j in range(s)])
    #
    #     return guesses
    #
    # #@partial(jax.jit, static_argnums=0)
    # def _preconditioner_V1(self, dx, e, x0):
    #     dx_ = dx.reshape((-1,) + self.shape)
    #     Ad, U, *_ = self._preconditioner_data
    #     diagd = Ad - (e - 1e-5)
    #
    #     dx_t = xp.einsum("Rrgi,BRrg->BRig", U, dx_)#, optimize=True)
    #     tr_t = dx_t / diagd
    #     tr_ = xp.einsum('Rigr,BRig->BRrg', U, tr_t)#, optimize=True)
    #
    #     return tr_.reshape(dx.shape)

    # Below here are a bunch of things related to immutability
    # https://docs.jax.dev/en/latest/faq.html#how-to-use-jit-with-methods
    def __hash__(self):
        if not getattr(self, '_locked', False):
            raise RuntimeError("Hash called before init")
        return self._hash

    def __eq__(self, other):
        if not getattr(self, '_locked', False):
            raise RuntimeError("Eq called before init")
        if not isinstance(other, Hamiltonian):
            return False
        try:
            return all(getattr(self, key) == getattr(other, key) for key in self.__slots__)
        except AttributeError:
            return False

    # prevent data from being modified
    def __setattr__(self, key, value):
        if getattr(self, '_locked', False):
            raise AttributeError(f"Cannot modify '{key}'; all members are frozen on creation")
        super().__setattr__(key, value)

    # Allow pickleing
    def __getstate__(self):
        return {slot: getattr(self, slot) for slot in self.__slots__}

    # Go around the locks at unpickle time
    def __setstate__(self, state):
        for key, value in state.items():
            object.__setattr__(self, key, value)



def parse_args():
    parser = ap.ArgumentParser(
        prog='3body-2D',
        description="computes the lowest k eigenvalues of a 3-body potential in 2D")

    class ArrayAction(ap.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            setattr(namespace, self.dest, xp.array(values, dtype=float))

    parser.add_argument('-k', metavar='num_eigenvalues', default=5, type=int)
    parser.add_argument('-t', metavar="num_threads", default=16, type=int)
    parser.add_argument('-g_1', metavar='g_1', required=True, type=float)
    parser.add_argument('-g_2', metavar='g_2', required=True, type=float)
    parser.add_argument('-M_1', required=True, type=float)
    parser.add_argument('-M_2', required=True, type=float)
    parser.add_argument('-J', default=0, type=int)
    parser.add_argument('-R', dest="NR", metavar="NR", default=80, type=int)
    parser.add_argument('-r', dest="Nr", metavar="Nr", default=80, type=int)
    parser.add_argument('-g', dest="Ng", metavar="Ng", default=80, type=int)
    parser.add_argument('--potential', choices=['soft_coulomb', 'borgis'],
                        default='soft_coulomb')
    parser.add_argument('--extent', metavar="X", action=ArrayAction,
                        nargs=3, help="Rmin Rmax rmax, in Bohr "
                        "(typically set automatically)")
    parser.add_argument('--exact_diagonalization', action='store_true')
    parser.add_argument('--bo_spectrum', metavar='spec.npz', type=Path, default=None)
    parser.add_argument('--preconditioner', choices=['naive', 'V1', 'BO', 'BO-int', 'jfull'],
                        default="naive", type=str)
    parser.add_argument('--verbosity', default=2, type=int)
    parser.add_argument('--backend', default='numpy')
    parser.add_argument('--iterations', metavar='max_iterations', default=10000, type=int)
    parser.add_argument('--subspace', metavar='max_subspace', default=1000, type=int)
    parser.add_argument('--guess', metavar="guess.npz", type=Path, default=None)
    parser.add_argument('--evecs', metavar="guess.npz", type=Path, default=None)
    parser.add_argument('--save', metavar="filename")

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print(args)

    # you can only select the backend once and it must be before you use any xp functions
    if xp.backend != args.backend:
        xp.backend = args.backend

    threadctl = ThreadpoolController()
    threadctl.limit(limits=args.t)

    with timer_ctx("Build H"):
        H = Hamiltonian(args)

    with timer_ctx("Load/make guesses"):
        guess = get_interpolated_guess(args.guess, (H.R, H.r, H.g))
        if guess is None:
            guess = H.make_guess(args.k)

    if args.bo_spectrum:
        with timer_ctx("BO spectrum"):
            Ad_vn, Ad_n = H.BO_spectrum(args.k)
            numpy.savez_compressed(args.bo_spectrum, bo_spectrum=Ad_vn, bo_surfaces=Ad_n)

    # FIXME: would like to use a callback to save intermediate
    # wavefunctions in case we need to do a restart.
    with timer_ctx(f"Davidson of size {H.size}"):
        conv, e_approx, evecs = lib.davidson1(
            H.Hx,
            guess,
            #H.diag,
            H.preconditioner,
            nroots=args.k,
            max_cycle=args.iterations,
            verbose=args.verbosity,
            max_space=args.subspace,
            max_memory=get_davidson_mem(0.75),
            #tol=1e-12, #FIXME:DEBUG
            tol=1e-10,
        )

    #guess quality
    #for i, (e,g) in enumerate(zip(evecs, guess)):
    #    print(i, xp.abs(xp.vdot(e, g))**2 / (xp.vdot(e, e) * xp.vdot(g, g)))

    print("Davidson:", e_approx)
    print(conv)

    if args.evecs:
        numpy.savez_compressed(args.evecs, guess=evecs, H=H)
        print("Wrote eigenvectors to", args.evecs)

    if args.bo_spectrum:
        e_bo = xp.sort(Ad_vn.flatten())
        bo = e_bo[1] - e_bo[0]
        print("BO gap", bo)
        if all(conv):
            ex = e_approx[1] - e_approx[0]
            print("exact, bo, error:", ex, bo, (bo-ex)/ex)
    elif all(conv):
        ex = e_approx[1] - e_approx[0]
        print("exact gap", ex)

    if args.save is not None:
        if all(conv):
            with open(args.save, "a") as f:
                print(args.M_1, args.M_2, args.g_1, args.g_2, args.J,
                      " ".join(map(str, e_approx)), file=f)
            print(f"Computed fixed center-of-mass eigenvalues",
                  f"for M_1={args.M_1}, M_2={args.M_2} amu",
                  f"with charges g_1={args.g_1}, g_2={args.g_2}",
                  f"and total J={args.J}",
                  f"and appended to {args.save}")
        else:
            print("Skipping saving unconverged results.")

    if args.exact_diagonalization:
        e_exact = solve_exact_gen(H.Hx, H.size, num_state=args.k)
        print("Exact:", e_exact)
        prms(e_approx, e_exact, "RMS deviation between Davidson and Exact")

    if not all(conv):
        print("WARNING: Not all eigenvalues converged")
        exit(1)
    else:
        print("All eigenvalues converged")
