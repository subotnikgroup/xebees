#!/usr/bin/env python
from scipy.special import lpmv

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
from hamiltonian import  KE, KE_Borisov_3D
from davidson import phase_match, get_interpolated_guess, get_davidson_mem, solve_exact_gen

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
        'R', 'r', 'g', 'j', 'Om',
        'axes', 'dtype', 'args',
        'max_threads',
        'preconditioner', 'make_guess', '_Vfunc',
        'Vgrid', 'Vint', 'Pjk', 'VOm', 'ddR2', 'ddr2',
        'Rinv2', 'rinv2', 'diag', '_preconditioner_data',
        'shape', 'size',
        '_locked', '_hash', 'r_lab', 'R_lab', 'ddr_lab2', 'ddR_lab2'
    )

    def __init__(self, args):
        # save number of threads for preconditioner
        self.max_threads = getattr(args, "t", 1) # default to single-threaded
        self.args = args

        self.m_e = 1
        self.M_1 = args.M_1
        self.M_2 = args.M_2

        self.g_1 = args.g_1
        self.g_2 = args.g_2

        self.J   = args.J
        self.dtype = xp.float64

        # Potential function selection
        if not hasattr(args, "potential"):
            args.potential = 'borgis'

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
            'borgis': (partial(potentials.borgis, asymmetry_param=1), potentials.extents_borgis),
            'erf_coulomb':(potentials.erf_coulomb, potentials.extents_erf_coulomb)
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
        # starting 1 "step" away from 0. This behavior is required
        # because we have terms that go like 1/r.
        self.r     = xp.linspace(r_max    /args.Nr, r_max, args.Nr)
        self.r_lab = xp.linspace(r_max_lab/args.Nr, r_max_lab, args.Nr)
        self.R     = xp.linspace(*R_range,     args.NR)
        self.R_lab = xp.linspace(*R_range_lab, args.NR)

        # require Ng to be even
        if args.Ng % 2 != 0:
            raise RuntimeError(f"Ng must be even!")

        # N.B.: We don't have consistent meaning for gamma in the
        # phase-space and exact codes. In the present (exact) case,
        # following Schatz and everyone else (the physicist's
        # notation), ɣ \on [0, π]. See the Potential section of our
        # overleaf for more details. Note also that if we erroneously
        # included the full interval, the potential goes to 0 because
        # the integral transform from the diagonal ɣ basis to the
        # (non-diagonal) j,j' basis,is over the product of even and
        # odd functions.

        #self.g = xp.linspace(0, xp.pi, args.Ng, endpoint=True)  # can't use this form for torch
        self.g = xp.asarray([i*xp.pi/(args.Ng-1) for i in range(args.Ng)]) # include the endpoint

        self.j  = xp.arange(0,args.Ng)
        self.Om = xp.arange(-self.J, self.J+1)

        self.axes = (self.R, self.r, self.j, self.Om)

        R_rgrid, r_rgrid, g_rgrid = xp.meshgrid(self.R, self.r, self.g, indexing='ij')
        self.Vgrid = self.V(R_rgrid, r_rgrid, g_rgrid)

        assert not xp.any(self.Vgrid)==xp.nan

        self.shape = self.Vgrid.shape + (len(self.Om),)

        with timer_ctx("Build Vsph from Vgrid"):
            # self.Vsph, self.Vint, self.Pjk  = self.buildVsph()
            self.Vint, self.Pjk  = self.buildVsph()

        # Clebsch-Gordon Coefficients between adjacent Ω
        self.VOm = self.buildVOm()

        self.size = xp.prod(xp.asarray(self.shape))

        dR = self.R[1] - self.R[0]
        dr = self.r[1] - self.r[0]

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
        # self.ddr2, _ = KE_Borisov_3D(self.r, bare=True)
        self.ddr2 = KE(args.Nr, self.r[1]-self.r[0], bare=True, cyclic=False)

        self.ddr_lab2, _ = KE_Borisov_3D(self.r_lab, bare=True)
        self.ddR_lab2    = KE(args.NR, self.R_lab[1]-self.R_lab[0], bare=True, cyclic=False, stencil_size=stencil_R)

        # since we need these in Hx
        R_grid, r_grid, _ , _ = xp.meshgrid(self.R, self.r, self.j, self.Om, indexing='ij')
        self.Rinv2 = 1.0/(R_grid)**2
        self.rinv2 = 1.0/(r_grid)**2

        self.diag = self.buildDiag()

        if not hasattr(args, "preconditioner"):
            args.preconditioner = 'naive'

        self.args = args

        builder, self.preconditioner, self.make_guess = {
            'BO':     (self._build_preconditioner_BO, self._preconditioner_BO,    self._make_guess_BO),
            'naive':  (lambda: (self.diag,),          self._preconditioner_naive, self._make_guess_naive),
            None:     (lambda: (self.diag,),          self._preconditioner_naive, self._make_guess_naive),
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


    def sph_transform(self, Vgrid, j1, j2, Om):
        ''' returns (int dγ sin(γ)
                        P1(j1,Ω1,γ) V(r,R,γ, ψ=0) P2(j2,Ω2, γ) )'''

        NR, Nr, Ng, NOm = self.shape

        if j1 < xp.abs(Om) or j2 < xp.abs(Om):  # these terms are excluded from the sum; c.f. eq. 32
            return xp.zeros((NR, Nr))

        def phase(j, Om):  # eq. 31
            c = ((xp.sqrt((2*j + 1) / 2) *
                  xp.sqrt(
                      xp.factorial(j - xp.abs(Om)) /
                      xp.factorial(j + xp.abs(Om))
                  )
                ))

            #if Om > 0 and Om % 2 == 1:  # N.B. lpmv includes the Condon-Shortley phase.
            #    c *= -1

            return c

        # eq. 30
        P1 = phase(j1, Om)*lpmv(xp.abs(Om), j1, xp.cos(self.g))
        P2 = phase(j2, Om)*lpmv(xp.abs(Om), j2, xp.cos(self.g))

        dg = self.g[1]-self.g[0]
        V_jjOmOm = xp.sum(
            (dg*P1*P2*xp.sin(self.g))[None,None,:]*Vgrid,
            axis=-1)

        return V_jjOmOm


    def buildVsph_serial(self):
        ''' V(R,r,j,j',Ω=Ω') '''
        NR, Nr, Nj, NOm = self.shape
        Vsph = xp.zeros((NR, Nr, Nj, Nj, NOm))

        Vsph_ = self.buildVsph_vec()

        for iOm, Om in enumerate(self.Om):
            for ij1, j1 in enumerate(self.j):
                for ij2, j2 in enumerate(self.j):
                    Vsph[:,:,ij1,ij2,iOm] = self.sph_transform(self.Vgrid, j1, j2, Om)
                    # devi = xp.sum(xp.abs(Vsph[:,:,ij1,ij2,iOm] - Vsph_[:,:,ij1,ij2,iOm]))
                    # if devi > 1e-10:
                        # print(devi, f"({ij1},{ij2},{iOm})", f"({j1},{j2},{Om})")

        assert not xp.any(xp.isnan(Vsph))
        # prms(Vsph, Vsph_, "diff Vsph_")
        assert (xp.allclose(Vsph, Vsph_))
        return Vsph
        #return Vsph_


    def buildVsph(self):
        # builds <jΩ|V(R,r)|j'Ω> by transforming over the ɣ and ψ
        # coordinates. (V is not a function of ψ so that part is
        # analytic.)
        Nj = len(self.j)
        m  = xp.abs(self.Om)

        # Precompute all the associated Legendre functions up to Nj, through order J
        # N.B. P has shape (1, Nj, 2J+1, ...) with the 2nd axis in order -J,..0...J
        # so the |Ω| index is in slot (self.J + m)
        Pj = xp.assoc_legendre_p_all(
                Nj - 1, self.J,
                xp.cos(self.g), norm=False)[0, :, m]
        # index with [|Ω|, j, ɣ]

        # phase magnitudes for each j, Om
        def phase(j, Om):  # eq. 31 less sign
            return xp.sqrt((2*j + 1) / 2.0 *
                             xp.factorial(j - abs(Om)) /
                             xp.factorial(j + abs(Om))
                    )

        signs = xp.where((self.Om > 0) & (self.Om % 2 == 1), -1, 1)
        phases = phase(self.j, self.Om[:, None]) * signs[:, None]

        # mask to remove j < |Ω|
        mask = self.j[None, :] >= m[:, None]
        # Apply mask and signed phases
        Pj = Pj * (mask * phases)[...,None]

        # Pmj(ɣ)Pmk(ɣ)
        Pjk = Pj[:, :, None, :] * Pj[:, None, :, :]

        dg = self.g[1] - self.g[0]
        Vint = dg * self.Vgrid * xp.sin(self.g)[None,None,:]

        kwargs = dict(optimize=True)
        if xp.backend == 'torch':
            kwargs = {}

        #Vsph = xp.einsum('Rrg,Ojkg->RrjkO', Vint, Pjk, **kwargs)

        # Storage of various objects at -R 90 -r 91 -g 92 -J 10
        # print(Pj.shape, Pj.nbytes/(1<<20))                #     1 MB
        # print(Pjk.shape, Pjk.nbytes/(1<<20))              #   125 MB
        # print(integrand.shape, integrand.nbytes/(1<<20))  #     6 MB
        # print(Vsph.shape, Vsph.nbytes/(1<<20))            # 11106 MB (11 GB)
        # exit()

        # FIXME: Given how huge Vsph is, if we're memory limited, we
        # might want to consider having the Tx function operate
        # directly with integrand and Pjk. Something like:

        # xa = xp.random.random((4,) + self.shape)
        # with timer_ctx("Tx : xa, Vsph; explicit"):
        #     vout = xp.einsum('BRrjO, RrjkO-> BRrkO', xa, Vsph, **kwargs)

        # with timer_ctx("Tx : xa, integrand, Pjk; implicit"):
        #     vout1 = xp.einsum('BRrjO,Rrg,Ojkg->BRrkO', xa, integrand, Pjk, **kwargs)

        # assert(xp.allclose(vout, vout1))

        # Looks like this will definitely be the right thing to do on
        # GPU because the variant that doesn't explicitly construct
        # Vsph is *faster* than the version that does at size 90 91 92
        # J=10 on the grace hopper node. The numpy backend sees the
        # implict Vx take an order of magnitude longer than the
        # explicit at that size.

        #assert not xp.any(xp.isnan(Vsph))
        #return Vsph, Vint, Pjk
        return Vint, Pjk

    def buildVOm(self):
        ''' Clebsch-Gordon Coefficients between adjacent Ω
            √(J(J+1)-Ω(Ω±1))√(j(j+1)-Ω(Ω±1)),
            shape: Ng x NΩ x NΩ '''
        NR, Nr, Nj, NOm = self.shape
        VOm = xp.zeros((Nj, NOm, NOm))

        # NB: recall self.Om = [-J, -J+1 ...0...J-1,J]
        # will not appear tridiagonal with this matrix element ordering!
        j, J = self.j, self.J
        for i, Oi in enumerate(self.Om):
            for k, Ok in enumerate(self.Om):
                s = Oi - Ok
                if abs(s) != 1 : continue
                VOm[:,i,k] = xp.sqrt(
                                 (J*(J+1) - Oi*Ok) *
                    xp.maximum(0, j*(j+1) - Oi*Ok)
                )


        # Overkill, vectorized version for reference
        # OO = self.Om[:, None] - self.Om[None, :]  # like xp.subtract.outer
        # i, k = xp.where(xp.abs(OO) == 1)
        # s = OO[i, k]
        # Oi = self.Om[i]
        # VOm[:, i, k] = xp.sqrt(              (J*(J+1)           - Oi*(Oi+s)) *
        #                        xp.maximum(0, (j*(j+1))[:, None] - Oi*(Oi+s)))

        assert (not xp.any(xp.isnan(VOm)))
        return VOm

    # allows H @ x
    def __matmul__(self, other):
        return self.Hx(other).reshape(other.shape)

    #@partial(jax.jit, static_argnums=0)
    def Hx(self, x):
        return self.Tx(x) + self.Vx(x)

    def Vx(self,x):
        kwargs = dict(optimize=True)
        if xp.backend == 'torch':
            xa = x.reshape((-1,) + self.shape).type(self.dtype)
            kwargs = {}
        else:
            xa = x.reshape((-1,) + self.shape).astype(self.dtype)

        #vout = xp.einsum('BRrjO, RrjkO-> BRrkO', xa, self.Vsph, **kwargs)
        vout = xp.einsum('BRrjO,Rrg,Ojkg->BRrkO', xa, self.Vint, self.Pjk, **kwargs)
        #assert xp.allclose(vout1, vout)
        return vout.reshape(x.shape)


    #@partial(jax.jit, static_argnums=0)
    def Tx(self, x):
        if xp.backend == 'torch':
            xa = x.reshape((-1,) + self.shape).type(self.dtype)
            kwargs = {}
        else:
            xa = x.reshape((-1,) + self.shape).astype(self.dtype)
            kwargs = dict(optimize=True)

        ke = xp.zeros_like(xa)

        # Radial Kinetic Energy terms, easy
        ke += xp.einsum('BRrjO,RS->BSrjO', xa, self.ddR2, **kwargs)  # ∂²/∂R²
        ke += xp.einsum('BRrjO,rs->BRsjO', xa, self.ddr2, **kwargs)  # ∂²/∂r²

        # Angular electronic ke terms: -j(j+1)(1/r² + 1/R²)
        kej = xp.einsum('BRrjO, j-> BRrjO', xa, self.j*(self.j+1), **kwargs)  # j(j+1)
        # kej = xa*self.j_grid*(self.j_grid+1) # we don't have a j_grid defined yet?
        ke -= (self.Rinv2 + self.rinv2)*kej  # -j(j+1)(1/r² + 1/R²)


        # Angular Kinetic Energy J terms
        if self.J != 0:
            keJdiag  = -xa * self.J * (self.J+1)                       # -J(J+1)
            keJdiag += 2*xp.einsum('BRrjO,O-> BRrjO', xa, self.Om**2, **kwargs)  # -J(J+1)+2Ω²

            keJoffdiag = xp.einsum('BRrjO,jOP-> BRrjP', xa, self.VOm, **kwargs)  # √(J(J+1)-Ω(Ω±1))√(j(j+1)-Ω(Ω±1))
            ke += self.Rinv2*(keJdiag + keJoffdiag)

        # mass portion of KE
        ke *= -1/(2*self.mu)
        return ke.reshape(x.shape)


    # N.B. This section *must* be kept in sync with Hx above
    def buildDiag(self):
        ke  = xp.zeros(self.shape)
        ke += xp.diag(self.ddR2)[:, None, None, None]
        ke += xp.diag(self.ddr2)[None, :, None, None]
        ke -= (self.Rinv2 + self.rinv2) * (self.j*(self.j+1))[None, None, :,None]

        # Angular Kinetic Energy J terms
        if self.J != 0:
            ke += self.Rinv2 * ( 2*self.Om**2
                -self.J*(self.J+1) )[None,None,None,:]

        # mass portion of KE
        ke *= -1 / (2*self.mu)

        Vdiag = xp.einsum('Rrg,Ojjg->RrjO', self.Vint, self.Pjk)
        #Vdiag1 = xp.einsum('RrjjO-> RrjO', self.Vsph)
        #assert xp.allclose(Vdiag, Vdiag1)

        # Potential terms
        diag = Vdiag + ke

        assert not xp.any(xp.isnan(diag))
        return diag.ravel()

    # FIXME: See concerns about jit-ing Hx. Currently jitting in the
    # @partial(jax.jit, static_argnums=0) fashion will break; not sure why.

    def _make_guess_naive(self, min_guess):
        Vdiag = xp.einsum('Rrg,Ojjg->RrjO', self.Vint, self.Pjk)
        guesses = xp.exp(-(Vdiag - xp.min(Vdiag))**2/27.211**2).ravel()
        #return xp.random.random(guesses.shape)
        return guesses

    #@partial(jax.jit, static_argnums=0)
    def _preconditioner_naive(self, dx, e, x0):
        diagd = self.diag - (e - 1e-5)
        return dx/diagd

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
        # Te  =  ∂²/∂r² - (1/r²)j(j+1) - (1/R²)(j(j+1) + J(J+1) - 2Ω²)
        # VOm = (1/R²)√(J(J+1) - Ω(Ω ± 1))√(j(j+1) - Ω(Ω ± 1)) ; (1/R²)*self.VOm
        # N.B. self.ddr2 = ∂²/∂r² + 1/4/r²
        Hel = xp.empty((NR, Nelec, Nelec), dtype=self.dtype)

        # build *bare* Te first
        # R-independent terms: ∂²/∂r² - (1/r²)j(j+1)
        Hel[:] = (
            xp.kron(self.ddr2, xp.eye(Nsph)) -   # ∂²/∂r²
            kron3(xp.diag(1 / self.r**2),        # -(1/r²)j(j+1)
                  xp.diag(self.j*(self.j+1)),
                  xp.eye(NOm))
        )


        # R-dependent terms: (1/R²)j(j+1)
        Rinv2 = (1 / self.R**2)[Ridx, None, None]  # (1/R²), ready for broadcasting
        Hel -= Rinv2 * kron3(xp.eye(Nr),           # -(1/R²) * j(j+1)
                             xp.diag(self.j*(self.j+1)),
                             xp.eye(NOm))[None]

        # J terms: -(1/R²)J(J+1) + 2Ω²/R²
        if self.J != 0:
            Hel[:, xp.arange(Nelec), xp.arange(Nelec)] -= (
                Rinv2[0] * self.J * (self.J+1)  # -(1/R²) J(J+1)
            )
            Hel += 2 * kron3(xp.eye(Nr), xp.eye(Nj), xp.diag(self.Om**2)) * Rinv2 # + 2Ω²/R²

        kwargs = dict(optimize=True)
        if xp.backend == 'torch':
            kwargs = {}

        # VOm term:
        VOm_big = xp.einsum('jOP,ij->iOjP', self.VOm, xp.eye(Nj), **kwargs).reshape(Nsph, Nsph)

        Hel += xp.kron(xp.eye(Nr), VOm_big) * Rinv2
        Hel *= -1 / (2 * self.mu)  # -1/2/μ · (Te + VOm)

        # N.B. While one might be tempted to write the output as
        # RrsjkOP, recall that when we reshape, we need to make sure
        # that we have Rx(Nelec)x(Nelec) => Rx(jrO)x(ksP). This
        # repeats the ordering of the indices that matches kron3.

        # Vsph_big1 = xp.einsum("rs,OP,RrjkO->RjrOksP",
        #                      xp.eye(Nr), xp.eye(NOm),
        #                      self.Vsph[Ridx]).reshape(NR, Nelec, Nelec)

        Hel += xp.einsum("rs,OP,Rrg,Ojkg->RjrOksP",
                         xp.eye(Nr), xp.eye(NOm),
                         self.Vint[Ridx], self.Pjk, **kwargs).reshape(NR, Nelec, Nelec)

        # assert xp.allclose(Vsph_big, Vsph_big1)
        return xp.squeeze(Hel)


    def _build_preconditioner_BO(self):
        print("Building U_n")
        NR, *other = self.shape
        Nelec = numpy.prod(other)

        with timer_ctx("Build Hel"):
            Hel = self.build_Hel()

        with timer_ctx(f"Diag  Hel"):
            if xp.backend == 'numpy':
                threadctl = ThreadpoolController()
                with threadctl.limit(limits=1), cf.ThreadPoolExecutor(max_workers=self.max_threads) as ex:
                    result = ex.map(lambda i: (i, xp.linalg.eigh(self.build_Hel(i))), range(NR))
                    U_n   = xp.zeros((NR, Nelec, Nelec), dtype=self.dtype)
                    Ad_n  = xp.zeros((NR, Nelec))
                    for i, (a, u) in result:
                        Ad_n[i] = a
                        U_n[i]  = u
            else:
                Ad_n, U_n = xp.linalg.eigh(Hel)

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

    def _make_guess_BO(self, min_guess):
        Ad_vn, U_n, U_v, *_ = self._preconditioner_data
        # BO states are like: U_n[:,:,n]
        # vib states are like: U_v[n,:,v]
        s = int(numpy.ceil(numpy.sqrt(min_guess)))

        guesses = xp.stack([
            (U_n[:,:,n] * U_v[n,:,v,xp.newaxis]).ravel()
            for n in range(s) for v in range(s)
        ])

        return guesses

    #@partial(jax.jit, static_argnums=0)
    def _preconditioner_BO(self, dx, e, _):
        Ad_vn, U_n, U_v, *_ = self._preconditioner_data
        diagd = Ad_vn - (e - 1e-5)
        NR, *Nother = self.shape
        dx_ = dx.reshape((-1, NR, xp.prod(Nother)))

        #FIXME: precompute optimal einsum path and provide that
        kwargs = dict(optimize=True)
        if xp.backend == 'torch':
            kwargs = {}

        tr_ = xp.einsum(
            'Rij,jRq,qj,jmq,mpj,Bmp->BRi',
            U_n, U_v, 1.0 / diagd, U_v, U_n, dx_, **kwargs
        )

        return tr_.reshape(dx.shape)


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
    parser.add_argument('--potential', choices=['soft_coulomb', 'borgis', 'erf_coulomb'],
                        default='borgis')
    parser.add_argument('--extent', metavar="X", action=ArrayAction,
                        nargs=3, help="Rmin Rmax rmax, in Bohr "
                        "(typically set automatically)")
    parser.add_argument('--exact_diagonalization', action='store_true')
    parser.add_argument('--bo_spectrum', metavar='spec.npz', type=Path, default=None)
    parser.add_argument('--preconditioner', choices=['naive', 'BO'],
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
            # with numpy.printoptions(precision=4):
            #     print(Ad_n.T[0] - Ad_n.T[0,-1])
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
        if hasattr(evecs, 'get'):
            evecs = evecs.get()
        numpy.savez_compressed(args.evecs, guess=evecs, H=H)
        print("Wrote eigenvectors to", args.evecs)

    if args.bo_spectrum:
        bo = Ad_vn[1,0] - Ad_vn[0,0]
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
