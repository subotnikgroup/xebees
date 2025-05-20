#!/usr/bin/env python
import jax
import jax.numpy as jnp
jax.config.update('jax_enable_x64', True)

import numpy as np
from numpy.fft import fft, fftshift
from scipy.integrate import simpson
from scipy.sparse.linalg import lobpcg

from sys import stderr
import argparse as ap
from pathlib import Path

import concurrent.futures as cf
from itertools import product
from functools import reduce, partial
import operator

from pyscf import lib as pyscflib
import linalg_helper as lib

from constants import *
from hamiltonian import  KE, KE_FFT, KE_Borisov
from davidson import phase_match, get_interpolated_guess, get_davidson_mem, solve_exact_gen, eye_lazy
from debug import prms, timer, timer_ctx
from threadpoolctl import ThreadpoolController
    
if __name__ == '__main__':

    from tqdm import tqdm
else:  # mock these out for use in Jupyter Notebooks etc
    # from contextlib import contextmanager
    
    # class ThreadpoolController:
    #     @contextmanager
    #     def limit(_, limits):
    #         print(f"Mock call to ThreadpoolController.limit(limits={limits})")
    #         yield

    def tqdm(iterator, **kwargs):
        print(f"Mock call to tqdm({kwargs})")
        return iterator


class Hamiltonian:
    __slots__ = ( # any new members must be added here
        'm_e', 'M_1', 'M_2', 'mu', 'mu12', 'aa', 'g_1', 'g_2', 'J',
        'R', 'P', 'R_grid', 'r', 'p', 'r_grid', 'g', 'pg', 'g_grid',
        'axes',
        'max_threads',
        'preconditioner', 'make_guess',
        'Vgrid', 'ddR2', 'ddr2', 'ddg2', 'ddg1',
        'Rinv2', 'rinv2', 'diag', '_preconditioner_data',
        'shape', 'size',
        '_locked', '_hash'
    )

    def __init__(self, args):
        self.m_e = AMU_TO_AU * 1
        self.M_1 = AMU_TO_AU * args.M_1
        self.M_2 = AMU_TO_AU * args.M_2

        self.mu  = np.sqrt(self.M_1*self.M_2*self.m_e/(self.M_1+self.M_2+self.m_e))
        self.mu12 = self.M_1*self.M_2/(self.M_1+self.M_2)
        self.aa = np.sqrt(self.mu/self.mu12) # factor of 'a' for lab and scaled coordinates

        self.g_1 = args.g_1
        self.g_2 = args.g_2

        self.J   = args.J

        # Grid setup
        # Scale coords so we see R \on [2,4] and r \on (0, 5] for M1=M2=1
        # FIXME: may need to pick ranges based on charges too
        R_range = np.array([1.861,3.722]) * self.aa
        r_max   = 5.373 / self.aa

        # (R_min, R_max, r_max)
        if hasattr(args, "extent") and args.extent is not None:
            R_range = args.extent[:2]
            r_max = args.extent[-1]

        # save number of threads for preconditioner
        self.max_threads = 1
        if hasattr(args, "t") and args.t is not None:
            self.max_threads = args.t
            
        self.R = np.linspace(*R_range, args.NR) * ANGSTROM_TO_BOHR

        # N.B.: We are careful not to include 0 in the range of r by
        # starting 1 "step" away from 0. It might be more consistent
        # to have Nr-1 points, but the confusion this would cause
        # would be intolerable. This behavior is required because we
        # have terms that go like 1/r.
        self.r = np.linspace(r_max/args.Nr, r_max, args.Nr) * ANGSTROM_TO_BOHR

        # require Ng to be even
        if args.Ng % 2 != 0:
            raise RuntimeError(f"Ng must be even!")
        
        # N.B.: It is essential that we not include the endpoint in
        # gamma lest our cyclic grid be ill-formed and 2nd derivatives
        # all over the place
        self.g = np.linspace(0, 2*np.pi, args.Ng, endpoint=False)

        self.axes = (self.R, self.r, self.g)
        
        self.R_grid, self.r_grid, self.g_grid = np.meshgrid(self.R, self.r, self.g, indexing='ij')
        self.Vgrid = self.V_2Dfcm(self.R_grid, self.r_grid, self.g_grid)
        self.shape = self.Vgrid.shape
        self.size = np.prod(self.shape)

        dR = self.R[1] - self.R[0]
        dr = self.r[1] - self.r[0]
        dg = self.g[1] - self.g[0]

        self.P  = np.fft.fftshift(np.fft.fftfreq(args.NR, dR)) * 2 * np.pi
        self.p  = np.fft.fftshift(np.fft.fftfreq(args.Nr, dr)) * 2 * np.pi
        self.pg = np.fft.fftshift(np.fft.fftfreq(args.Ng, dg)) * 2 * np.pi

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
        self.ddR2, _ = KE_Borisov(self.R, bare=True)
        self.ddr2, _ = KE_Borisov(self.r, bare=True)

        # Part of the reason for using a cyclic *stencil* for gamma
        # rather than KE_FFT is that it wasn't immediately obvious how
        # I would represent ∂/∂γ. (∂²/∂γ² was clear.)  N.B.: The
        # default stencil degree is 11
        self.ddg2 = KE(args.Ng, dg, bare=True, cyclic=True)
        self.ddg1 = KE(args.Ng, dg, bare=True, cyclic=True, order=1)

        # since we need these in Hx; maybe fine to compute on the fly?
        self.Rinv2 = 1.0/(self.R_grid)**2
        self.rinv2 = 1.0/(self.r_grid)**2

        self.diag = self.buildDiag()

        preconditioner = 'naive'
        if hasattr(args, "preconditioner"):
            preconditioner = args.preconditioner
        
        builder, self.preconditioner, self.make_guess = {
            'BO':     (self._build_preconditioner_BO, self._preconditioner_BO,    self._make_guess_BO),
            'V1':     (self._build_preconditioner_V1, self._preconditioner_V1,    self._make_guess_V1),
            'naive':  (lambda: (self.diag,),          self._preconditioner_naive, self._make_guess_naive),
            'power':  (lambda: (self.diag,),          self._preconditioner_power, self._make_guess_naive),
            'diagbo': (self._build_preconditioner_BO, self._preconditioner_naive, self._make_guess_BO),
            None:     (lambda: (self.diag,),          self._preconditioner_naive, self._make_guess_naive),
            }[preconditioner]

        with timer_ctx(f"Build preconditioner {preconditioner}"):
            self._preconditioner_data = builder()

        # Lock the object and protect arrays from writing
        for key in self.__slots__:
            if (hasattr(self, key) and 
                isinstance(member := super().__getattribute__(key), np.ndarray)):
                member.flags.writeable = False

        self._hash = self._make_hash()
        self._locked = True


    def V_2Dfcm(self, R_amu, r_amu, gamma):
        R = R_amu / ANGSTROM_TO_BOHR
        r = r_amu / ANGSTROM_TO_BOHR

        D, d, a, c = 60, 0.95, 2.52, 1
        A, B, C = 2.32e5, 3.15, 2.31e4

        mu12 = self.mu12
        aa = self.aa
        M_1 = self.M_1
        M_2 = self.M_2
        
        kappa2 = r*R*np.cos(gamma)
        r1e = np.sqrt((aa*r)**2 + (R/aa)**2*(mu12/M_1)**2 - 2*kappa2*mu12/M_1)
        re2 = np.sqrt((aa*r)**2 + (R/aa)**2*(mu12/M_2)**2 + 2*kappa2*mu12/M_2)
        
        D2 = self.g_2 * D * (    np.exp(-2*a * (re2-d))
                                 - 2*np.exp(  -a * (re2-d))
                                 + 1)
        D1 = self.g_1 * D * c**2 * (    np.exp(-(2*a/c) * (r1e-d))
                                        - 2*np.exp(-(  a/c) * (r1e-d)))

        #return KCALMOLE_TO_HARTREE * (D1 + D2 + (A*np.exp(-B*R/aa) - C/(R/aa)**6))
        return KCALMOLE_TO_HARTREE * (D1 + D2 +
                                      self.g_1*self.g_2*(A*np.exp(-B*R/aa) - C/(R/aa)**6))

    # allows H @ x
    def __matmul__(self, other):
        return self.Hx(other).reshape(other.shape)

    # FIXME: can likely speed these up using opt_einsum for constant arguments
    @partial(jax.jit, static_argnums=0)
    def Hx(self, x):
        return self.Tx(x) + x.ravel() * self.Vgrid.ravel()

    @partial(jax.jit, static_argnums=0)
    def Tx(self, x):
        xa = x.reshape(self.shape)
        ke = np.zeros(self.shape)

        # Radial Kinetic Energy terms, easy
        ke += jnp.einsum('Rrg,RS->Srg', xa, self.ddR2)  # ∂²/∂R²
        ke += jnp.einsum('Rrg,rs->Rsg', xa, self.ddr2)  # ∂²/∂r²

        #  ∂²/∂γ² + 1/4 terms
        keg  = jnp.einsum('Rrg,gh->Rrh', xa, self.ddg2)  # ∂²/∂γ²
        ke += (self.Rinv2 + self.rinv2)*keg              # (1/R^2 + 1/r^2) (∂²/∂γ²)

        # Angular Kinetic Energy J terms
        if self.J != 0:
            keg  = xa*self.J**2                                       # J^2
            keg -= 2j*self.J*jnp.einsum('Rrg,gh->Rrh', xa, self.ddg1)  # J^2 - 2J ∂/∂γ
            ke += self.Rinv2*keg                                 # (1/R^2)*(J^2 - 2J ∂/∂γ)

        # mass portion of KE
        ke *= -1 / (2*self.mu)
        return ke.ravel()

    
    # N.B. This section *must* be kept in sync with Hx above
    def buildDiag(self):
        # ke = sum(np.diag(op).reshape(
        #         [self.shape[i] if i == axis else 1 for i in range(3)]
        #     )
        #     for axis, op in [(0, self.ddR2), (1, self.ddr2)])
        ke  = np.zeros(self.shape)
        ke += np.diag(self.ddR2)[:, None, None]
        ke += np.diag(self.ddr2)[None, :, None]
        ke += (self.Rinv2 + self.rinv2) * np.diag(self.ddg2)[None, None, :]
        
        # Angular Kinetic Energy J terms
        if self.J != 0:
            ke += self.Rinv2 * (
                self.J**2 -
                2*self.J*np.ones(self.shape) * np.diag(self.ddg1)[None, None, :]
            )

        # mass portion of KE
        ke *= -1 / (2*self.mu)

        # Potential terms
        diag = self.Vgrid + ke

        return diag.ravel()

    # FIXME: See concerns about jit-ing Hx. Currently jitting in the
    # @partial(jax.jit, static_argnums=0) fashion will break; not sure why.

    def _make_guess_naive(self, min_guess):
        guesses = np.exp(-(self.Vgrid - np.min(self.Vgrid))**2/27.211).ravel()
        return guesses

    @partial(jax.jit, static_argnums=0)
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

    def BO_spectrum(self, nroots=None):
        print("Building BO spectrum")
        NR, Nr, Ng = self.shape
        Nelec = Nr*Ng
        Ad_n  = np.zeros((NR, Nelec))
        Ad_vn = np.zeros((NR, Nelec))
        
        def diag_Hel(i, Ad_n=Ad_n):
            R = self.R[i]
            Hel = (
                -1/(2*self.mu)*(
                    np.kron(self.ddr2, np.eye(Ng)) +
                    np.kron((1/R**2 + np.diag(1/self.r**2)), self.ddg2) +
                    0 # FIXME: need J terms
                ) + np.diag(self.Vgrid[i].ravel())
            )
            Ad_n[i] = np.linalg.eigvalsh(Hel)

        threadctl = ThreadpoolController()
        with cf.ThreadPoolExecutor(max_workers=self.max_threads) as ex, threadctl.limit(limits=1):
            list(tqdm(
                ex.map(diag_Hel, range(NR)),
                total=NR, desc="Building electronic surfaces"))

        def diag_Hbo(i, Ad_n=Ad_n, Ad_vn=Ad_vn):
            Hbo = -1/(2*self.mu)*self.ddR2 + np.diag(Ad_n[:,i])
            Ad_vn[:,i] = np.linalg.eigvalsh(Hbo)

        with cf.ThreadPoolExecutor(max_workers=self.max_threads) as ex, threadctl.limit(limits=1):
            list(tqdm(
                ex.map(diag_Hbo, range(Nelec)),
                total=Nelec, desc="Building vibrational states"))

        for i in range(nroots):
            with np.printoptions(linewidth=np.inf):
                print(f"BO state {i} spectrum:", Ad_vn[:nroots,i])
        return Ad_vn  # energies are Ad_vn[v,n]

    
    def _build_preconditioner_BO(self):
        NR, Nr, Ng = self.shape
        Nelec = Nr*Ng

        U_n   = np.zeros((NR, Nr*Ng, Nelec))
        U_v   = np.zeros((Nelec, NR, NR))
        Ad_n  = np.zeros((NR, Nelec))
        Ad_vn = np.zeros((NR, Nelec))
        
        def diag_Hel(i, Ad_n=Ad_n, U_n=U_n):
            R = self.R[i]
            Hel = (
                -1/(2*self.mu)*(
                    np.kron(self.ddr2, np.eye(Ng)) +
                    np.kron((1/R**2 + np.diag(1/self.r**2)), self.ddg2) +
                    0 # FIXME: need J terms
                ) + np.diag(self.Vgrid[i].ravel())
            )
            print(i, Hel.shape)
            Ad_n[i], U_n[i] = np.linalg.eigh(Hel)

        threadctl = ThreadpoolController()
        with cf.ThreadPoolExecutor(max_workers=self.max_threads) as ex, threadctl.limit(limits=1):
            list(tqdm(
                ex.map(diag_Hel, range(NR)),
                total=NR, desc="Building U_n"))

        phase_match(U_n)

        def diag_Hbo(i, Ad_n=Ad_n, Ad_vn=Ad_vn, U_v=U_v):
            Hbo = -1/(2*self.mu)*self.ddR2 + np.diag(Ad_n[:,i])
            Ad_vn[:,i], U_v[i] = np.linalg.eigh(Hbo)

        with cf.ThreadPoolExecutor(max_workers=self.max_threads) as ex, threadctl.limit(limits=1):
            list(tqdm(
                ex.map(diag_Hbo, range(Nelec)),
                total=Nelec, desc="Building U_v"))

        phase_match(U_v)

        Ad_vn.flags.writeable = False
        U_n.flags.writeable   = False
        U_v.flags.writeable   = False

        return (Ad_vn, U_n, U_v)

    def _make_guess_BO(self, min_guess):
        Ad_vn, U_n, U_v, *_ = self._preconditioner_data
        # BO states are like: U_n[:,:,n]
        # vib states are like: U_v[n,:,v]
        s = int(np.ceil(np.sqrt(min_guess)))

        guesses = [
            (U_n[:,:,n] * U_v[n,:,v,np.newaxis]).ravel()
            for n in range(s) for v in range(s)
        ]

        return guesses
    
    #@partial(jax.jit, static_argnums=0)
    def _preconditioner_BO(self, dx, e, x0):
        Ad_vn, U_n, U_v, *_ = self._preconditioner_data
        diagd = Ad_vn - (e - 1e-5)

        NR, Nr, Ng = self.shape
        Nelec = Nr*Ng

        dx_ = dx.reshape((NR, Nelec))

        dx_vn = np.einsum('nji,jqn,jq->in', U_v, U_n, dx_, optimize=True)
        tr_vn = dx_vn / diagd
        tr_ = np.einsum('Rij,jRq,qj->Ri', U_n, U_v, tr_vn, optimize=True)

        return tr_.ravel()


    def _build_preconditioner_V1(self, min_guess=4):
        NR, Nr, Ng = self.shape
        dg = self.g[1] - self.g[0]

        threadctl = ThreadpoolController()
        threadctl.limit(limits=1)

        j_full = fftshift(np.arange(Ng)-Ng//2)
        COS = np.cos(2*j_full[:,None] * self.g)
        V1 = simpson(self.Vgrid[:,:,:], dx=dg, axis=-1)/np.pi/2

        # V1 is V2(j==0)
        # IF debugging
        #V2 = simpson(self.Vgrid[:,:,None,:] * COS[None,None,:,:], dx=dg, axis=-1)/np.pi/2
        #assert(np.allclose(V1, V2[:,:,np.squeeze(np.where(j_full == 0)).item()]))
        
        Ad = np.zeros((NR, Nr, Ng))
        U  = np.zeros((NR, Nr, Ng, Nr))

        def diag_H0(args, Ad=Ad, U=U):
            Ri, (ji, j) = args
            H_el = -(
                self.ddr2 + np.diag(j**2/self.r**2 + (self.J-j)**2/self.R[Ri]**2)
            )/2/self.mu + np.diag(V1[Ri])
            Ad[Ri, :, ji], U[Ri, :, ji, :] = np.linalg.eigh(H_el)


        # Presumably because of gated access, this tops out pretty fast at ~
        with cf.ThreadPoolExecutor(max_workers=self.max_threads) as ex:
            list(tqdm(
                ex.map(diag_H0, product(range(NR),enumerate(j_full))),
                total=NR*Ng, desc="Building Preconditioner", delay=3))

        # Align phases by iterating over Nr eigen vectors at each Ri, ji
        for Ri, ji in tqdm(product(range(NR),
                                   range(Ng)),
                           total=NR*Ng,
                           desc="Phase Matching",
                           delay=3, # only display if longer than 3 seconds
                           ):
            for n in range(Nr):
                current = (Ri, slice(None), ji, n)

                if Ri == 0 and ji == 0:   # consistent, arbitrary reference phase
                    reference = (0, 0, 0, 0)
                elif ji > 0:              # line up with previous j
                    reference = (Ri, slice(None), ji-1, n)
                elif ji == 0 and Ri > 0:  # line up with previous R
                    reference = (Ri-1, slice(None), ji, n)
                else:
                    raise RuntimeError("We should always have a reference!")

                # be careful not to undo your work!! (Ri=0)-1=-1
                # indexes the last one and breaks things!

                # if any(map(lambda x: x < 0 if type(x) is not slice else False,
                #             current + reference
                #             )):
                #     raise RuntimeError(f"invalid reference: {current(0)}, {reference(0)}")

                # actually match the phase
                if np.sum(U[current] * U[reference]) < 0:
                    U[current] *= -1

        # yolo
        #oddjs = j_full%2 == 1
        #U[:, :, oddjs, :] = 0
        #U = fft(U, axis=2)
        #assert(np.mean(np.abs(U.imag)) < 1e-12)
        #U = U.real

        U = np.fft.ifft(U, axis=2)
        assert(np.mean(np.abs(U.imag)) < 1e-12)
        U = U.real
        
        Ad.flags.writeable = False
        U.flags.writeable  = False

        return (Ad, U)

    def _make_guess_V1(self, min_guess):
        Ad, U, *_ = self._preconditioner_data
        NR, Nr, Ng = self.shape
        # States are U[R, :, Ng//2 + j, n]
        s = int(np.ceil(np.sqrt(min_guess)))
        guesses = [
            np.copy(np.broadcast_to(
                U[:, :, Ng//2 + j, i][:, :, np.newaxis],
                self.shape
            )).ravel() for i in range(s) for j in range(s)]

        return guesses

    #@partial(jax.jit, static_argnums=0)
    def _preconditioner_V1(self, dx, e, x0):
        dx_ = dx.reshape(self.shape)
        Ad, U, *_ = self._preconditioner_data
        diagd = Ad - (e - 1e-5)

        dx_t = np.einsum("Rrgi,Rrg->Rig", U, dx_, optimize=True)
        tr_t = dx_t / diagd
        tr_ = np.einsum('Rigr,Rig->Rrg', U, tr_t, optimize=True)

        return tr_.ravel()

    # Below here are a bunch of things related to immutability
    # https://docs.jax.dev/en/latest/faq.html#how-to-use-jit-with-methods
    def _make_hash(self):
        def recursive_hash(obj):
            if isinstance(obj, np.ndarray):
                if obj.flags.writeable:
                    raise ValueError("Refusing to hash mutable array")
                return hash((obj.shape, obj.dtype, obj.tobytes()))
            elif isinstance(obj, tuple):
                return hash(tuple(recursive_hash(x) for x in obj))
            else:
                return hash(obj)

        return reduce(
            operator.xor,
            (recursive_hash(getattr(self, key)) for key in self.__slots__ if key not in ['_locked', '_hash']),
            0)

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
    
    parser.add_argument('-k', metavar='num_eigenvalues', default=5, type=int)
    parser.add_argument('-t', metavar="num_threads", default=16, type=int)
    parser.add_argument('-g_1', metavar='g_1', required=True, type=float)
    parser.add_argument('-g_2', metavar='g_2', required=True, type=float)
    parser.add_argument('-M_1', required=True, type=float)
    parser.add_argument('-M_2', required=True, type=float)
    parser.add_argument('-J', default=0, type=float)
    parser.add_argument('-R', dest="NR", metavar="NR", default=101, type=int)
    parser.add_argument('-r', dest="Nr", metavar="Nr", default=400, type=int)
    parser.add_argument('-g', dest="Ng", metavar="Ng", default=250, type=int)
    parser.add_argument('--extent', metavar="X", default=None,
                        type=float, nargs=3, help="Rmin Rmax rmax, but set automatically")
    parser.add_argument('--exact_diagonalization', action='store_true')
    parser.add_argument('--bo_spectrum', metavar='spec.npz', type=Path, default=None)
    parser.add_argument('--preconditioner', default="naive", type=str)
    parser.add_argument('--verbosity', default=2, type=int)
    parser.add_argument('--iterations', metavar='max_iterations', default=10000, type=int)
    parser.add_argument('--subspace', metavar='max_subspace', default=1000, type=int)
    parser.add_argument('--guess', metavar="guess.npz", type=Path, default=None)
    parser.add_argument('--evecs', metavar="guess.npz", type=Path, default=None)
    parser.add_argument('--save', metavar="filename")

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print(args)

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
            spec = H.BO_spectrum(args.k)
            np.savez_compressed(args.bo_spectrum, bo_spectrum=spec)

            
    # with timer_ctx(f"LOBPCG of size {np.prod(H.shape)}"):
    #     e_approx_lobpcg, evecs_lobpcg = lobpcg(
    #         lambda xs: np.asarray([ H @ x for x in xs.T ]).T,
    #         np.asarray(guess).T,
    #         tol=1e-6,
    #         maxiter=args.iterations,
    #         verbosityLevel=args.verbosity,
    #         restartControl=4,
    #         largest=False,
    #     )
    
    # FIXME: would like to use a callback to save intermediate
    # wavefunctions in case we need to do a restart.
    with timer_ctx(f"Davidson of size {np.prod(H.shape)}"):
        #conv, e_approx, evecs = pyscflib.davidson1(
        conv, e_approx, evecs = lib.davidson1(
            lambda xs: [ H @ x for x in xs ],
            guess,
            #H.diag,
            H.preconditioner,
            nroots=args.k,
            max_cycle=args.iterations,
            verbose=args.verbosity,
            max_space=args.subspace,
            max_memory=get_davidson_mem(0.75),
            tol=1e-12,
        )

    #guess quality
    #for i, (e,g) in enumerate(zip(evecs, guess)):
    #    print(i, np.abs(np.vdot(e, g))**2 / (np.vdot(e, e) * np.vdot(g, g)))
    
    print("Davidson:", e_approx)
    print(conv)
    print("Exact excitations / meV", (e_approx - e_approx[0]) * HARTREE_TO_EV*1000)

    if args.evecs:
        np.savez_compressed(args.evecs, guess=evecs, H=H)
        print("Wrote eigenvectors to", args.evecs)

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
