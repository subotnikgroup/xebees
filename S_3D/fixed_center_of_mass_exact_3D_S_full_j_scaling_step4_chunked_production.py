#!/usr/bin/env python
"""Production Step 4 chunked J-scaling fixed center-of-mass solver."""

import argparse as ap
from pathlib import Path

import concurrent.futures as cf
from functools import partial
from contextlib import contextmanager
import gc

import os, sys
sys.path.append(os.path.abspath("lib"))

import xp
import numpy  # only use this for reading and writing objects

if os.environ.get('PS_NVTX', '').lower() in ('1', 'true', 'yes', 'on'):
    import linalg_helper_nvtx as lib
else:
    import linalg_helper_locking as lib
import potentials
from constants import *
from hamiltonian import  KE, KE_FFT_R, KE_ColbertMiller_zero_inf
from davidson import phase_match, get_davidson_guess_3D, get_davidson_mem
from analysis import get_wfc_Om_proj_wS, get_jls_expectations, get_p01_radial, get_spin_expectations

from debug import timer_ctx
from threadpoolctl import ThreadpoolController


if __name__ == '__main__':
    from tqdm import tqdm
else:  # mock this out for use in Jupyter Notebooks etc
    def tqdm(iterator, **kwargs):
        print(f"Mock call to tqdm({kwargs})")
        return iterator


def _nvtx_is_enabled():
    return xp.backend == 'cupy' and os.environ.get('PS_NVTX', '').lower() in ('1', 'true', 'yes', 'on')


@contextmanager
def _nvtx_range(name):
    if not _nvtx_is_enabled():
        yield
        return
    pop = None
    try:
        import cupy
        nvtx = cupy.cuda.nvtx
        push = getattr(nvtx, 'RangePush', None) or getattr(nvtx, 'range_push')
        pop = getattr(nvtx, 'RangePop', None) or getattr(nvtx, 'range_pop')
        push(name)
    except Exception:
        pop = None
    try:
        yield
    finally:
        if pop is not None:
            try:
                pop()
            except Exception:
                pass


def _nsys_capture_davidson_enabled():
    return (xp.backend == 'cupy'
            and os.environ.get('PS_NSYS_CAPTURE_DAVIDSON', '').lower()
            in ('1', 'true', 'yes', 'on'))


def _nsys_capture_start():
    if not _nsys_capture_davidson_enabled():
        return False
    import cupy
    cupy.cuda.Stream.null.synchronize()
    cupy.cuda.runtime.profilerStart()
    return True


def _nsys_capture_stop(started):
    if not started:
        return
    import cupy
    cupy.cuda.Stream.null.synchronize()
    cupy.cuda.runtime.profilerStop()


def make_checkpoint_callback(evecs_path, every):
    ''' Build a davidson1 callback that atomically snapshots the current Ritz
    vectors + eigenvalues to <evecs_path>.ckpt.npz every `every` cycles.  Cheap
    insurance for long runs: a wall-time kill (SLURM) or an OOM on a *later* cycle
    then leaves a recoverable snapshot -- reload it with --guess to resume, or
    recompute the diagnostics (char/<l.s>/spin) from it offline.  Returns None
    (davidson1 treats that as "no callback") when disabled or no path is given. '''
    if not evecs_path or not every or every <= 0:
        return None
    ckpt = str(evecs_path) + ".ckpt.npz"
    tmp = ckpt + ".tmp.npz"
    to_host = lambda a: a.get() if hasattr(a, 'get') else numpy.asarray(a)

    def _cb(envs):
        icyc = envs.get('icyc', 0)
        if icyc % every != 0:
            return
        x0 = envs.get('x0'); e = envs.get('e'); conv = envs.get('conv')
        if x0 is None or e is None:
            return
        try:
            # uncompressed savez: fast even for multi-GB Ritz blocks.  os.replace
            # is atomic, so a kill mid-write can never leave a corrupt checkpoint.
            kw = dict(guess=to_host(x0), e_approx=to_host(e), icyc=icyc)
            if conv is not None:
                kw['conv'] = to_host(conv)
            os.makedirs(os.path.dirname(ckpt) or ".", exist_ok=True)
            numpy.savez(tmp, **kw)
            os.replace(tmp, ckpt)
            print(f"[checkpoint] cycle {icyc}: {ckpt}  e[0]={float(kw['e_approx'][0]):.6e}",
                  flush=True)
        except Exception as exc:            # a failed checkpoint must never kill the solve
            print(f"[checkpoint] cycle {icyc}: save FAILED ({exc!r}); continuing", flush=True)
    return _cb


class Hamiltonian:
    __slots__ = ( # any new members must be added here
        'm_e', 'M_1', 'M_2', 'mu', 'mu12', 'mur', 'aa', 'g_1', 'g_2', 'J',
        'R', 'r', 'g', 'j', 'Om','sg','ls',
        'soc_const',
        'axes', 'dtype', 'args',
        'max_threads',
        'preconditioner', 'make_guess', '_Vfunc','_Efunc',
        'Vgrid', 'Vint', 'Pjkst', 'Cspin', 'PC', 'VOm', 'ddR2', 'ddr2',
        'PCmat', 'PCdiag', 'VOm_shift', 'C0k', 'C1k', 'C2k', 'Csc_r',  # Step 4 derived tensors
        'Rinv2', 'rinv2','rinv3', 'diag', '_preconditioner_data',
        'shape', 'size',
        '_locked', '_hash', 'r_lab', 'R_lab', 'ddr_lab2', 'ddR_lab2', 'ddg1',
        'E1', 'E2','Efield', 'C_scnab', 'ddr1',
        's_z', 'l_z', 's_x', 's_y'
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
        print("mass rescaling: mu, aa", self.mu, self.aa)

        self.soc_const =  args.alpha/137**2/self.m_e**2/2 # alpha* g_e/c²me²/4
        print("soc const, alpha", self.soc_const, args.alpha)

        self._Vfunc, extent_func, self._Efunc = {
            'soft_coulomb': (potentials.soft_coulomb, potentials.extents_soft_coulomb, None),
            'borgis': (partial(potentials.borgis, asymmetry_param=1), potentials.extents_borgis, potentials.Efield_borgis),
            'erf_coulomb':(potentials.erf_coulomb, potentials.extents_erf_coulomb, potentials.Efield_coulomb)
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
        

        # # require Ng to be even
        # if args.Ng % 2 != 0:
        #     raise RuntimeError(f"Ng must be even!")

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
        self.g = xp.asarray([i*xp.pi/(args.Nint-1) for i in range(args.Nint)]) # include the endpoint

        self.j  = xp.arange(0.0,args.Ng, dtype=xp.float64)
        self.j[:] += 0.5
        self.Om = xp.arange(-self.J, self.J+1, dtype=xp.float64)
        self.sg = xp.array([-0.5, 0.5])

        kappa = self.sg[None,:]*(2*self.j[:,None]+1)
        # self.ls = -(kappa+1)/2
        self.ls = (self.j[:,None]*(self.j[:,None]+1) 
                            - (self.j[:,None] + self.sg[None,:])*(self.j[:,None]+self.sg[None,:]+1) 
                            - 0.75)/2
        

        self.axes = (self.R, self.r, self.j, self.Om, self.sg)

        R_rgrid, r_rgrid, g_rgrid = xp.meshgrid(self.R, self.r, self.g, indexing='ij')
        self.Vgrid = self.V(R_rgrid, r_rgrid, g_rgrid)
        Vmin = xp.unravel_index(xp.argmin(self.Vgrid), self.Vgrid.shape)
        print("min of Vgrid R", self.Vgrid[Vmin], Vmin)

        self.shape = (len(self.R), len(self.r), len(self.j), len(self.sg), len(self.Om))

        with timer_ctx("Build Vsph from Vgrid"):
            self.Vint, self.Pjkst = self.buildVsph()
            self.Cspin = self.buildVspincoef()
        with timer_ctx("Build PC = Pjkst*Cspin"):
            self.PC = xp.einsum(
                'Ojkstag,jkstOa->Ojkstg',
                self.Pjkst,
                self.Cspin,
                optimize=True,
            )
            self.Pjkst = None
            self.Cspin = None
            gc.collect()
            if xp.backend == 'cupy':
                import cupy
                cupy.cuda.Stream.null.synchronize()
                cupy.get_default_memory_pool().free_all_blocks()
                cupy.get_default_pinned_memory_pool().free_all_blocks()

        # --- Step 4 change 1 & 7: matmul-friendly PC layout + diagonal slice ---
        with timer_ctx("Build PCmat / PCdiag (Step 4)"):
            _, _, Nj_, Nsg_, NOm_ = self.shape
            Njs_ = Nj_ * Nsg_
            # PCmat[O, g, (j,s), (k,t)]  <-  PC[O, j, k, s, t, g]   (axes 0,5,1,3,2,4)
            self.PCmat = xp.ascontiguousarray(
                self.PC.transpose(0, 5, 1, 3, 2, 4)
            ).reshape(NOm_, self.args.Nint, Njs_, Njs_)
            # PCdiag[O, j, s, g] = PC[O, j, j, s, s, g]  (j==k and s==t diagonal)
            ji_ = xp.arange(Nj_)
            si_ = xp.arange(Nsg_)
            self.PCdiag = self.PC[:, ji_[:, None], ji_[:, None],
                                  si_[None, :], si_[None, :], :]

        dg = self.g[1]-self.g[0]
        
        # with xp.printoptions(precision=2, suppress=True):
        #     print("Norm coef:\n", (xp.einsum('jkstOa-> Ojskt', self.Cspin))[1].reshape(len(self.j)*2, len(self.j)*2))
        #     print("Norm Poly")
        #     print(((xp.einsum('Ojkstag, g -> Oajskt', self.Pjkst, dg*xp.sin(self.g))[1]).reshape(2,len(self.j)*2, len(self.j)*2)))
        #     print(self.j[1],self.sg, (xp.einsum('Ojkstag, g -> Oajskt', self.Pjkst, dg*xp.sin(self.g))[1,:,1,:,1,:]))
        #     print((self.j[:,None]+self.sg[None,:]).reshape(self.j.size*2))
        # exit()


        # Clebsch-Gordon Coefficients between adjacent Ω
        self.VOm = self.buildVOm()
        # --- Step 4 change 8: nearest-neighbour Omega coupling coefficients ---
        # VOm_shift[j, m] = VOm[j, m+1, m] couples Omega index m <-> m+1.
        NOm_ = self.shape[-1]
        if NOm_ > 1:
            _m = xp.arange(NOm_ - 1)
            self.VOm_shift = self.VOm[:, _m + 1, _m]          # (Nj, NOm-1)
        else:
            self.VOm_shift = xp.zeros((self.shape[2], 0), dtype=self.dtype)
        # coef for ROI soc, build later
        # if args.soc=='roi':
        self.C_scnab = self.buildC_scnab()
        # --- Step 4 change 6: precompute scnab weighted coefficient tensors ---
        # kappa[j,s] = sigma*(2j+1); C0k,C2k fold kappa[js], C1k folds -kappa[kt].
        _kappa = self.sg[None, :] * (2 * self.j[:, None] + 1)         # (j,s)
        self.C0k = self.C_scnab[0] * _kappa[:, :, None, None, None]   # (j,s,k,t,O)
        self.C1k = self.C_scnab[1] * (-_kappa)[None, None, :, :, None]
        self.C2k = self.C_scnab[2] * _kappa[:, :, None, None, None]
        self.Csc_r = self.C0k + self.C1k + self.C2k                   # 1/r terms combined
        # else: self.C_scnab = None
        self.s_z, self.l_z = self.build_sz_lz()
        self.s_x, self.s_y = self.build_spin_ops()  # body-frame s_x, s_y for <s_x>,<s_y>

        self.size = int(xp.prod(xp.asarray(self.shape)))

        dR = self.R[1] - self.R[0]
        dr = self.r[1] - self.r[0]
        dg = self.g[1] - self.g[0]

        self.E1, self.E2 = self.E_field(R_rgrid,r_rgrid, g_rgrid)
        self.Efield = xp.stack((self.E1,self.E2))

        stencil_R = min(11, args.NR)
        if stencil_R%2==0: stencil_R -= 1

        # self.ddR2    = KE(args.NR, dR, bare=True, cyclic=False, stencil_size = stencil_R)
        P_R = xp.fft.fftfreq(args.NR, dR) * 2 * xp.pi
        self.ddR2 = KE_FFT_R(args.NR, P_R, self.R)

        
        # self.ddr2 = KE(args.Nr, self.r[1]-self.r[0], bare=True, cyclic=False)
        # self.ddr1 = (KE(args.Nr, self.r[1]-self.r[0], bare=True, cyclic=False, order=1)
        #                 )#-xp.diag(1/self.r)) # dr - 1/r due to wfc rescaling
        self.ddr2 = KE_ColbertMiller_zero_inf(args.Nr, dr, bare=True, order=2)
        self.ddr1 = KE_ColbertMiller_zero_inf(args.Nr, dr, bare=True, order=1)
        
        self.ddr_lab2  =  KE_ColbertMiller_zero_inf(args.Nr, self.r_lab[1]-self.r_lab[0], bare=True, order=2)
        self.ddR_lab2  =  KE(args.NR, self.R_lab[1]-self.R_lab[0], bare=True, cyclic=False, stencil_size=stencil_R)
        self.ddg1 = KE(self.g.size, dg, bare=True, order=1, cyclic=False)
                            

        # since we need these in Hx
        R_grid, r_grid, _ , _ , _ = xp.meshgrid(self.R, self.r, self.j, self.sg, self.Om, indexing='ij')
        self.Rinv2 = 1.0/(R_grid)**2
        self.rinv2 = 1.0/(r_grid)**2
        self.rinv3 = 1.0/(r_grid)**3

        self.diag = self.buildDiag()

        if not hasattr(args, "preconditioner"):
            args.preconditioner = 'naive'

        self.args = args

        builder, self.preconditioner, self.make_guess = {
            'BO':     (self._build_preconditioner_BO, self._preconditioner_BO,    self._make_guess_BO),
            'naive':  (lambda: (self.diag,),          self._preconditioner_naive, self._make_guess_naive),
            'davBO':  (lambda: (self.diag,),          self._preconditioner_naive, self._make_guess_davBO),
            'davBOpc':(lambda: (self.diag,),          self._preconditioner_davBOpc, self._make_guess_davBO),
            'davBOs': (lambda: (self.diag,),          self._preconditioner_naive, self._make_guess_davBO_single),
            None:     (lambda: (self.diag,),          self._preconditioner_naive, self._make_guess_naive),
            }[args.preconditioner]

        with timer_ctx(f"Build preconditioner {args.preconditioner}"):
            self._preconditioner_data = builder()
            size = sum([x.nbytes for x in self._preconditioner_data]) / 1024**2
            print(f"Preconditioner requires {int(size)}MB.")

        # Step 4 change 1: the downstream Hx path (Vx + SOCx_full + buildDiag, all
        # already built) only needs PCmat / PCdiag, so free the large full PC tensor
        # to avoid keeping both copies.  build_Hel / Vx_BO (the BO preconditioner,
        # davBO* guesses, --bo_spectrum, --davBOspec) still require PC, so keep it
        # for those paths only.
        need_pc = (args.preconditioner in ('BO', 'davBO', 'davBOpc', 'davBOs')
                   or bool(getattr(args, 'bo_spectrum', None))
                   or bool(getattr(args, 'davBOspec', False)))
        if not need_pc and self.PC is not None:
            self.PC = None
            gc.collect()
            if xp.backend == 'cupy':
                import cupy
                cupy.cuda.Stream.null.synchronize()
                cupy.get_default_memory_pool().free_all_blocks()
                cupy.get_default_pinned_memory_pool().free_all_blocks()
            print("[step4] freed full PC tensor (kept PCmat/PCdiag).")

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
    
    def E_field(self, R, r, gamma):
        mu12 = self.mu12
        aa = self.aa
        M_1 = self.M_1
        M_2 = self.M_2

        kappa2 = r*R*xp.cos(gamma)

        r1e2 = (aa*r)**2 + (R/aa)**2*(mu12/M_1)**2 - 2*kappa2*mu12/M_1
        r2e2 = (aa*r)**2 + (R/aa)**2*(mu12/M_2)**2 + 2*kappa2*mu12/M_2

        r1e = xp.sqrt(xp.where(r1e2 < 0, 0, r1e2))
        r2e = xp.sqrt(xp.where(r2e2 < 0, 0, r2e2))

        return (self._Efunc(r1e,self.g_1), self._Efunc(r2e,self.g_2))
        # return (self._Efunc(r1e,self.g_1) *xp.sin(gamma), self._Efunc(r2e,self.g_2)*xp.sin(gamma))


    def buildVsph(self):
        # builds <jΩ|V(R,r)|j'Ω> by transforming over the ɣ and ψ
        # coordinates. (V is not a function of ψ so that part is
        # analytic.)
        Nj = len(self.j)
        ma = xp.abs(self.Om -0.5).astype(int)
        mb = xp.abs(self.Om +0.5).astype(int)

        # Precompute all the associated Legendre functions up to Nj, through order J
        # N.B. P has shape (1, Nj+1, 2J+1, ...) with the 2nd axis in order -J,..0...J
        # so the |Ω| index is in slot (self.J + m)

        # Pja corresponds to spin alpha component: P_l^(Ω-0.5) 
        # Pjb corresponds to spin beta  component: P_l^(Ω+0.5) 
        # where l is an int, we deal with l = j + sg for the sake of indexing inside the sn,sm loops
        # print(xp.assoc_legendre_p_all(
        #         Nj, int(self.J+0.5),
        #         xp.cos(self.g), norm=False).shape)
        dg = self.g[1] - self.g[0]
        Vint = dg * self.Vgrid * xp.sin(self.g)[None,None,:]
        
        kwargs = dict(optimize=True)
        if xp.backend == 'torch':
            kwargs = {}
        
        Pja = xp.assoc_legendre_p_all(
                Nj, int(self.J+0.5),
                xp.cos(self.g), norm=False)[0, :, ma]
        Pjb = xp.assoc_legendre_p_all(
                Nj, int(self.J+0.5),
                xp.cos(self.g), norm=False)[0, :, mb]
        # index with [|Ω|, j, ɣ]

        # phase magnitudes for each j, Om
        def phase(j, Om):  # eq. 31 less sign
            return xp.sqrt((2*j + 1) / 2.0 *
                             xp.factorial(j - abs(Om)) /
                             xp.factorial(j + abs(Om))
                    )

        Pjkst = xp.zeros((len(self.Om), Nj,Nj,2,2,2, self.args.Nint))
                        # Ω,j, j', sg, sg', a/b, ɣ

        for n, sn in enumerate(self.sg):
            for m, sm in enumerate(self.sg):
                l1 = (self.j + sn)[:Nj]
                signsa = xp.where((self.Om -0.5 > 0) & ((self.Om-0.5).astype(int) % 2 == 1), -1, 1)
                signsb = xp.where((self.Om +0.5 > 0) & ((self.Om+0.5).astype(int) % 2 == 1), -1, 1)
                phasesa = phase(l1, (self.Om-0.5)[:, None]) * signsa[:, None]
                phasesb = phase(l1, (self.Om+0.5)[:, None]) * signsb[:, None]
                # mask to remove j < |Ω|
                maska = l1[None, :] >= ma[:, None]
                maskb = l1[None, :] >= mb[:, None]
                # maska = self.j[None, :] >= ma[:, None]
                # maskb = self.j[None, :] >= mb[:, None]
                # Apply mask and signed phases
                if sn==-0.5:
                    Pl1a = Pja[:,:Nj,:] * (maska * phasesa)[...,None]
                    Pl1b = Pjb[:,:Nj,:] * (maskb * phasesb)[...,None]
                if sn==0.5:
                    Pl1a = Pja[:,1:,:] * (maska * phasesa)[...,None]
                    Pl1b = Pjb[:,1:,:] * (maskb * phasesb)[...,None]

                l2 = (self.j + sm)[:Nj]
                phasesa = phase(l2, (self.Om-0.5)[:, None]) * signsa[:, None]
                phasesb = phase(l2, (self.Om+0.5)[:, None]) * signsb[:, None]
                # mask to remove j < |Ω|
                maska = l2[None, :] >= ma[:, None]
                maskb = l2[None, :] >= mb[:, None]
                # maska = self.j[None, :] >= ma[:, None]
                # maskb = self.j[None, :] >= mb[:, None]
                # Apply mask and signed phases
                
                if sm==-0.5:
                    Pl2a = Pja[:,:Nj,:] * (maska * phasesa)[...,None]
                    Pl2b = Pjb[:,:Nj,:] * (maskb * phasesb)[...,None]
                if sm==0.5:
                    Pl2a = Pja[:,1:,:] * (maska * phasesa)[...,None]
                    Pl2b = Pjb[:,1:,:] * (maskb * phasesb)[...,None]

                Pjkst[:,:,:,n,m,0,:] = Pl1a[:, :, None, :] * Pl2a[:, None, :, :]
                Pjkst[:,:,:,n,m,1,:] = Pl1b[:, :, None, :] * Pl2b[:, None, :, :]

        return Vint, Pjkst
    
    def buildVspincoef(self):
        ''' Build the spinor coefficients for the potential in spherical harmonic coordinates. 
        Output shape Ca( self.Ng, self.Ng, 2,2, self.Om) and same for Cb'''
        NR, Nr, Nj, Nsg, NOm = self.shape

        Ca = xp.zeros((Nj,Nj,Nsg,Nsg,NOm))
        Cb = xp.zeros((Nj,Nj,Nsg,Nsg,NOm))

        for i,ji in enumerate(self.j):
            for k, jk in enumerate(self.j):
                for n, sn in enumerate(self.sg):
                    for m, sm in enumerate(self.sg):
                        for o, Oo in enumerate(self.Om):
                            if ji+0.5+sn < 0: continue
                            if jk+0.5+sm < 0: continue
                            if ji+0.5+sn-2*sn*Oo < 0 or jk+0.5+sm-2*sm*Oo < 0: continue
                            termA = (ji+0.5+sn-2*sn*Oo)*(jk+0.5+sm-2*sm*Oo)/(ji+0.5+sn)/(jk+0.5+sm)
                            Ca[i,k,n,m,o] = 2*sn*sm*xp.sqrt(termA)

        for i,ji in enumerate(self.j):
            for k, jk in enumerate(self.j):
                for n, sn in enumerate(self.sg):
                    for m, sm in enumerate(self.sg):
                        for o, Oo in enumerate(self.Om):

                            if ji+0.5+sn < 0: continue
                            if jk+0.5+sm < 0: continue
                            if ji+0.5+sn+2*sn*Oo < 0 or jk+0.5+sm+2*sm*Oo < 0: continue
                            termB = (ji+0.5+sn+2*sn*Oo)*(jk+0.5+sm+2*sm*Oo)/(ji+0.5+sn)/(jk+0.5+sm)
                            Cb[i,k,n,m,o] = 0.5*xp.sqrt(termB)
        Cspin = xp.stack((Ca,Cb), axis=5)
        return Cspin

    def buildVOm(self):
        ''' Clebsch-Gordon Coefficients between adjacent Ω
            √(J(J+1)-Ω(Ω±1))√(j(j+1)-Ω(Ω±1)),
            shape: Ng x NΩ x NΩ '''
        NR, Nr, Nj, Nsg, NOm = self.shape
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
        return VOm

    # allows H @ x
    def __matmul__(self, other):
        return self.Hx(other).reshape(other.shape)

    #@partial(jax.jit, static_argnums=0)
    def Hx(self, x):
        with _nvtx_range("Hamiltonian.Hx"):
            with _nvtx_range("Hamiltonian.Tx"):
                out = self.Tx(x)
            with _nvtx_range("Hamiltonian.Vx"):
                out += self.Vx(x)
            if self.args.soc == 'full':
                with _nvtx_range("Hamiltonian.SOCx_full"):
                    out += self.SOCx_full(x)
            return out

    def Hx_chunked(self, x, chunk=1):
        ''' Step 4 change 5: apply Hx to the Davidson trial block in small batches
        of `chunk` vectors and concatenate, so the per-matvec working set scales
        with `chunk` (not the full subspace block).  aop(xt) with 6 trial vectors
        OOMs at J=9.5 even after the PC precontraction; calling this with
        --matvec_batch_size 1 keeps each SOCx_full/Vx contraction tiny. '''
        with _nvtx_range("Hamiltonian.Hx_chunked"):
            xa = x.reshape((-1,) + self.shape)
            B = xa.shape[0]
            if chunk is None or chunk <= 0 or chunk >= B:
                return self.Hx(x)
            outs = []
            for i in range(0, B, chunk):
                with _nvtx_range(f"Hamiltonian.Hx_chunked batch {i // chunk}"):
                    outs.append(self.Hx(xa[i:i + chunk]))
            with _nvtx_range("Hamiltonian.Hx_chunked concatenate"):
                return xp.concatenate(outs, axis=0).reshape(x.shape)

    def Hx_BO(self,x, iR=None):
        # N.B. the BO path has no full-SOC operator, so the BO preconditioner /
        # spectrum are built without spin-orbit (Tx_BO + Vx_BO only).
        out = self.Tx_BO(x, iR=iR) + self.Vx_BO(x, iR=iR)
        return out

    def _dipole_chunked(self, xa, pot, g_chunk=None):
        ''' Step 4: memory-lean replacement for
              xp.einsum('BRrjsO, Rrg, Ojkstg -> BRrktO', xa, pot, self.PC).
        It loops over Omega and over gamma blocks, contracting through the
        matmul-friendly self.PCmat[O, g, (j,s), (k,t)] layout, so CuPy never
        materializes the full (R, r, O, j, s, k, t) temporary that OOMs at
        large J.  Output shape/dtype match the original einsum exactly.
            pot : (R, r, gamma) weight tensor (Vint for Vx, Eint for SOC dipole). '''
        B = xa.shape[0]
        NR, Nr, Nj, Nsg, NOm = self.shape
        Njs = Nj * Nsg
        Nint = self.args.Nint
        if g_chunk is None or g_chunk <= 0:
            g_chunk = getattr(self.args, 'dipole_g_chunk', None) or Nint
        kwargs = {} if xp.backend == 'torch' else dict(optimize=True)
        with _nvtx_range(f"Hamiltonian._dipole_chunked g_chunk={g_chunk}"):
            out = xp.zeros((B, NR, Nr, Nj, Nsg, NOm), dtype=xa.dtype)
            pot2 = pot.reshape(NR * Nr, Nint)                       # (N, g)
            for O in range(NOm):
                with _nvtx_range(f"Hamiltonian._dipole_chunked Omega {O}"):
                    xa_O = xa[..., O].reshape(B, NR * Nr, Njs)      # (B, N, js)
                    acc = xp.zeros((B, NR * Nr, Njs), dtype=xa.dtype)  # (B, N, kt)
                    for g0 in range(0, Nint, g_chunk):
                        gsl = slice(g0, min(g0 + g_chunk, Nint))
                        PCblk = self.PCmat[O, gsl]                  # (gc, js, kt)
                        # contract j,s then weight by pot(R,r,g) and sum over the g block
                        inner = xp.einsum('BNs, gsk -> BNgk', xa_O, PCblk, **kwargs)
                        acc += xp.einsum('BNgk, Ng -> BNk', inner, pot2[:, gsl], **kwargs)
                    out[..., O] = acc.reshape(B, NR, Nr, Nj, Nsg)
            return out

    def Vx(self,x):
        with _nvtx_range("Hamiltonian.Vx body"):
            if xp.backend == 'torch':
                xa = x.reshape((-1,) + self.shape).type(self.dtype)
            else:
                xa = x.reshape((-1,) + self.shape).astype(self.dtype)

            # Step 4 change 2: chunked PCmat contraction (was a single huge einsum
            #   xp.einsum('BRrjsO, Rrg, Ojkstg-> BRrktO', xa, self.Vint, self.PC))
            vout = self._dipole_chunked(xa, self.Vint)
            return vout.reshape(x.shape)
    
    def Vx_BO(self,x, iR=None):
        kwargs = dict(optimize=True)
        if xp.backend == 'torch':
            xa = x.reshape((-1,)+self.shape[1:]).type(self.dtype)
            kwargs = {}
        else:
            xa = x.reshape((-1,)+self.shape[1:]).astype(self.dtype)

        vout =  xp.einsum('BrjsO, rg, Ojkstg-> BrktO', xa, self.Vint[iR], self.PC, **kwargs)
        return vout.reshape(x.shape)

        # Hel += xp.einsum("rs,OP,Rrg,Ojkg->RjrOksP",
        #                  xp.eye(Nr), xp.eye(NOm),
        #                  self.Vint[Ridx], self.Pjk, **kwargs).reshape(NR, Nelec, Nelec)

    #@partial(jax.jit, static_argnums=0)
    def Tx(self, x):
        with _nvtx_range("Hamiltonian.Tx body"):
            if xp.backend == 'torch':
                xa = x.reshape((-1,) + self.shape).type(self.dtype)
                kwargs = {}
            else:
                xa = x.reshape((-1,) + self.shape).astype(self.dtype)
                kwargs = dict(optimize=True)

            ke = xp.zeros_like(xa)

            # Radial Kinetic Energy terms, easy
            ke += xp.einsum('BRrjnO, RS -> BSrjnO', xa, self.ddR2, **kwargs)  # ∂²/∂R²
            ke += xp.einsum('BRrjnO, rs -> BRsjnO', xa, self.ddr2, **kwargs)  # ∂²/∂r²

            # Angular electronic ke terms: -j(j+1)(1/r² + 1/R²)
            kej = xp.einsum('BRrjsO, j  -> BRrjsO', xa, self.j*(self.j+1), **kwargs)  # j(j+1)
            kel = xp.einsum('BRrjsO, js -> BRrjsO', xa, (self.j[:,None]+self.sg[None,:])*(self.j[:,None]+self.sg[None,:]+1), **kwargs)
            # kej = xa*self.j_grid*(self.j_grid+1) # we don't have a j_grid defined yet?
            ke -= (self.Rinv2)*kej + (self.rinv2*kel)  # -j(j+1)/R² - l(l+1)/r²


            # Angular Kinetic Energy J terms
            keJdiag  = -xa * self.J * (self.J+1)                       # -J(J+1)
            keJdiag += 2*xp.einsum('BRrjsO,O-> BRrjsO', xa, self.Om**2, **kwargs)  # -J(J+1)+2Ω²

            # Step 4 change 8: VOm is nearest-neighbour in Omega, so replace the dense
            # 'BRrjsO,jOP->BRrjsP' einsum with explicit O->O+-1 shifts using the
            # precomputed VOm_shift[j,m] = VOm[j,m+1,m] = VOm[j,m,m+1] (symmetric).
            keJoffdiag = xp.zeros_like(xa)
            NOm = self.shape[-1]
            if NOm > 1:
                cb = self.VOm_shift[None, None, None, :, None, :]   # (1,1,1,Nj,1,NOm-1)
                keJoffdiag[..., :-1] += xa[..., 1:] * cb            # P=m   gets xa[...,m+1]*c[j,m]
                keJoffdiag[..., 1:]  += xa[..., :-1] * cb           # P=m+1 gets xa[...,m]  *c[j,m]
            ke += self.Rinv2*(keJdiag + keJoffdiag)

            # mass portion of KE
            ke *= -1/(2*self.mu)
            return ke.reshape(x.shape)

    def Tx_BO(self, x, iR=None):
        if xp.backend == 'torch':
            xa = x.reshape((-1,)+self.shape[1:]).type(self.dtype)
            kwargs = {}
        else:
            xa = x.reshape((-1,)+self.shape[1:]).astype(self.dtype)
            kwargs = dict(optimize=True)

        ke = xp.zeros_like(xa)

        # Radial Kinetic Energy terms, easy
        ke += xp.einsum('BrjsO,rt->BtjsO', xa, self.ddr2, **kwargs)  # ∂²/∂r²

        # Angular electronic ke terms: -j(j+1)(1/r² + 1/R²)
        kej = xp.einsum('BrjsO, j  -> BrjsO', xa, self.j*(self.j+1), **kwargs)  # j(j+1)
        kel = xp.einsum('BrjsO, js -> BrjsO', xa, (self.j[:,None]+self.sg[None,:])*(self.j[:,None]+self.sg[None,:]+1), **kwargs)
        ke -= (self.Rinv2[iR])*kej + (self.rinv2[iR]*kel)  # -j(j+1)/R² - l(l+1)/r²

        
        # Angular Kinetic Energy J terms
        keJdiag  = -xa * self.J * (self.J+1)                       # -J(J+1)
        keJdiag += 2*xp.einsum('BrjsO,O-> BrjsO', xa, self.Om**2, **kwargs)  # -J(J+1)+2Ω²
        keJoffdiag = xp.einsum('BrjsO,jOP-> BrjsP', xa, self.VOm, **kwargs)  # √(J(J+1)-Ω(Ω±1))√(j(j+1)-Ω(Ω±1))
        ke += self.Rinv2[iR]*(keJdiag + keJoffdiag)

        # mass portion of KE
        ke *= -1/(2*self.mu)
        return ke.reshape(x.shape)

    def apply_scnab(self, x):
        ''' Applies the 'vector' part of the SOC Efield: R(s.c x ∇) '''
        with _nvtx_range("Hamiltonian.apply_scnab"):
            if xp.backend == 'torch':
                xa = x.reshape((-1,) + self.shape).type(self.dtype)
                kwargs = {}
            else:
                xa = x.reshape((-1,) + self.shape).astype(self.dtype)
                kwargs = dict(optimize=True)
            
            # Step 4 change 6: the three 1/r terms share the same einsum structure and
            # only differ by their (now precomputed, kappa-folded) coefficient tensor,
            # so combine them into a single contraction through self.Csc_r = C0k+C1k+C2k.
            td = xp.einsum('BRrjsO, R, CjsktO,    rp -> BRpktO', xa, self.R, self.C_scnab, self.ddr1, **kwargs)  # R*C*ddr1
            tr = xp.einsum('BRrjsO, R,  jsktO,     r -> BRrktO', xa, self.R, self.Csc_r,   1/self.r, **kwargs)   # R*(C0k+C1k+C2k)/r
            return  (td+tr).reshape(x.shape)

    def SOCx_full(self, x):
        ''' SOC "without the ROI" — COMPLETE (bare spin-orbit + mass polarization),
        Hermitian-symmetrized, summed over the two nuclei n:

            H_SO = soc_const · ½ Σ_n [ D_n (L + s_n c_n S) + (L + s_n c_n S) D_n ]

          L = l·s  : DIAGONAL in (j,σ,Ω) with the exact eigenvalue
                     self.ls[j,σ] = ½[ j(j+1) − l(l+1) − ¾ ],  l = j+σ   (apply_ls)
          S        : mass polarization  R×(s×∇) about each nucleus (apply_scnab, C_scnab);
                     arises because l about nucleus n = l about the Jacobi origin
                     ∓ (μ12/M_n) R×p  (the electron–nucleus offset r_ne = r ∓ (μ12/M_n)R)
          D_n      : dipole rotation ⟨jσΩ|d̂_n|kτΩ⟩, with the dg·sin(g) measure (cf. Vx)
          s_n      : −1 for nucleus 1, +1 for nucleus 2   (sign of the R offset)
          c_n = μ/M_n : mass-pol prefactor.

        These are scaled coordinates (aa=√(μ12/μ)), so the physical
        (μ12/M_n)·R_lab×p_lab becomes (μ/M_n)·R_scaled×p_scaled in terms of
        the code's self.R / self.ddr1. To run bare l·s-only SOC, set
        self.args.soc_masspol = False. '''
        if xp.backend == 'torch':
            xa = x.reshape((-1,) + self.shape).type(self.dtype)
            kwargs = {}
        else:
            xa = x.reshape((-1,) + self.shape).astype(self.dtype)
            kwargs = dict(optimize=True)
        dg = self.g[1] - self.g[0]
        sgm = dg * xp.sin(self.g)[None, None, :]
        # per-nucleus dipole d̂_n with the integration measure dg·sin(g) (as Vint does):
        Eint1 = sgm * self.E1
        Eint2 = sgm * self.E2

        def apply_ls(y):                      # diagonal l·s
            with _nvtx_range("Hamiltonian.SOCx_full apply_ls"):
                return xp.einsum('BRrjsO, js -> BRrjsO', y, self.ls, **kwargs)

        def apply_dipole(y, Eint):            # ⟨jσΩ| d̂_n |kτΩ⟩ rotation (cf. Vx)
            # Step 4 change 3: reuse the chunked Vx contraction (Eint replaces Vint),
            # so the dipole rotation never builds the full (R,r,O,j,s,k,t) temporary.
            with _nvtx_range("Hamiltonian.SOCx_full apply_dipole"):
                return self._dipole_chunked(y, Eint)

        # Step 4 change 4: compute Lx / Sx / D1x / D2x once and reuse them in the
        # symmetrized SOC expression (the old code recomputed apply_dipole(xa,·)
        # twice per nucleus and apply_scnab(xa) twice).  The math is unchanged.
        with _nvtx_range("Hamiltonian.SOCx_full Lx"):
            Lx = apply_ls(xa)                               # L xa
        if getattr(self.args, 'soc_masspol', True):
            c1, c2 = self.mu / self.M_1, self.mu / self.M_2     # mass-pol prefactor μ/M
            S = self.apply_scnab
            with _nvtx_range("Hamiltonian.SOCx_full Sx"):
                Sx = S(xa)                                  # S xa
            with _nvtx_range("Hamiltonian.SOCx_full D1x"):
                D1x = apply_dipole(xa, Eint1)               # D_1 xa
            with _nvtx_range("Hamiltonian.SOCx_full D2x"):
                D2x = apply_dipole(xa, Eint2)               # D_2 xa
            # per nucleus: ½(D_n A_n + A_n D_n),  A_1 = L − c1 S,  A_2 = L + c2 S
            with _nvtx_range("Hamiltonian.SOCx_full D1_A1"):
                out  = apply_dipole(Lx - c1 * Sx, Eint1)
            with _nvtx_range("Hamiltonian.SOCx_full D2_A2"):
                out += apply_dipole(Lx + c2 * Sx, Eint2)
            with _nvtx_range("Hamiltonian.SOCx_full A1_D1"):
                out += apply_ls(D1x) - c1 * S(D1x)
            with _nvtx_range("Hamiltonian.SOCx_full A2_D2"):
                out += apply_ls(D2x) + c2 * S(D2x)
            with _nvtx_range("Hamiltonian.SOCx_full scale"):
                out *= 0.5
        else:                                  # bare l·s only (no mass polarization)
            with _nvtx_range("Hamiltonian.SOCx_full bare"):
                out = 0.5 * (apply_dipole(Lx, Eint1 + Eint2)
                             + apply_ls(apply_dipole(xa, Eint1 + Eint2)))
        with _nvtx_range("Hamiltonian.SOCx_full output"):
            return self.soc_const * out.reshape(x.shape)


    def buildC_scnab(self):
        '''builds array with coefs out the front of s.c nab terms'''
        coef0 = (self.sg[None,:,None]*self.Om[None,None,:]*(2*self.j[:,None,None]+1)
                /(self.j[:,None,None]*(self.j[:,None,None]+1)))
        sigx = xp.array([[0,1],[1,0]]) # send sigma --> -sigma
        C0 = xp.einsum('jso, st, jk -> jskto',coef0, sigx, xp.eye(self.shape[2]))

        C1 = xp.zeros(C0.shape)
        C2 = xp.zeros(C0.shape)
        for i,ji in enumerate(self.j):
            for k, jk in enumerate(self.j):
                for n, sn in enumerate(self.sg):
                    for m, _ in enumerate(self.sg):
                        for o, Oo in enumerate(self.Om):
                            if n != m: continue
                            kappa = sn*(2*ji+1)
                            kappai = sn*(2*ji+1)
                            kappak = sn*(2*jk+1)
                            if   kappai == kappak-1:
                                C1[i,n,k,m,o] =    xp.sqrt(xp.maximum((kappa+0.5)**2-Oo**2, 0.0))/(xp.abs(2*kappa +1))
                            elif kappai == kappak+1:
                                C2[i,n,k,m,o] = -1*xp.sqrt(xp.maximum((kappa-0.5)**2-Oo**2, 0.0))/(xp.abs(2*kappa -1))

        return xp.stack((C0,C1,C2), axis=0, dtype=self.dtype)/2
    
    def build_sz_lz(self):
        _, _, Nj, Nsg, NOm = self.shape
        Sz = xp.zeros((Nj,Nsg,Nj,Nsg,NOm)) # jsktO ordering
        Lz = xp.zeros(Sz.shape)

        for j,ji in enumerate(self.j):
            for k, jk in enumerate(self.j):
                for s, sn in enumerate(self.sg):
                    for t, sm in enumerate(self.sg):
                        for o, Oo in enumerate(self.Om):
                            kappaj = sn*(2*ji+1)
                            kappak = sm*(2*jk+1)
                            if kappak==kappaj: # diagonal term
                                # N.B. s_z (not sigma_z): the factor was -2*Oo/.. which is
                                # sigma_z = 2 s_z (a latent 2x error in <s_z>).  Halved so that
                                # s_z has eigenvalues +/-1/2 and l_z + s_z = Omega = j_z exactly.
                                Sz[j,s,k,t,o] = -Oo/(2*kappaj+1)
                                Lz[j,s,k,t,o] = 2*Oo*(kappaj+1)/(2*kappaj+1)
                            if kappak== -kappaj-1: # spin flip term
                                scoef = xp.sqrt(xp.maximum((kappaj+0.5)**2-Oo**2, 0.0))/(xp.abs(2*kappaj+1))
                                Sz[j,s,k,t,o] = -scoef   # was -2*scoef (sigma_z); see note above
                                Lz[j,s,k,t,o] = scoef
        # with xp.printoptions(precision=3, linewidth=xp.inf, suppress=True):
        #     print(Lz[:,:,:,:,0].reshape((Nj*Nsg,Nj*Nsg)))
        #     print()
        #     print(Lz[:,:,:,:,1].reshape((Nj*Nsg,Nj*Nsg)))
        #     exit()
        return Sz, Lz

    def build_spin_ops(self):
        ''' Body-frame electron-spin operators s_x, s_y, s_z in the (j, sigma, Omega)
        spin-angular basis, built from explicit Clebsch-Gordan coefficients.  With
        l = j + sigma the basis function is the spinor spherical harmonic

            |j,sigma,Omega> = c_up |l, Omega-1/2> |up> + c_dn |l, Omega+1/2> |dn>,
            sigma = -1/2 (j=l+1/2):  c_up = +sqrt((l+Omega+1/2)/(2l+1)),
                                     c_dn = +sqrt((l-Omega+1/2)/(2l+1))
            sigma = +1/2 (j=l-1/2):  c_up = -sqrt((l-Omega+1/2)/(2l+1)),
                                     c_dn = +sqrt((l+Omega+1/2)/(2l+1))

        s_z is diagonal in Omega; s_x = (s_+ + s_-)/2 and s_y = (s_+ - s_-)/(2i)
        flip the spin and so connect Omega <-> Omega' = Omega +/- 1 (they preserve l,
        hence couple the two j = l +/- 1/2 partners).  s_y is pure imaginary.

        Returns s_x (real), s_y (complex), each (Nj,Nsg,Nj,Nsg,NOm,NOm); index order
        is (j,sigma | k,tau | Omega,Omega').  Validated against the su(2) algebra
        (s^2=3/4, [s_i,s_j]=i eps_ijk s_k), against l.s = self.ls, and against
        l_z + s_z = Omega.  See S_3D/spin_expectations.md. '''
        _, _, Nj, Nsg, NOm = self.shape
        s_x = xp.zeros((Nj, Nsg, Nj, Nsg, NOm, NOm))
        s_y = xp.zeros((Nj, Nsg, Nj, Nsg, NOm, NOm), dtype=xp.complex128)

        def cg(j, sg, Om):              # (c_up, c_dn) for |j,sg,Om>,  l = j+sg
            l = j + sg
            c_up = c_dn = 0.0
            if abs(Om - 0.5) <= l:
                c_up = (xp.sqrt((l+Om+0.5)/(2*l+1)) if sg == -0.5
                        else -xp.sqrt((l-Om+0.5)/(2*l+1)))
            if abs(Om + 0.5) <= l:
                c_dn = (xp.sqrt((l-Om+0.5)/(2*l+1)) if sg == -0.5
                        else  xp.sqrt((l+Om+0.5)/(2*l+1)))
            return c_up, c_dn

        for j, jb in enumerate(self.j):
            for s, sgb in enumerate(self.sg):
                lb = jb + sgb
                for k, jk in enumerate(self.j):
                    for t, tt in enumerate(self.sg):
                        if abs((jk + tt) - lb) > 1e-9:   # spin ops preserve l
                            continue
                        for o, Oo in enumerate(self.Om):
                            cu_b, cd_b = cg(jb, sgb, Oo)
                            for p, Op in enumerate(self.Om):
                                cu_k, cd_k = cg(jk, tt, Op)
                                if abs(Oo - (Op + 1)) < 1e-9:   # <up_bra| . |dn_ket>
                                    s_x[j, s, k, t, o, p] += 0.5  * cu_b * cd_k
                                    s_y[j, s, k, t, o, p] += -0.5j * cu_b * cd_k
                                if abs(Oo - (Op - 1)) < 1e-9:   # <dn_bra| . |up_ket>
                                    s_x[j, s, k, t, o, p] += 0.5  * cd_b * cu_k
                                    s_y[j, s, k, t, o, p] += 0.5j * cd_b * cu_k
        return s_x, s_y

    def apply_Sx(self, x):
        ''' Apply the body-frame spin operator s_x (couples Omega <-> Omega +/- 1).
        Preserves the input dtype (a complex x stays complex). '''
        xa = x.reshape((-1,) + self.shape)
        kwargs = {} if xp.backend == 'torch' else dict(optimize=True)
        return xp.einsum('BRrjsO, jsktOP -> BRrktP', xa, self.s_x, **kwargs).reshape(x.shape)

    def apply_Sy(self, x):
        ''' Apply the body-frame spin operator s_y (pure imaginary; couples
        Omega <-> Omega +/- 1).  Returns a COMPLEX array.  For the real eigenvectors
        of the real (time-reversal-symmetric) Hamiltonian <psi|s_y|psi> = 0
        identically; a non-zero value requires a complex wavefunction / a complex
        combination within a Kramers doublet. '''
        xa = x.reshape((-1,) + self.shape)
        if xp.backend == 'torch':
            return xp.einsum('BRrjsO, jsktOP -> BRrktP', xa.type(self.s_y.dtype),
                             self.s_y).reshape(x.shape)
        return xp.einsum('BRrjsO, jsktOP -> BRrktP', xa.astype(self.s_y.dtype), self.s_y,
                         optimize=True).reshape(x.shape)

    def apply_Sz(self,x):
        # preserve input dtype (a complex x stays complex; needed for <s_z> on
        # complex combinations within a Kramers doublet)
        xa = x.reshape((-1,) + self.shape)
        kwargs = {} if xp.backend == 'torch' else dict(optimize=True)
        return xp.einsum('BRrjsO, jsktO -> BRrktO', xa, self.s_z, **kwargs).reshape(x.shape)
    def apply_Lz(self,x):
        if xp.backend == 'torch':
            xa = x.reshape((-1,) + self.shape).type(self.dtype)
            kwargs = {}
        else:
            xa = x.reshape((-1,) + self.shape).astype(self.dtype)
            kwargs = dict(optimize=True)
        
        return xp.einsum('BRrjsO, jsktO -> BRrktO', xa, self.l_z, **kwargs).reshape(x.shape)

    # N.B. This section *must* be kept in sync with Hx above
    def buildDiag(self):
        diag = xp.zeros(self.shape, dtype=xp.float64) # DIAG IS REAL!
        ke  = xp.zeros(self.shape)
        ke += xp.diag(self.ddR2)[:, None, None, None, None] # ∂²/∂R²
        ke += xp.diag(self.ddr2)[None, :, None, None, None] # ∂²/∂r²
        # l(l+1)/r²
        ke -= (self.rinv2) * ((self.j[:,None] + self.sg[None,:])*(self.j[:,None]+self.sg[None,:]+1))[None, None, :, :, None]
        ke -= (self.Rinv2)*(self.j*(self.j+1))[None,None,:,None,None] # j(j+1)/R²
        # Angular Kinetic Energy J terms
        if self.J != 0: # (J(J+1)-2Ω²)/R²
            ke += self.Rinv2 * ( 2*self.Om**2
                -self.J*(self.J+1) )[None,None,None,None,:]

        # mass portion of KE
        ke *= -1 / (2*self.mu)

        kwargs = dict(optimize=True)
        # Step 4 change 7: contract against the diagonal slice PCdiag[O,j,s,g]
        # (= PC[O,j,j,s,s,g]) instead of the full PC tensor.
        # Vdiag = xp.einsum('Rrg,Ojjg->RrjO', self.Vint, self.Pjk)
        Vdiag = xp.einsum('Rrg, Ojsg-> RrjsO', self.Vint, self.PCdiag, **kwargs)

        # Potential terms
        diag = Vdiag + ke

        if self.args.soc=='full':
            # diagonal of SOCx_full = soc_const·½ Σ_n [D_n(L + s_n c_n S) + h.c.].
            # Must mirror SOCx_full exactly.
            #  bare l·s:   soc_const · ls[j,σ] · ⟨jσΩ|d̂|jσΩ⟩   (both nuclei, w/ measure)
            #  mass pol:   soc_const · s_n c_n · diag(D_n S);  only the 1/r pieces of S
            #              (t0,t1,t2) survive on the diagonal (the ∂r piece is r-off-diag).
            dg = self.g[1] - self.g[0]
            sgm = dg * xp.sin(self.g)[None, None, :]
            Eint1 = sgm * self.E1; Eint2 = sgm * self.E2
            diag += self.soc_const * xp.einsum('js, Rrg, Ojsg -> RrjsO',
                        self.ls, Eint1 + Eint2, self.PCdiag, **kwargs)
            if getattr(self.args, 'soc_masspol', True):
                def scn_diag(Eint):   # diag(D_n ∘ scnab), 1/r pieces of scnab
                    # Step 4 change 6: use the precomputed kappa-folded C0k/C1k/C2k
                    # (off-diagonal in (j,k),(s,t), so full PC -- not PCdiag -- is needed).
                    t0 = xp.einsum('R, jsktO, r, Rrg, Okjtsg -> RrjsO',
                            self.R, self.C0k, 1/self.r, Eint, self.PC, **kwargs)
                    t1 = xp.einsum('R, jsktO, r, Rrg, Okjtsg -> RrjsO',
                            self.R, self.C1k, 1/self.r, Eint, self.PC, **kwargs)
                    t2 = xp.einsum('R, jsktO, r, Rrg, Okjtsg -> RrjsO',
                            self.R, self.C2k, 1/self.r, Eint, self.PC, **kwargs)
                    return t0 + t1 + t2
                diag += self.soc_const * (-self.mu/self.M_1 * scn_diag(Eint1)
                                          + self.mu/self.M_2 * scn_diag(Eint2))

        return diag.ravel()

    def _make_guess_naive(self, min_guess):
        # Step 4 change 7 (caveat): this Vdiag sums the potential over ALL input
        # (j,s) -- it is Vx applied to an all-ones angular vector, NOT the (j==k,
        # s==t) diagonal -- so PCdiag is mathematically WRONG here.  Use the chunked
        # _dipole_chunked helper (PCmat) instead, which reproduces the old full
        # 'Rrg,Ojkstg->RrktO' einsum exactly while staying memory-lean and avoiding
        # self.PC (freed after build on the naive path).
        ones = xp.ones((1,) + self.shape, dtype=self.dtype)
        Vdiag = self._dipole_chunked(ones, self.Vint)[0]    # (R, r, k, t, O)
        # self.shape: R, r, j, s, Om
        # Vdiag = xp.einsum('Rrg,Ojjg->RrjO', self.Vint, self.Pjk)
        g = xp.exp(-(Vdiag - xp.min(Vdiag))**2/27.211**2)

        *_, Nsg, NOm = self.shape;
        mask = xp.eye(NOm, dtype=g.dtype).reshape(NOm, 1, 1, 1, 1, NOm)  # shape (NOm, 1, 1, 1, NOm)
        # mask = xp.kron(xp.eye(Nsg, dtype=g.dtype),xp.eye(NOm, dtype=g.dtype)).reshape(NOm*Nsg, 1, 1, 1, Nsg, NOm)  # shape (NOm, 1, 1, 1, NOm)
        # guesses = (mask * g).reshape(NOm*Nsg, -1)
        guesses = (mask * g).reshape(NOm, -1)
        print("guesses", guesses.shape, self.size, self.shape)
        return guesses

    #@partial(jax.jit, static_argnums=0)
    def _preconditioner_naive(self, dx, e, x0):
        diagd = self.diag - (e - 1e-5)
        return dx/diagd

    def BO_spectrum(self, nroots=0, Hel_func=None):
        print("Building BO spectrum")
        NR, Nr, Ng, Nsg, NOm = self.shape
        Nelec = Nr*Ng*Nsg*NOm

        if Hel_func is None:
            Hel_func = self.build_Hel

        mem_thresh = 1e5
        memory_constrained = self.size > mem_thresh

        print(f"memory constraint threshold = {mem_thresh}, {memory_constrained}")

        if self.args.davBOspec:
            min_guess = nroots
            U_n   = xp.zeros((NR, Nelec, min_guess), dtype=self.dtype)
            Ad_n  = xp.zeros((NR, min_guess))

            guess = xp.random.random((min_guess, Nelec))
            max_memory = get_davidson_mem(0.75)

            with timer_ctx(f"DAVBO: {NR} Davidson of size {Nelec}"):
                for i, R in enumerate(self.R):
                    conv, Ad_n[i], guess = lib.davidson1(
                        partial(self.Hx_BO, iR=i),
                        guess,
                        xp.reshape(self.diag, self.shape)[i].ravel(),
                        #callback=reporter,
                        nroots=min_guess,
                        max_cycle=500,
                        verbose=1,
                        max_space=1000,
                        max_memory=max_memory,
                        tol=1e-12,
                    )

                    if not all(conv):
                        raise Warning("All roots not converged")
                    print(Ad_n[i])
                    U_n[i] = guess.T
        else:
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

        if self.args.davBOspec:
            Hbo = xp.empty((nroots,NR,NR))
        else:
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
        kwargs = dict(optimize=True)
        if xp.backend == 'torch':
            kwargs = {}
        NR, Nr, Nj, Nsg, NOm = self.shape
        Nsph = Nj * Nsg * NOm 
        Nl = Nj * Nsg
        Nelec = Nr * Nsph

        if Ridx is None:
            Ridx = xp.arange(NR)
        else:
            Ridx = xp.atleast_1d(Ridx)
            NR,  = Ridx.shape

        def kron3(Or, Ol, OO):
            return xp.kron(Or, xp.kron(Ol, OO))
        
        def kron4(Or, Oj, Os, OO):
            return xp.kron(Or, xp.kron(Oj, xp.kron(Os,OO)))
        

        # Hel = -1/2/μ · (Te + VOm) + V
        # Te  =  ∂²/∂r² - (1/r²)l(l+1) - (1/R²)(j(j+1) + J(J+1) - 2Ω²)
        # VOm = (1/R²)√(J(J+1) - Ω(Ω ± 1))√(j(j+1) - Ω(Ω ± 1)) ; (1/R²)*self.VOm
        # N.B. self.ddr2 = ∂²/∂r² + 1/4/r²
        Hel = xp.empty((NR, Nelec, Nelec), dtype=self.dtype)

        # build *bare* Te first
        # R-independent terms: ∂²/∂r² - (1/r²)j(j+1)
        l = (self.j[:,None]+self.sg[None,:]).ravel() #shape Nl

        Hel[:] = (
            xp.kron(self.ddr2, xp.eye(Nsph)) -   # ∂²/∂r²
            kron3(xp.diag(1 / self.r**2),        # -(1/r²)l(l+1)
                  xp.diag(l*(l+1)),
                  xp.eye(NOm))
        )


        # R-dependent terms: (1/R²)j(j+1)
        Rinv2 = (1 / self.R**2)[Ridx, None, None]  # (1/R²), ready for broadcasting
        Hel -= Rinv2 * kron4(xp.eye(Nr),           # -(1/R²) * j(j+1)
                             xp.diag(self.j*(self.j+1)), xp.eye(Nsg),
                             xp.eye(NOm))[None]

        # J terms: -(1/R²)J(J+1) + 2Ω²/R²
        Hel[:, xp.arange(Nelec), xp.arange(Nelec)] -= (
            Rinv2[:,:,0] * self.J * (self.J+1)  # -(1/R²) J(J+1)
        )
        Hel += 2 * kron4(xp.eye(Nr), xp.eye(Nj), xp.eye(Nsg), xp.diag(self.Om**2)) * Rinv2 # + 2Ω²/R²

        # VOm term:

        VOm_big = xp.einsum('jOP,ij,st->isOjtP', self.VOm, xp.eye(Nj), xp.eye(Nsg), **kwargs).reshape(Nsph, Nsph)

        Hel += xp.kron(xp.eye(Nr), VOm_big) * Rinv2
        Hel *= -1 / (2 * self.mu)  # -1/2/μ · (Te + VOm)

        # N.B. While one might be tempted to write the output as
        # RrsjkOP, recall that when we reshape, we need to make sure
        # that we have Rx(Nelec)x(Nelec) => Rx(rjO)x(skP). This
        # repeats the ordering of the indices that matches kron3.
        def build_potential(pot):
            return xp.einsum("rs,OP,Rrg,Ojkabg ->RrjaOskbP",
                 xp.eye(Nr), xp.eye(NOm), pot[Ridx], self.PC, **kwargs).reshape(NR, Nelec, Nelec)
    
        Hel += build_potential(self.Vint)
        
        if self.args.soc=='full':
            # H_SO = soc_const·½ Σ_n (D_n A_n + A_n D_n),  A_n = L + s_n c_n S.
            # D_n = per-nucleus dipole rotation (dg·sin(g) measure), L = diag(self.ls),
            # S = scnab (mass pol), c_n = μ/M_n.  Mirrors SOCx_full / buildDiag.
            dg = self.g[1] - self.g[0]
            sgm = dg * xp.sin(self.g)[None, None, :]
            D1mat = build_potential(sgm * self.E1)
            D2mat = build_potential(sgm * self.E2)
            ls_mat = xp.einsum('js, R, rp, jk, st, OP -> RrjsOpktP', self.ls, xp.ones(NR)[Ridx],
                            xp.eye(Nr), xp.eye(Nj), xp.eye(Nsg), xp.eye(NOm), **kwargs).reshape(NR, Nelec, Nelec)
            A1 = ls_mat; A2 = ls_mat
            if getattr(self.args, 'soc_masspol', True):
                kappa = self.sg[None, :] * (2 * self.j[:, None] + 1)
                scnab_mat = ( xp.einsum('CjsktO, rp, R, OP -> RrjsOpktP', self.C_scnab, self.ddr1,
                                    self.R[Ridx], xp.eye(NOm), **kwargs)
                            + xp.einsum('jsktO, js, rp, R, OP -> RrjsOpktP', self.C_scnab[0], kappa,
                                    xp.diag(1/self.r), self.R[Ridx], xp.eye(NOm), **kwargs)
                            + xp.einsum('jsktO, kt, rp, R, OP -> RrjsOpktP', self.C_scnab[1], -kappa,
                                    xp.diag(1/self.r), self.R[Ridx], xp.eye(NOm), **kwargs)
                            + xp.einsum('jsktO, js, rp, R, OP -> RrjsOpktP', self.C_scnab[2], kappa,
                                    xp.diag(1/self.r), self.R[Ridx], xp.eye(NOm), **kwargs)).reshape(NR, Nelec, Nelec)
                A1 = ls_mat - (self.mu/self.M_1) * scnab_mat
                A2 = ls_mat + (self.mu/self.M_2) * scnab_mat
            Hsoc = D1mat @ A1 + D2mat @ A2
            Hel += self.soc_const * 0.5 * (Hsoc + Hsoc.transpose((0, 2, 1)))


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
        NR, Nr, Nj, Nsg, NOm = self.shape
        Nelec = Nr*Nj*Nsg*NOm

        dx_ = dx.reshape((-1, NR, Nelec))

        kwargs = dict(optimize=True)
        if xp.backend == 'torch':
            kwargs = {}

        tr_ = xp.einsum(
            'Rij,jRq,qj,jmq,mpj,Bmp->BRi',
            U_n, U_v, 1.0 / diagd, U_v, U_n, dx_, **kwargs
        )

        return tr_.reshape(dx.shape)

    def ensure_davBOpc_data(self, min_guess):
        if len(self._preconditioner_data) >= 5:
            return self._preconditioner_data[1:]

        Ad_n, U_n, Ad_vn, U_v = self._build_davBO_lowrank_data(min_guess)
        object.__setattr__(
            self,
            '_preconditioner_data',
            (self.diag, U_n, U_v, Ad_vn, Ad_n),
        )
        size = sum(x.nbytes for x in self._preconditioner_data) / 1024**2
        print(f"[davBOpc] Low-rank preconditioner data requires {int(size)}MB.")
        return U_n, U_v, Ad_vn, Ad_n

    def _build_davBO_lowrank_data(self, min_guess):
        NR, Nr, Nj, Nsg, NOm = self.shape
        Nelec = Nr*Nj*Nsg*NOm

        U_n   = xp.zeros((NR, Nelec, min_guess), dtype=self.dtype)
        Ad_n  = xp.zeros((NR, min_guess))

        guess = xp.random.random((min_guess, Nelec))
        max_memory = get_davidson_mem(0.75)

        with timer_ctx(f"DAVBO: {NR} Davidson of size {Nelec}"):
            for i, R in enumerate(self.R):
                conv, Ad_n[i], guess = lib.davidson1(
                    partial(self.Hx_BO, iR=i),
                    guess,
                    xp.reshape(self.diag, self.shape)[i].ravel(),
                    #callback=reporter,
                    nroots=min_guess,
                    max_cycle=self.args.davbo_iterations,
                    verbose=self.args.davbo_verbosity,
                    max_space=self.args.davbo_subspace,
                    max_memory=max_memory,
                    tol=self.args.davbo_tol,
                )

                if not all(conv):        
                    print(f"WARNING: davBO fixed-R solve did not converge at iR={i}, R_lab={self.R_lab[i]}")
                print(self.R_lab[i], Ad_n[i])
                U_n[i] = guess.T

        phase_match(U_n)

        with timer_ctx("Build Hbo"):
            Hbo = xp.empty((min_guess, NR, NR))                # Hbo = -1/2/μ(∂²/∂R² + 1/4/R²) + V_n
            Hbo[:] = -1 / 2 / self.mu * self.ddR2          #       -1/2/μ(∂²/∂R² + 1/4/R²)
            Hbo[:, xp.arange(NR), xp.arange(NR)] += Ad_n.T # V_n

        with timer_ctx("Diag  Hbo"):
            Ad_vn, U_v = xp.linalg.eigh(Hbo)  # xp.linalg.eigh(Hbo)
            Ad_vn = Ad_vn.T

        with timer_ctx("Phase match U_v"):
            phase_match(U_v)

        return Ad_n, U_n, Ad_vn, U_v

    def _make_guess_davBO(self, min_guess):
        if self.args.preconditioner == 'davBOpc':
            U_n, U_v, Ad_vn, _ = self.ensure_davBOpc_data(min_guess)
        else:
            _, U_n, Ad_vn, U_v = self._build_davBO_lowrank_data(min_guess)

        s = int(numpy.ceil(numpy.sqrt(min_guess)))
        guesses = xp.stack([
            (U_n[:,:,n] * U_v[n,:,v,xp.newaxis]).ravel()
            for n in range(s) for v in range(s)
        ])

        return guesses

    def _preconditioner_davBOpc(self, dx, e, x0):
        if len(self._preconditioner_data) < 5:
            return self._preconditioner_naive(dx, e, x0)

        diag, U_n, U_v, Ad_vn, _ = self._preconditioner_data
        NR, Nr, Nj, Nsg, NOm = self.shape
        Nelec = Nr*Nj*Nsg*NOm
        dx_ = dx.reshape((-1, NR, Nelec))

        kwargs = dict(optimize=True)
        if xp.backend == 'torch':
            kwargs = {}

        # Project residuals onto the retained fixed-R electronic BO channels.
        proj = xp.einsum('Rin,BRi->BRn', U_n.conj(), dx_, **kwargs)
        dx_low = xp.einsum('Rin,BRn->BRi', U_n, proj, **kwargs)
        dx_perp = dx_ - dx_low

        # Invert the reduced nuclear BO Hamiltonian in the vibrational eigenbasis.
        coeff = xp.einsum('nRv,BRn->Bvn', U_v.conj(), proj, **kwargs)
        level_shift = getattr(self.args, 'davbopc_level_shift', 1e-5)
        denom_floor = getattr(self.args, 'davbopc_denom_floor', 1e-8)
        denom = Ad_vn - (e - level_shift)
        denom = xp.where(
            xp.abs(denom) < denom_floor,
            xp.where(denom >= 0, denom_floor, -denom_floor),
            denom,
        )
        coeff = coeff / denom[None, :, :]
        low = xp.einsum('nRv,Bvn->BRn', U_v, coeff, **kwargs)
        low = xp.einsum('Rin,BRn->BRi', U_n, low, **kwargs)

        diagd = diag - (e - level_shift)
        jacobi_full = dx / diagd
        jacobi_perp = dx_perp.reshape(dx.shape) / diagd
        structured = low.reshape(dx.shape) + jacobi_perp
        weight = getattr(self.args, 'davbopc_weight', 1.0)
        return (1.0 - weight) * jacobi_full + weight * structured

    def release_full_pc(self, label=""):
        """Free full PC after davBO/davBOs guess generation.

        The production Step 4 matvec uses PCmat/PCdiag.  davBO needs full PC only
        while building fixed-R BO guesses through Hx_BO/Vx_BO.
        """
        if self.PC is None:
            return
        object.__setattr__(self, 'PC', None)
        gc.collect()
        if xp.backend == 'cupy':
            import cupy
            cupy.cuda.Stream.null.synchronize()
            cupy.get_default_memory_pool().free_all_blocks()
            cupy.get_default_pinned_memory_pool().free_all_blocks()
        suffix = f" {label}" if label else ""
        print(f"[step4] freed full PC tensor{suffix} (kept PCmat/PCdiag).")


    def _make_guess_davBO_single(self, min_guess):
        iR = xp.unravel_index(xp.argmin(self.Vgrid), self.Vgrid.shape)[0]
        print("DavBO guess evaluated at iR,R_lab = ", iR, self.R_lab[iR])
        diag = xp.reshape(self.diag, self.shape)[iR]

        with timer_ctx(f"DAVBO: Davidson of size {diag.size}"):
            conv, e_approx, evecs = lib.davidson1(
                partial(self.Hx_BO, iR=iR),
                xp.random.random(diag.size),
                diag.ravel(),
                nroots=self.args.k,
                max_cycle=self.args.davbo_iterations,
                verbose=self.args.davbo_verbosity,
                max_space=self.args.davbo_subspace,
                max_memory=get_davidson_mem(0.75),
                tol=self.args.davbo_tol,
            )
        sR = (self.R[-1]-self.R[0])/4
        nuc_wfc = xp.exp(-(self.R-self.R[iR])**2/sR**2)
        hermite = xp.stack([xp.ones(self.R.size),
                   self.R-self.R[iR],
                   (self.R-self.R[iR])**2 -1,
                   (self.R-self.R[iR])**3 -3*self.R-self.R[iR]])

        guess = (xp.reshape(evecs[0], diag.shape)[None,None,:,:,:,:]*nuc_wfc[None,:,None,None,None,None]
                                                *hermite[:,:,None,None,None,None])
        print("guess shape", guess.shape, guess.size, self.shape, self.size, nuc_wfc.shape)

        return xp.reshape(guess, (hermite.shape[0],-1))


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
        prog='3body-3D',
        description="computes the lowest k eigenvalues of a 3-body potential in 3D")

    class ArrayAction(ap.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            setattr(namespace, self.dest, xp.array(values, dtype=float))

    parser.add_argument('-k', metavar='num_eigenvalues', default=5, type=int)
    parser.add_argument('-t', metavar="num_threads", default=16, type=int)
    parser.add_argument('-g_1', metavar='g_1', required=True, type=float)
    parser.add_argument('-g_2', metavar='g_2', required=True, type=float)
    parser.add_argument('-M_1', required=True, type=float)
    parser.add_argument('-M_2', required=True, type=float)
    parser.add_argument('-J', default=0, type=float)
    parser.add_argument('-R', dest="NR", metavar="NR", default=80, type=int)
    parser.add_argument('-r', dest="Nr", metavar="Nr", default=80, type=int)
    parser.add_argument('-g', dest="Ng", metavar="Ng", default=80, type=int)
    parser.add_argument('-int', dest="Nint", metavar="number of int points for Vjj'Om", default=None, type=int)
    parser.add_argument('--potential', choices=['soft_coulomb', 'borgis', 'erf_coulomb'],
                        default='borgis')
    parser.add_argument('--extent', metavar="X", action=ArrayAction,
                        nargs=3, help="Rmin Rmax rmax, in Bohr "
                        "(typically set automatically)")
    parser.add_argument('--no_soc_masspol', dest='soc_masspol', action='store_false',
                        help="for --soc full: drop the mass-polarization (scnab) term, "
                             "keeping only the bare l.s (default: mass pol ON)")
    parser.set_defaults(soc_masspol=True)
    # NOTE: this clean build only implements the 'full' spin-orbit operator.
    parser.add_argument('--bo_spectrum', metavar='spec.npz', type=Path, default=None)
    parser.add_argument('--preconditioner', choices=['naive', 'BO', 'davBO', 'davBOpc', 'davBOs'],
                        default="naive", type=str)
    parser.add_argument('--davbo_iterations', type=int, default=1000,
                        help="for --preconditioner davBO/davBOpc/davBOs: max cycles for each fixed-R BO Davidson")
    parser.add_argument('--davbo_subspace', type=int, default=1000,
                        help="for --preconditioner davBO/davBOpc/davBOs: max subspace for each fixed-R BO Davidson")
    parser.add_argument('--davbo_tol', type=float, default=1e-10,
                        help="for --preconditioner davBO/davBOpc/davBOs: tolerance for each fixed-R BO Davidson")
    parser.add_argument('--davbo_verbosity', type=int, default=3,
                        help="for --preconditioner davBO/davBOpc/davBOs: verbosity for each fixed-R BO Davidson")
    parser.add_argument('--davbopc_weight', type=float, default=1.0,
                        help="for --preconditioner davBOpc: blend weight; 0=naive Jacobi, 1=BO-channel inverse plus Jacobi complement")
    parser.add_argument('--davbopc_level_shift', type=float, default=1e-5,
                        help="for --preconditioner davBOpc: denominator level shift")
    parser.add_argument('--davbopc_denom_floor', type=float, default=1e-8,
                        help="for --preconditioner davBOpc: minimum absolute denominator")
    parser.add_argument('--verbosity', default=2, type=int)
    parser.add_argument('--backend', default='numpy')
    parser.add_argument('--iterations', metavar='max_iterations', default=10000, type=int)
    parser.add_argument('--subspace', metavar='max_subspace', default=1000, type=int)
    parser.add_argument('--davidson_tol', type=float, default=1e-10,
                        help="main Davidson energy convergence tolerance")
    parser.add_argument('--davidson_residual_tol', type=float, default=None,
                        help="main Davidson residual tolerance; default is sqrt(--davidson_tol)")
    parser.add_argument('--lock', action='store_true',
                        help="enable Davidson root locking/deflation: freeze a root out of "
                             "the active working set (and stop rebuilding it from scratch on "
                             "every --subspace restart) once its eigenvalue has converged and "
                             "its residual is within --lock_tol_factor of --davidson_tol's "
                             "residual bound. Default off (identical to the unlocked solver).")
    parser.add_argument('--lock_tol_factor', type=float, default=10.,
                        help="how much looser than the residual convergence bound a root's "
                             "residual may be before --lock freezes it (default 10x)")
    parser.add_argument('--guess', metavar="guess.npz", type=Path, default=None)
    parser.add_argument('--evecs', metavar="guess.npz", type=Path, default=None)
    parser.add_argument('--checkpoint_every', type=int, default=0,
                        help="if >0 and --evecs is set, atomically snapshot the "
                             "Davidson Ritz vectors to <evecs>.ckpt.npz every N "
                             "cycles (crash/wall-time insurance; 0 = off)")
    parser.add_argument('--soc', metavar="SOC type:None/full", choices=['None','full'], type=str, default=None)
    parser.add_argument('--alpha', metavar="SOC enhancement", type=float, default=1.)
    parser.add_argument('--save', metavar="filename")
    parser.add_argument('--davBOspec', action='store_true')
    parser.add_argument('--matvec_batch_size', type=int, default=None,
                        help="Step 4: apply Hx to Davidson trial vectors in batches of "
                             "this size (Hx_chunked); 1 minimizes per-matvec GPU memory")
    parser.add_argument('--dipole_g_chunk', type=int, default=64,
                        help="Step 4: gamma block size for the chunked Vx/SOC dipole "
                             "contraction (smaller = less transient memory; default 64)")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.Nint==None: # if none, override to number of Ng
        args.Nint = args.Ng

    print(args)

    # you can only select the backend once and it must be before you use any xp functions
    if xp.backend != args.backend:
        xp.backend = args.backend

    threadctl = ThreadpoolController()
    threadctl.limit(limits=args.t)

    with timer_ctx("Build H"):
        H = Hamiltonian(args)

    with timer_ctx("Load/make guesses"):
        guess = get_davidson_guess_3D(args.guess, H)
        if guess is None:
            guess = H.make_guess(args.k)
        elif args.preconditioner == 'davBOpc':
            H.ensure_davBOpc_data(args.k)
        if (args.preconditioner in ('davBO', 'davBOpc', 'davBOs')
                and not args.bo_spectrum
                and not args.davBOspec):
            H.release_full_pc("after davBO guess")

    if args.bo_spectrum:
        with timer_ctx("BO spectrum"):
            Ad_vn, Ad_n = H.BO_spectrum(args.k)
            if hasattr(Ad_vn, 'get'):
                Ad_vn = Ad_vn.get()
                Ad_n = Ad_n.get()
            numpy.savez_compressed(
                args.bo_spectrum,
                bo_spectrum=Ad_vn,
                bo_surfaces=Ad_n,
                args=vars(args),
            )
        if args.guess is None and args.preconditioner in ('davBO', 'davBOs'):
            H.release_full_pc("after BO spectrum")

    # Step 4 change 5: optionally batch the Davidson matvec (Hx_chunked) so the
    # per-iteration working set scales with --matvec_batch_size, not the subspace.
    if args.matvec_batch_size is not None:
        print(f"[step4] Davidson matvec batched with chunk={args.matvec_batch_size}")
        matvec = partial(H.Hx_chunked, chunk=args.matvec_batch_size)
    else:
        matvec = H.Hx

    # Periodically snapshot the Ritz vectors so a wall-time kill or OOM on a later
    # cycle leaves a recoverable checkpoint (<evecs>.ckpt.npz).  Off unless both
    # --evecs and --checkpoint_every are set.
    _ckpt_cb = make_checkpoint_callback(args.evecs, args.checkpoint_every)
    with timer_ctx(f"Davidson of size {H.size}"):
        _capture_started = _nsys_capture_start()
        try:
            conv, e_approx, evecs = lib.davidson1(
                matvec,
                guess,
                H.preconditioner,
                nroots=args.k,
                max_cycle=args.iterations,
                verbose=args.verbosity,
                max_space=args.subspace,
                max_memory=get_davidson_mem(0.75),
                tol=args.davidson_tol,
                tol_residual=args.davidson_residual_tol,
                callback=_ckpt_cb,
                lock=args.lock,
                lock_tol_factor=args.lock_tol_factor,
            )
        finally:
            _nsys_capture_stop(_capture_started)

    print("Davidson:", e_approx)
    print(conv)

    char,proj = get_wfc_Om_proj_wS(evecs,H)
    el2, ej2, elz, ejz, esz = get_jls_expectations(evecs, H)
    p02_z, p02_r, P02_R = get_p01_radial(evecs,H)
    print("p02, radial momentum between state 0 and 1:", p02_r)
    print("P02 nuclear:", P02_R)
    print("e_approx, char, proj:")

    with numpy.printoptions(precision=3, linewidth=numpy.inf, suppress=True):
        for e, M, prj, in zip(e_approx, char, proj):
            print(f"{e:12e}", M, prj)
    print()
    print("Now reprinting with <l^2>, <j^2>, <lz>, <jz>, <sz>:")
    with numpy.printoptions(precision=3, linewidth=numpy.inf, suppress=True):
        for e, l,j,lz,jz,sz in zip(e_approx, el2, ej2, elz, ejz, esz):
            print(f"{e:12e}", f"{l:.2f}", f"{j:.2f}", f"{lz:.2f}", f"{jz:.2f}", f"{sz:.2f}")
    print()
    # --- SOC-visibility diagnostics ------------------------------------------
    # <l.s> is diagonal in the spinor-spherical-harmonic basis, so it is exactly
    # 1/2(<j^2> - <l^2> - 3/4).  <l.s> ~ 0 => spin decoupled (Sigma, Lambda=0)
    # state where SOC (~ l.s) does nothing regardless of alpha; |<l.s>| ~ O(1)
    # (approaching -1 for a j=1/2 p-state, +1/2 for j=3/2) => genuine orbital
    # angular momentum on which SOC acts.  weight(|Omega|=3/2) is the Pi-character
    # signal: a 2Pi state carries Omega = 1/2 AND 3/2, a 2Sigma only Omega = 1/2.
    print("Now reprinting SOC diagnostics: <l.s>, weight(|Omega|=1/2), weight(|Omega|=3/2):")
    print("  (<l.s>~0 => spin-decoupled Sigma; weight(|Omega|=3/2)>0 => Pi character / SOC live)")
    els = 0.5 * (ej2 - el2 - 0.75)                       # <l.s> = 1/2(<j^2> - <l^2> - 3/4)
    Om_abs = xp.abs(H.Om)
    mask_half  = (xp.abs(Om_abs - 0.5) < 1e-6)           # |Omega| = 1/2 columns
    mask_three = (xp.abs(Om_abs - 1.5) < 1e-6)           # |Omega| = 3/2 columns
    w_half  = xp.sum(proj * mask_half[None, :],  axis=1)  # proj is per-state, per-Omega weight
    w_three = xp.sum(proj * mask_three[None, :], axis=1)
    with numpy.printoptions(precision=3, linewidth=numpy.inf, suppress=True):
        for e, ls_, w1, w3 in zip(e_approx, els, w_half, w_three):
            print(f"{e:12e}", f"<l.s>={ls_:+.4f}", f"w|O|=1/2={w1:.4f}", f"w|O|=3/2={w3:.4f}")
    print()
    print("Now reprinting with <sx>, <sy>, <sz>  (note: <sy>=0 for real eigenvectors):")
    esx, esy, esz_spin = get_spin_expectations(evecs, H)
    with numpy.printoptions(precision=3, linewidth=numpy.inf, suppress=True):
        for e, sx, sy, sz in zip(e_approx, esx, esy, esz_spin):
            print(f"{e:12e}", f"sx={sx:+.4f}", f"sy={sy:+.4e}", f"sz={sz:+.4f}")

    if args.evecs:
        if hasattr(evecs, 'get'):
            evecs = evecs.get()
        # warning: even though evecs will be cpu readable, H will only be readable on a node with gpu
        # best to reconstruct H from args on a cpu for plotting purposes 
        # H_new = Hamiltonian(Namespace(**NPZFILE['args'].item()))
        numpy.savez_compressed(args.evecs, guess=evecs, args=vars(args), e_approx=e_approx)
        print("Wrote eigenvectors to", args.evecs)
        # final save succeeded -> the periodic checkpoint is now redundant; drop it
        _ckpt = str(args.evecs) + ".ckpt.npz"
        if args.checkpoint_every and os.path.exists(_ckpt):
            try:
                os.remove(_ckpt)
            except OSError:
                pass

    if args.bo_spectrum:
        bo = Ad_vn[1,0] - Ad_vn[0,0]
        print("BO gap", bo)
        if all(conv):
            ex = e_approx[1] - e_approx[0]
            ex2 = e_approx[2] -e_approx[0]
            print("exact, exact2, bo, error2:", ex,ex2, bo, (bo-ex2)/ex2)
    elif all(conv):
        ex = e_approx[1] - e_approx[0]
        ex2 = e_approx[2] - e_approx[0]
        print("exact gap, exact gap2", ex, ex2)

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

    if not all(conv):
        print("WARNING: Not all eigenvalues converged")
        exit(1)
    else:
        print("All eigenvalues converged")
