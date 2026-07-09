import argparse as ap
from pathlib import Path

from itertools import chain
from functools import partial

import os, sys
sys.path.append(os.path.abspath("lib"))

import xp
import numpy as np # only use this for reading and writing objects
import linalg_helper as lib
#from pyscf import lib
import potentials
from constants import AMU_TO_AU
from hamiltonian import KE, KE_FFT, inverse_weyl_transform
from debug import timer_ctx
from time import perf_counter


if __name__ == '__main__':
    from tqdm import tqdm
else:  # mock this out for use in Jupyter Notebooks etc
    def tqdm(iterator, **kwargs):
        print(f"Mock call to tqdm({kwargs})")
        return iterator


def _to_numpy(value):
    """Copy a backend array to a small NumPy array for diagnostic printing."""
    if hasattr(value, "get"):
        value = value.get()
    elif hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def get_davidson_mem(fraction):
    if fraction > 1 or fraction < 0:
        raise RuntimeError("Fraction of memory for Davidson must be on [0, 1]")

    if xp.backend == 'cupy':
        import cupy
        free_bytes, total_bytes = cupy.cuda.Device().mem_info
        davidson_mem = fraction * free_bytes / 1024**2
        print(
            f"Davidson will consume up to {int(davidson_mem)}MB of GPU memory "
            f"(free {int(free_bytes / 1024**2)}MB / total {int(total_bytes / 1024**2)}MB)."
        )
        return davidson_mem

    try:
        system_memory_mb = (os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')) / 1024**2
    except (ValueError, OSError, AttributeError):
        print("Unable to determine system memory!")
        system_memory_mb = 8000
    davidson_mem = fraction * system_memory_mb
    print(f"Davidson will consume up to {int(davidson_mem)}MB of memory.")
    return davidson_mem


def free_gpu_cache():
    if xp.backend == 'cupy':
        import cupy
        cupy.cuda.Stream.null.synchronize()
        cupy.get_default_memory_pool().free_all_blocks()
        cupy.get_default_pinned_memory_pool().free_all_blocks()


# -------------------------------------------------------------------------
# Cheap Pauli-matrix applications on the spin axis (axis=1) of an array
# whose shape is (B, 2, Nx, Ny, Nz).  These replace einsums of the form
# ``einsum('sS,...,BSxyz->Bsxyz', sigma, ..., x)`` which otherwise force a
# generic einsum kernel even though sigma is 2x2 and trivial.  Each helper
# allocates one new tensor; arithmetic afterwards stays contiguous.
# -------------------------------------------------------------------------
def _apply_sigma_x(X):
    """sx = [[0,1],[1,0]] applied along spin axis (axis=1)."""
    return xp.stack((X[:, 1], X[:, 0]), axis=1)


def _apply_sigma_y(X):
    """sy = [[0,-1j],[1j,0]] applied along spin axis (axis=1)."""
    return xp.stack((-1j * X[:, 1], 1j * X[:, 0]), axis=1)


def _apply_sigma_z(X):
    """sz = [[1,0],[0,-1]] applied along spin axis (axis=1)."""
    return xp.stack((X[:, 0], -X[:, 1]), axis=1)


def phase4_rbm_predict(Hx, sources, nroots, svd_tol=1e-10, compute_residuals=True):
    """Project the current Hamiltonian into existing snapshot vectors."""
    blocks = []
    total = 0
    for name, vectors in sources:
        nvec = len(vectors)
        blocks.append((name, vectors, total, total + nvec))
        total += nvec

    overlap = xp.zeros((total, total), dtype=complex)
    heff = xp.zeros((total, total), dtype=complex)
    hblocks = [] if compute_residuals else None

    for _, va, ia, ib in blocks:
        for _, vb, ja, jb in blocks:
            overlap[ia:ib, ja:jb] = xp.einsum(
                'an,bn->ab', xp.conj(va), vb, optimize=True
            )

    for _, vb, ja, jb in blocks:
        hvb = Hx(vb)
        if compute_residuals:
            hblocks.append((hvb, ja, jb))
        for _, va, ia, ib in blocks:
            heff[ia:ib, ja:jb] = xp.einsum(
                'an,bn->ab', xp.conj(va), hvb, optimize=True
            )
        if not compute_residuals:
            del hvb

    s = _to_numpy(overlap)
    h = _to_numpy(heff)
    s = 0.5 * (s + s.conj().T)
    h = 0.5 * (h + h.conj().T)

    metric_evals, metric_vecs = np.linalg.eigh(s)
    keep = metric_evals > svd_tol * max(float(metric_evals[-1].real), 1.0)
    if not np.any(keep):
        raise RuntimeError("RBM snapshot overlap matrix is numerically rank zero")

    basis = metric_vecs[:, keep] / np.sqrt(metric_evals[keep])[None, :]
    h_orth = basis.conj().T @ h @ basis
    h_orth = 0.5 * (h_orth + h_orth.conj().T)
    evals, eigvecs = np.linalg.eigh(h_orth)
    nout = min(nroots, len(evals))
    coeff = basis @ eigvecs[:, :nout]

    nelem = sources[0][1].shape[1]
    ritz = xp.zeros((nout, nelem), dtype=complex)
    for _, vectors, ia, ib in blocks:
        cblock = xp.asarray(coeff[ia:ib, :].T)
        ritz += xp.einsum('ra,an->rn', cblock, vectors, optimize=True)

    residual_norms = None
    if compute_residuals:
        hritz = xp.zeros((nout, nelem), dtype=complex)
        for hvectors, ja, jb in hblocks:
            cblock = xp.asarray(coeff[ja:jb, :].T)
            hritz += xp.einsum('ra,an->rn', cblock, hvectors, optimize=True)

        residual = hritz - xp.asarray(evals[:nout])[:, None] * ritz
        residual_norms = _to_numpy(xp.linalg.norm(residual, axis=1)).real
    ritz_norms = _to_numpy(xp.linalg.norm(ritz, axis=1)).real
    return {
        "basis_vectors": total,
        "canonical_dim": int(np.count_nonzero(keep)),
        "metric_min": float(metric_evals[keep][0].real),
        "metric_max": float(metric_evals[keep][-1].real),
        "eigenvalues": evals[:nout].real,
        "residual_norms": residual_norms,
        "ritz_norms": ritz_norms,
        "ritz_vectors": ritz,
    }


def phase4_select_rbm_sources(snapshot_store, target_j, bank_size):
    """Choose the nearest solved P snapshots for the reduced basis."""
    if bank_size <= 0:
        return []
    nearest = sorted(
        snapshot_store,
        key=lambda item: (abs(item[0] - target_j), item[2]),
    )[:bank_size]
    nearest.sort(key=lambda item: item[2])
    return [(f"P{j}", vectors) for j, vectors, _ in nearest]


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
        #self.ddR2, _ = KE_Borisov_3D(self.R, bare=True)
        #self.ddR2  = KE(args.NR, dR, bare=True, cyclic=False)
        self.ddR2  = KE_FFT(args.NR, self.P_R, self.R)
    
        self.ddx2 = KE(args.Nx, dx, bare=True, cyclic=False)
        self.P_x  = xp.fft.fftshift(xp.fft.fftfreq(args.Nx, dx)) * 2 * xp.pi
        #self.ddx2 = KE_FFT(args.Nx, self.P_x, self.x)
        self.ddx1 = KE(args.Nx, dx, bare=True, cyclic=False, order=1) 

        self.ddy2 = KE(args.Ny, dy, bare=True, cyclic=False)
        self.P_y  = xp.fft.fftshift(xp.fft.fftfreq(args.Ny, dy)) * 2 * xp.pi  
        #self.ddy2 = KE_FFT(args.Ny, self.P_y, self.y)
        self.ddy1 = KE(args.Ny, dy, bare=True, cyclic=False, order=1)

        self.ddz2 = KE(args.Nz, dz, bare=True, cyclic=False)
        self.P_z  = xp.fft.fftshift(xp.fft.fftfreq(args.Nz, dz)) * 2 * xp.pi
        #self.ddz2 = KE_FFT(args.Nz, self.P_z, self.z)
        self.ddz1 = KE(args.Nz, dz, bare=True, cyclic=False, order=1)
    
        self.R_grid = self.xb_grid = self.yb_grid = self.zb_grid = None
        self.x_grid = self.y_grid = self.z_grid = None
        self.Vgrid = None
        
        #only pauli matrices no hbar/2 term
        #self.sx = xp.array([[0,1],[1,0]])
        #self.sy = xp.array([[1,0],[0,-1]])
        #self.sz = xp.array([[0,1j],[-1j,0]])

        self.sx = xp.array([[0,1],[1,0]])
        self.sy = xp.array([[0,-1j],[1j,0]])
        self.sz = xp.array([[1,0],[0,-1]])

        self.si = xp.eye(2)

        self.E1 = self.E2 = None
        
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

    def xyz_broadcast(self):
        return (
            self.x[:, None, None],
            self.y[None, :, None],
            self.z[None, None, :],
        )

    def V_at_R(self, Ri, spitvals=False):
        x, y, z = self.xyz_broadcast()
        return self.V(self.R[Ri], x, y, z, spitvals=spitvals)

    def Efield_at_R(self, Ri):
        x, y, z = self.xyz_broadcast()
        return self.Efield(self.R[Ri], x, y, z)

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

    def BO_energies(self, iR, sequence, args):
        
        NR,Nx,Ny,Nz = self.shape
    
        Ad_nsg = xp.zeros(NR)
        Ad_nse = xp.zeros(NR)
        ivalg = xp.zeros([NR,1])
        ivale = xp.zeros([NR,1])

        evecs = None
        for i in sequence:
            print("Atom Ri idx",i, "Atom Ri",self.R[i],flush=True)
            diag = self.buildDiag(i)   
            V_i = self.V_at_R(i)

            guess_ns = xp.exp(-(V_i - xp.min(V_i))**2/27.211**2).ravel()
            guess_zeros = xp.zeros(len(guess_ns))
            guess_spin = xp.array([xp.append(guess_ns, guess_zeros),xp.append(guess_zeros, guess_ns)])
            if i == iR or evecs is None:
                guess_bo = guess_spin
            else:
                guess_bo = evecs

            E1, E2 = self.Efield_at_R(i)
            c1 = 0.5 * (1/137)**2 * E1 * self.alpha / (self.m_e**2)
            c2 = 0.5 * (1/137)**2 * E2 * self.alpha / (self.m_e**2)
            coef12 = c1 + c2
            xi, yi, zi = self.x, self.y, self.z
            mu12, M1, M2 = self.mu12, self.M_1, self.M_2
            w_y_coef12 = yi[None, :, None] * coef12
            w_z_coef12 = zi[None, None, :] * coef12
            # Combine the two ``w_x`` channels once and for all so that the
            # matvec only ever touches a single (Nx, Ny, Nz) tensor.
            w_x_total = ((xi - self.R[i] * mu12 / M1)[:, None, None] * c1
                       + (xi + self.R[i] * mu12 / M2)[:, None, None] * c2)
            soc_data_i = (w_y_coef12, w_z_coef12, w_x_total)

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
    
        #print("Ad_nsg",Ad_nsg)
        #exit()

        return Ad_nsg, Ad_nse, ivalg, ivale        
                
    def Tx(self, x):
        # ``x`` already has shape (B, 2, Nx, Ny, Nz).  ``self.si`` was the 2x2
        # identity, so the spin-index einsum it used to do is a no-op; we
        # drop it and keep just the three 1D second-derivative contractions.
        return -1.0/(2*self.mur) * (
            xp.einsum('xa,Bsayz->Bsxyz', self.ddx2, x, optimize=True)
            + xp.einsum('yb,Bsxbz->Bsxyz', self.ddy2, x, optimize=True)
            + xp.einsum('zc,Bsxyc->Bsxyz', self.ddz2, x, optimize=True)
        )

    def soc_full(self, x, soc_data_i):
        # ``soc_data_i`` now carries the combined ``w_x_total = w_x_coef1 +
        # w_x_coef2`` so that ``w_x_coef1`` / ``w_x_coef2`` need never be
        # touched separately again inside the matvec.
        w_y_coef12, w_z_coef12, w_x_total = soc_data_i
        ddx1, ddy1, ddz1 = self.ddx1, self.ddy1, self.ddz1

        # ----- Block 1 :  sigma . (w * D x) -----------------------------------
        # Share the three first-derivative tensors of x across every term.
        # The original code's 8 einsums each redid one of these derivatives.
        Dx_x = xp.einsum('xa,Bsayz->Bsxyz', ddx1, x, optimize=True)
        Dy_x = xp.einsum('yb,Bsxbz->Bsxyz', ddy1, x, optimize=True)
        Dz_x = xp.einsum('zc,Bsxyc->Bsxyz', ddz1, x, optimize=True)

        A_sx = w_y_coef12 * Dz_x - w_z_coef12 * Dy_x
        A_sy = w_z_coef12 * Dx_x - w_x_total * Dz_x
        A_sz = w_x_total * Dy_x - w_y_coef12 * Dx_x

        # ----- Block 2 :  sigma . D (w * x) -----------------------------------
        # Group the two ``w_x_*`` terms by precomputing ``w_x_total * x`` once,
        # then take only the derivatives we actually need (6 of them, vs the
        # original 8 hidden inside einsum).
        wy_x  = w_y_coef12 * x
        wz_x  = w_z_coef12 * x
        wxt_x = w_x_total  * x

        Dz_wy  = xp.einsum('zc,Bsxyc->Bsxyz', ddz1, wy_x,  optimize=True)
        Dy_wz  = xp.einsum('yb,Bsxbz->Bsxyz', ddy1, wz_x,  optimize=True)
        Dx_wz  = xp.einsum('xa,Bsayz->Bsxyz', ddx1, wz_x,  optimize=True)
        Dz_wxt = xp.einsum('zc,Bsxyc->Bsxyz', ddz1, wxt_x, optimize=True)
        Dy_wxt = xp.einsum('yb,Bsxbz->Bsxyz', ddy1, wxt_x, optimize=True)
        Dx_wy  = xp.einsum('xa,Bsayz->Bsxyz', ddx1, wy_x,  optimize=True)

        B_sx = Dz_wy  - Dy_wz
        B_sy = Dx_wz  - Dz_wxt
        B_sz = Dy_wxt - Dx_wy

        # ----- Combine sigma_x / sigma_y / sigma_z parts ---------------------
        # Cheap spin-axis swaps + sign flips beat a generic einsum here.
        return -0.25j * (
            _apply_sigma_x(A_sx + B_sx)
            + _apply_sigma_y(A_sy + B_sy)
            + _apply_sigma_z(A_sz + B_sz)
        )


    def ps_ham(
        self,
        Veff,
        theta_R,
        gammacoeff,
        R,
        soc_data_i,
        include_gammasq,
        include_spin_terms=True,
    ):

        neg_inv2mu = -1.0/(2*self.mu12)
        gammacoeff_R, gammacoeff_phi, gammacoeff_theta = gammacoeff
        use_soc = (self.soc == 'full')
        inv_R = 1 / R
        inv_R2 = inv_R**2
        xcoord = self.x[None, None, :, None, None]
        ycoord = self.y[None, None, None, :, None]
        zcoord = self.z[None, None, None, None, :]
        theta = theta_R[None, None, :, :, :]

        def Dx(v):
            return xp.einsum('xa,Bsayz->Bsxyz', self.ddx1, v, optimize=True)

        def Dy(v):
            return xp.einsum('yb,Bsxbz->Bsxyz', self.ddy1, v, optimize=True)

        def Dz(v):
            return xp.einsum('zc,Bsxyc->Bsxyz', self.ddz1, v, optimize=True)

        def Gamma_x(v):
            return -0.5 * (theta * Dx(v) + Dx(theta * v))

        def Lz_xpy(v):
            return xcoord * Dy(v)

        def Lz_ypx(v):
            return -ycoord * Dx(v)

        def Ly_zpx(v):
            return zcoord * Dx(v)

        def Ly_xpz(v):
            return -xcoord * Dz(v)

        def gammasq_action(v):
            gamma_x_v = Gamma_x(v)
            out = Gamma_x(gamma_x_v)
            Dxx_v = Dx(Dx(v))
            out = out + inv_R2 * (
                (ycoord**2 + zcoord**2) * Dxx_v
                + xcoord**2 * Dy(Dy(v))
                + xcoord**2 * Dz(Dz(v))
                - xcoord * Dy(ycoord * Dx(v))
                - ycoord * Dx(xcoord * Dy(v))
                - zcoord * Dx(xcoord * Dz(v))
                - xcoord * Dz(zcoord * Dx(v))
            )
            spin_y = 1j * inv_R2 * (Ly_zpx(v) + Ly_xpz(v))
            spin_z = 1j * inv_R2 * (Lz_xpy(v) + Lz_ypx(v))
            if include_spin_terms:
                out = out + _apply_sigma_y(spin_y) + _apply_sigma_z(spin_z)
            return neg_inv2mu * out

        def Hx_ps(xdav):
            x = xdav.reshape((-1,) + self.bospinshape).astype(complex, copy=False)

            Hpsdav = (
                Veff * x
                + self.Tx(x)
                + gammacoeff_R * Gamma_x(x)
                - gammacoeff_phi * inv_R * (Lz_xpy(x) + Lz_ypx(x))
                + gammacoeff_theta * inv_R * (Ly_zpx(x) + Ly_xpz(x))
            )
            if include_spin_terms:
                Hpsdav = (
                    Hpsdav
                    - 0.5j * gammacoeff_phi * inv_R * _apply_sigma_z(x)
                    + 0.5j * gammacoeff_theta * inv_R * _apply_sigma_y(x)
                )

            if include_gammasq:
                Hpsdav = Hpsdav + gammasq_action(x)

            if use_soc:
                Hpsdav = Hpsdav + self.soc_full(x, soc_data_i)

            return Hpsdav.reshape(xdav.shape)

        return Hx_ps

    def Hbo_dav(self, Ri, soc_data_i):
        # Snapshot the per-R Hamiltonian data once.  All identity-spin einsums
        # collapse into pure broadcasted multiplications.
        V = self.V_at_R(Ri)
        use_soc = (self.soc == 'full')

        def Hxbo(xdav):
            x = xdav.reshape((-1,) + self.bospinshape)
            Hbodav = V * x + self.Tx(x)
            if use_soc:
                Hbodav = Hbodav + self.soc_full(x, soc_data_i)
            return Hbodav.reshape(xdav.shape)

        return Hxbo

    def buildDiag(self,Ri):
        NR,Nx,Ny,Nz = self.shape
        ke  = xp.zeros([Nx,Ny,Nz],dtype=self.ddx2.dtype)
        ke += xp.diag(self.ddx2)[:,None,None]
        ke += xp.diag(self.ddy2)[None,:,None]
        ke += xp.diag(self.ddz2)[None,None,:]
        ke *= -1 / (2*self.mur)
        diag = self.V_at_R(Ri) + ke
        diagravel = diag.ravel()
        diagspin = xp.append(diagravel,diagravel)
        return diagspin




def generalized_sequence(NR, num_splits,split_idx):
        nodes = xp.linspace(0, NR, num_splits + 1, dtype=xp.int32).tolist()
    
        parts = []
        midpoint_idx = num_splits // 2
    
        for i in range(num_splits):
          start = nodes[i]
          end = nodes[i+1]
          
          if i < midpoint_idx:
            if i == 0:
                chunk = xp.arange(end, start - 1, -1)
            else:
                chunk = xp.arange(end, start, -1)

          else:
            if i == num_splits - 1:
                chunk = xp.arange(start+1, end)
            else:
              chunk = (xp.arange(start + 1, end + 1))

          parts.append(chunk)

        return parts[split_idx-1]

def exp_l_s(H, evecs_save, Ridx, t1, t2):
    """Compute <L>, <S>, and <L^2> for the first two states."""
    Nx, Ny, Nz = H.boshape
    evecs = evecs_save[:2, :].reshape(2, 2, Nx, Ny, Nz)
    evecs_conj = xp.conj(evecs)

    # Apply each spatial derivative once to both states and both spin components.
    dx_evecs = xp.einsum('xa,nsayz->nsxyz', H.ddx1, evecs, optimize=True)
    dy_evecs = xp.einsum('yb,nsxbz->nsxyz', H.ddy1, evecs, optimize=True)
    dz_evecs = xp.einsum('zc,nsxyc->nsxyz', H.ddz1, evecs, optimize=True)

    x = H.x[None, None, :, None, None]
    y = H.y[None, None, None, :, None]
    z = H.z[None, None, None, None, :]

    # L = -i r x grad. The -i phase drops out of <L^2>.
    lx_evecs = y * dz_evecs - z * dy_evecs
    exp_lx = -1j * xp.einsum('nsxyz,nsxyz->n', evecs_conj, lx_evecs, optimize=True)
    exp_lx2 = xp.einsum('nsxyz,nsxyz->n', xp.conj(lx_evecs), lx_evecs, optimize=True)
    del lx_evecs

    ly_evecs = z * dx_evecs - x * dz_evecs
    exp_ly = -1j * xp.einsum('nsxyz,nsxyz->n', evecs_conj, ly_evecs, optimize=True)
    exp_ly2 = xp.einsum('nsxyz,nsxyz->n', xp.conj(ly_evecs), ly_evecs, optimize=True)
    del ly_evecs, dz_evecs

    lz_evecs = x * dy_evecs - y * dx_evecs
    exp_lz = -1j * xp.einsum('nsxyz,nsxyz->n', evecs_conj, lz_evecs, optimize=True)
    exp_lz2 = xp.einsum('nsxyz,nsxyz->n', xp.conj(lz_evecs), lz_evecs, optimize=True)

    exp_sx = xp.einsum('nsxyz,sS,nSxyz->n', evecs_conj, 0.5 * H.sx, evecs, optimize=True)
    exp_sy = xp.einsum('nsxyz,sS,nSxyz->n', evecs_conj, 0.5 * H.sy, evecs, optimize=True)
    exp_sz = xp.einsum('nsxyz,sS,nSxyz->n', evecs_conj, 0.5 * H.sz, evecs, optimize=True)

    exp_l = (exp_lx[0], exp_lx[1], exp_ly[0], exp_ly[1], exp_lz[0], exp_lz[1])
    exp_s = (exp_sx[0], exp_sx[1], exp_sy[0], exp_sy[1], exp_sz[0], exp_sz[1])
    exp_l2 = (exp_lx2[0], exp_lx2[1], exp_ly2[0], exp_ly2[1], exp_lz2[0], exp_lz2[1])

    return exp_l, exp_s, exp_l2


def R_Gamma_exp_mem_opt(H, Ridx, evecs_save):
    """Memory-light R x Gamma expectation for arbitrary Nx, Ny, and Nz."""
    Nx, Ny, Nz = H.boshape

    evecs = evecs_save[:2, :].reshape(2, 2, Nx, Ny, Nz)
    evecs_conj = xp.conj(evecs)

    # Each contraction uses distinct indices for its own input/output axis.
    # The other two spatial axes are carried through unchanged.
    dx_evecs = xp.einsum('ij,nsjkl->nsikl', H.ddx1, evecs, optimize=True)
    dy_evecs = xp.einsum('ij,nskjl->nskil', H.ddy1, evecs, optimize=True)
    dz_evecs = xp.einsum('ij,nsklj->nskli', H.ddz1, evecs, optimize=True)

    x = H.x.reshape(1, 1, Nx, 1, 1)
    y = H.y.reshape(1, 1, 1, Ny, 1)
    z = H.z.reshape(1, 1, 1, 1, Nz)

    gamma_z_orb_evecs = z * dx_evecs - x * dz_evecs
    gamma_y_orb_evecs = y * dx_evecs - x * dy_evecs

    sigma_y = xp.einsum('nsxyz,sS,nSxyz->n', evecs_conj, H.sy, evecs, optimize=True)
    sigma_z = xp.einsum('nsxyz,sS,nSxyz->n', evecs_conj, H.sz, evecs, optimize=True)

    R_Gamma_y = (
        1j * xp.einsum('nsxyz,nsxyz->n', evecs_conj, gamma_z_orb_evecs, optimize=True)
        - 0.5 * sigma_y
    )
    R_Gamma_z = (
        -1j * xp.einsum('nsxyz,nsxyz->n', evecs_conj, gamma_y_orb_evecs, optimize=True)
        - 0.5 * sigma_z
    )

    return R_Gamma_y[0], R_Gamma_y[1], R_Gamma_z[0], R_Gamma_z[1]


def R_Gamma_exp_mem_opt_check(H, Ridx, evecs_save, t1=None, t2=None):
    """Return R x Gamma values and R x Gamma + L + S residuals."""
    R_Gamma_y_gs, R_Gamma_y_es, R_Gamma_z_gs, R_Gamma_z_es = R_Gamma_exp_mem_opt(
        H, Ridx, evecs_save
    )
    exp_l, exp_s, _ = exp_l_s(H, evecs_save, Ridx, t1, t2)
    _, _, ly_gs, ly_es, lz_gs, lz_es = exp_l
    _, _, sy_gs, sy_es, sz_gs, sz_es = exp_s

    return {
        "R_Gamma": (R_Gamma_y_gs, R_Gamma_y_es, R_Gamma_z_gs, R_Gamma_z_es),
        "residual_y_gs": R_Gamma_y_gs + ly_gs + sy_gs,
        "residual_y_es": R_Gamma_y_es + ly_es + sy_es,
        "residual_z_gs": R_Gamma_z_gs + lz_gs + sz_gs,
        "residual_z_es": R_Gamma_z_es + lz_es + sz_es,
    }


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
    parser.add_argument('-Pphi', default=0, type=float)
    parser.add_argument('-Ptheta', default=0, type=float)
    parser.add_argument('-alpha', default=0, type=float)
    parser.add_argument('-R', dest="NR", metavar="NR", default=101, type=int)
    parser.add_argument('-x', dest="Nx", metavar="Nx", default=250, type=int)
    parser.add_argument('-y', dest="Ny", metavar="Ny", default=250, type=int)
    parser.add_argument('-z', dest="Nz", metavar="Nz", default=250, type=int)
    parser.add_argument('--bo_spectrum', metavar='bo_spectrum True or spec.npz', type=Path, default=None)
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
    parser.add_argument('--backend', default='cupy')
    parser.add_argument('--soc', choices=['no_soc','full'], type=str, default='full')
    parser.add_argument('--Gammasq', action='store_true')
    parser.add_argument('--no-spin-terms', '--no_spin_terms', action='store_true',
                        help='omit spin-dependent PS Hamiltonian terms, matching the no_spin_erf path')
    parser.add_argument('--phase4-rbm-diagnostics', action='store_true',
                        help='predict each PS solve from existing snapshot vectors before full Davidson')
    parser.add_argument('--phase4-rbm-svd-tol', type=float, default=1e-10,
                        help='relative overlap eigenvalue cutoff for RBM diagnostics')
    parser.add_argument('--phase4-rbm-bank-size', type=int, default=4,
                        help='maximum number of solved P snapshots retained for RBM diagnostics')
    parser.add_argument('--phase4-rbm-store-size', type=int, default=8,
                        help='maximum solved P snapshots kept for nearest-bank selection')
    parser.add_argument('--phase4-rbm-polish-guess', action='store_true',
                        help='use RBM Ritz vectors as the Davidson initial guess when available')
    parser.add_argument('--phase4-stop-after-ps-solve', type=int, default=None, metavar='N',
                        help='exit after completing and reporting zero-based PS solve N')
    parser.add_argument('--phase4-skip-expectations', action='store_true',
                        help='skip PS expectation-value work after each Davidson solve')
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
    folder = os.getcwd()
    
    H = Hamiltonian(args)
    start_script = perf_counter()   
    NR,Nx,Ny,Nz = H.shape
    Nelec = 2*Nx*Ny*Nz 
    
    Ad_nsg = xp.zeros(NR)
    Ad_nse = xp.zeros(NR)

    Rval, Pval = H.RP_grid

    EPSg = xp.zeros((NR, NR))
    EPSe = xp.zeros((NR, NR))


    expectation_names = (
        "RxGamma_y_gs", "RxGamma_y_es",
        "RxGamma_z_gs", "RxGamma_z_es",
        "sx_gs", "sx_es", "sy_gs", "sy_es", "sz_gs", "sz_es",
        "lx_gs", "lx_es", "ly_gs", "ly_es", "lz_gs", "lz_es",
        "lx2_gs", "lx2_es", "ly2_gs", "ly2_es", "lz2_gs", "lz2_es",
    )
    
    expectations = {
        name: xp.zeros((NR, NR))
        for name in expectation_names
    }

    gammacoeff_R = -1j*(Pval)/H.mu12 
    gammacoeff_phi = -1j*(H.Pphi/H.R)/H.mu12
    gammacoeff_theta = +1j*(H.Ptheta/H.R)/H.mu12

    ivalg = xp.zeros([NR,1])
    ivale = xp.zeros([NR,1])

    energy_bo = xp.zeros([NR,args.k])
    #evecs_bo = xp.zeros([NR,Nelec],dtype=complex)
    #print("evecs",evecs_bo.shape)


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

    print("sequence",sequence)
    

    jR = NR//2
    #jR = 2
    ps_sequence = list( chain(
            [jR],
            range(jR - 1, -1, -1),
            range(jR + 1, NR)))
    gammacoeff = (gammacoeff_R, gammacoeff_phi, gammacoeff_theta)

    ### BO loop: does BO only and exits
    if (args.bo_spectrum):
        Ad_nsg, Ad_nse, ivalg, ivale = H.BO_energies(iR, sequence, args)

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

        if (args.bo_spectrum != "True"): # i.e. not str("True") but instead it's a path to a save file
            surfaces = xp.stack((Ad_nsg,Ad_nse))
            spectrum = xp.stack((e_bo_g, e_bo_e, EPSv_bog,EPSv_boe))
            if hasattr(Ad_nsg, 'get'):
                surfaces = surfaces.get()
                spectrum = spectrum.get()
            np.savez_compressed(args.bo_spectrum, bo_spectrum=spectrum, bo_surfaces=surfaces, args=vars(args))

        exit()

    ### PS loop, makes E(R,P)    
    evecs = None
    ps_solve_index = 0
    with timer_ctx("R for loop"):
        for i in sequence:
            print("Atom Ri idx",i, "Atom Ri",H.R[i],flush=True)
            diag = H.buildDiag(i)   
            V_i = H.V_at_R(i)

            guess_ns = xp.exp(-(V_i - xp.min(V_i))**2/27.211**2).ravel()
            guess_zeros = xp.zeros(len(guess_ns))
            guess_spin = xp.array([xp.append(guess_ns, guess_zeros),xp.append(guess_zeros, guess_ns)])
            if i == iR or evecs is None:
                guess_bo = guess_spin
            else:
                guess_bo = evecs
            

            E1, E2 = H.Efield_at_R(i)
            c1 = 0.5 * (1/137)**2 * E1 * H.alpha / (H.m_e**2)
            c2 = 0.5 * (1/137)**2 * E2 * H.alpha / (H.m_e**2)
            coef12 = c1 + c2
            xi, yi, zi = H.x, H.y, H.z
            mu12, M1, M2 = H.mu12, H.M_1, H.M_2

            w_y_coef12 = yi[None, :, None] * coef12
            w_z_coef12 = zi[None, None, :] * coef12
            # ``w_x_coef1`` and ``w_x_coef2`` only ever appear as a sum in the
            # matvec, so collapse them now and carry one tensor instead of two.
            w_x_total = ((xi - H.R[i] * mu12 / M1)[:, None, None] * c1
                       + (xi + H.R[i] * mu12 / M2)[:, None, None] * c2)
            soc_data_i = (w_y_coef12, w_z_coef12, w_x_total)

            #guess_spin = xp.append(guess_ns, guess_ns)
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
            #exit()
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
            del guess_bo
            free_gpu_cache()
    
            r1e2, r2e2 = H.V_at_R(i, spitvals=True)
            theta1 = xp.exp(-r1e2)
            theta2 = xp.exp(-r2e2)
            partition = (theta1 + theta2)

            t1 = 1/(1+xp.exp(r1e2-r2e2))
            t2 = 1/(1+xp.exp(r2e2-r1e2))
            theta_R = ((H.M_2*t1-H.M_1*t2)/(H.M_1+H.M_2))

            x, y, z = H.x, H.y, H.z
            
            R = H.R[i]
            evecs_center = None

            xsqR = ( H.x[:, None, None]/R)**2
            diagsq = -1.0/(2*H.mu12)*(
                    - (theta_R**2) * xp.diag(H.ddx2)[:, None, None]
                    + xsqR*xp.diag(H.ddy2)[None, :, None]
                    + xsqR*xp.diag(H.ddz2)[None, None, :]
                ).flatten()

            diag += xp.append(diagsq,diagsq)

            ddy_spin_sq = -1.0/(2*H.mu12)*(0.5j/H.R[i])**2*xp.ones(H.boshape)
            ddz_spin_sq = -1.0/(2*H.mu12)*(0.5j/H.R[i])**2*xp.ones(H.boshape)

            if args.Gammasq and not args.no_spin_terms:
                V_eff = V_i + (ddy_spin_sq + ddz_spin_sq)
            else:
                V_eff = V_i

            evecs_save = None
            rbm_snapshot_store = []
            rbm_legacy_upper_seeded = False
            with timer_ctx("P for loop"):
                print("Pseq", ps_sequence)
                for j in ps_sequence:
                    print("Atom Ri",i,"Atom Pj",j,flush=True)

                    gammacoeff_terms = (gammacoeff_R[i,j], gammacoeff_phi[i], gammacoeff_theta[i])
                    Hx_ps = H.ps_ham(
                        V_eff,
                        theta_R,
                        gammacoeff_terms,
                        R,
                        soc_data_i,
                        args.Gammasq,
                        include_spin_terms=not args.no_spin_terms,
                    )

                    #rng = xp.random.default_rng(0)                        
                    #shape =  (1,)+H.bospinshape
                    #print("shape",shape)
                    #u = rng.standard_normal(shape) + 1j*rng.standard_normal(shape)
                    #v = rng.standard_normal(shape) + 1j*rng.standard_normal(shape)
                    #print("u",u.shape)
                    #Hu = Hx_ps(u)
                    #Hv = Hx_ps(v)
                    ##Hu = H.soc_full(u,soc_data_i)
                    ##Hv = H.soc_full(v,soc_data_i)
                    #lhs = xp.vdot(u.ravel(), Hv.ravel())
                    #rhs = xp.vdot(Hu.ravel(), v.ravel())
                    #print("hermiticity residual:", abs(lhs - rhs), "  |lhs|:", abs(lhs))
                    #exit()

                    if j == jR:
                        guess_ps = evecs
                    elif j == jR + 1 and evecs_center is not None:
                        guess_ps = evecs_center
                    elif evecs_save is not None:
                        guess_ps = evecs_save
                    else:
                        guess_ps = evecs

                    phase4_rbm_enabled = (
                        args.phase4_rbm_diagnostics
                        or args.phase4_rbm_polish_guess
                    )
                    if (
                        phase4_rbm_enabled
                        and j == jR + 1
                        and evecs_center is not None
                        and not rbm_legacy_upper_seeded
                    ):
                        rbm_snapshot_store = [(jR, evecs_center, -1)]
                        rbm_legacy_upper_seeded = True
                        print(
                            f"PHASE4_RBM_BANK_RESET solve={ps_solve_index} "
                            f"R={i} P={j} seeded_from=P{jR}",
                            flush=True,
                        )

                    rbm_prediction = None
                    if phase4_rbm_enabled:
                        rbm_sources = phase4_select_rbm_sources(
                            rbm_snapshot_store,
                            j,
                            args.phase4_rbm_bank_size,
                        )
                        if rbm_sources:
                            with timer_ctx(f"Phase 4 RBM prediction R={i} P={j}"):
                                rbm_prediction = phase4_rbm_predict(
                                    Hx_ps,
                                    rbm_sources,
                                    min(args.k, 4),
                                    args.phase4_rbm_svd_tol,
                                    compute_residuals=args.phase4_rbm_diagnostics,
                                )
                            rbm_e = rbm_prediction["eigenvalues"]
                            if args.phase4_rbm_diagnostics:
                                print(
                                    f"PHASE4_RBM_PREDICT solve={ps_solve_index} R={i} P={j} "
                                    f"sources={[name for name, _ in rbm_sources]} "
                                    f"basis_vectors={rbm_prediction['basis_vectors']} "
                                    f"canonical_dim={rbm_prediction['canonical_dim']} "
                                    f"metric_min={rbm_prediction['metric_min']:.6e} "
                                    f"metric_max={rbm_prediction['metric_max']:.6e} "
                                    f"residuals={np.array2string(rbm_prediction['residual_norms'], precision=6, separator=',')} "
                                    f"ritz_norms={np.array2string(rbm_prediction['ritz_norms'], precision=6, separator=',')} "
                                    f"e={np.array2string(rbm_e, precision=12, separator=',')}",
                                    flush=True,
                                )
                            if args.phase4_rbm_polish_guess:
                                guess_ps = rbm_prediction["ritz_vectors"]
                                if rbm_prediction["residual_norms"] is None:
                                    print(
                                        f"PHASE4_RBM_POLISH_GUESS solve={ps_solve_index} "
                                        f"R={i} P={j}",
                                        flush=True,
                                    )
                                else:
                                    print(
                                        f"PHASE4_RBM_POLISH_GUESS solve={ps_solve_index} "
                                        f"R={i} P={j} residuals="
                                        f"{np.array2string(rbm_prediction['residual_norms'], precision=6, separator=',')}",
                                        flush=True,
                                    )

                    with timer_ctx(f"Davidson of size {H.size}"):
                        conv, e_ps_approx, evecs_save = lib.davidson1(
                            Hx_ps,
                            guess_ps,
                            lambda dx, e, x0: dx/(diag-e+1e-5),
                            nroots=args.k,
                            max_cycle=args.iterations,
                            verbose=args.verbosity,
                            max_space=args.subspace,
                            max_memory=get_davidson_mem(0.75),
                            #tol=1e-12, #FIXME:DEBUG
                            tol=1e-12,
                        )

    
                    print("Davidson:", e_ps_approx)
                    print(conv)
                    if rbm_prediction is not None and args.phase4_rbm_diagnostics:
                        rbm_e = rbm_prediction["eigenvalues"]
                        full_e = _to_numpy(e_ps_approx[:len(rbm_e)]).real
                        err = rbm_e - full_e
                        print(
                            f"PHASE4_RBM_COMPARE solve={ps_solve_index} R={i} P={j} "
                            f"max_abs_err={np.max(np.abs(err)):.6e} "
                            f"err={np.array2string(err, precision=6, separator=',')}",
                            flush=True,
                        )
                    if args.phase4_stop_after_ps_solve == ps_solve_index:
                        print(f"PHASE4_STOP_AFTER_PS_SOLVE solve={ps_solve_index}", flush=True)
                        raise SystemExit(0)
                    if j == jR:
                        evecs_center = evecs_save
                    if phase4_rbm_enabled and args.phase4_rbm_bank_size > 0:
                        rbm_snapshot_store.append((j, evecs_save, ps_solve_index))
                        if len(rbm_snapshot_store) > args.phase4_rbm_store_size:
                            rbm_snapshot_store.pop(0)
                    EPSg[i, j] = e_ps_approx[0]
                    EPSe[i, j] = e_ps_approx[1]
                    ps_solve_index += 1
                    if args.phase4_skip_expectations:
                        continue

                    exp_l, exp_s, exp_l2 = exp_l_s(H,evecs_save,i,t1,t2)
                    lx_gs, lx_es, ly_gs, ly_es, lz_gs, lz_es = exp_l
                    sx_gs, sx_es, sy_gs, sy_es, sz_gs, sz_es = exp_s
                    lx2_gs, lx2_es, ly2_gs, ly2_es, lz2_gs, lz2_es = exp_l2

                    R_Gamma_y_gs, R_Gamma_y_es, R_Gamma_z_gs, R_Gamma_z_es = (
                        R_Gamma_exp_mem_opt(H, i, evecs_save)
                    )

                    expectation_values = {
                        "RxGamma_y_gs": R_Gamma_y_gs,
                        "RxGamma_y_es": R_Gamma_y_es,
                        "RxGamma_z_gs": R_Gamma_z_gs,
                        "RxGamma_z_es": R_Gamma_z_es,
                    
                        "sx_gs": sx_gs, "sx_es": sx_es,
                        "sy_gs": sy_gs, "sy_es": sy_es,
                        "sz_gs": sz_gs, "sz_es": sz_es,
                    
                        "lx_gs": lx_gs, "lx_es": lx_es,
                        "ly_gs": ly_gs, "ly_es": ly_es,
                        "lz_gs": lz_gs, "lz_es": lz_es,
                    
                        "lx2_gs": lx2_gs, "lx2_es": lx2_es,
                        "ly2_gs": ly2_gs, "ly2_es": ly2_es,
                        "lz2_gs": lz2_gs, "lz2_es": lz2_es,
                    }
                    
                    for name, value in expectation_values.items():
                        expectations[name][i, j] = value.real

                    check_gamma_y_gs = R_Gamma_y_gs + sy_gs + ly_gs
                    check_gamma_y_es = R_Gamma_y_es + sy_es + ly_es
                    check_gamma_z_gs = R_Gamma_z_gs + sz_gs + lz_gs
                    check_gamma_z_es = R_Gamma_z_es + sz_es + lz_es
                    check_x_gs = sx_gs + lx_gs
                    check_x_es = sx_es + lx_es

                    def _s(x):
                        return float(xp.asarray(x).real) if xp.size(x) == 1 else float(x)

                    fmt = "  {:>12.7f}"
                    print("gs")
                    print("         Rxgamma          l           s             sum")
                    print("  x " + fmt.format(0.0)           + fmt.format(_s(lx_gs)) + fmt.format(_s(sx_gs)) + fmt.format(_s(check_x_gs)))
                    print("  y " + fmt.format(_s(R_Gamma_y_gs)) + fmt.format(_s(ly_gs)) + fmt.format(_s(sy_gs)) + fmt.format(_s(check_gamma_y_gs)))
                    print("  z " + fmt.format(_s(R_Gamma_z_gs)) + fmt.format(_s(lz_gs)) + fmt.format(_s(sz_gs)) + fmt.format(_s(check_gamma_z_gs)))
                    print("es")
                    print("         Rxgamma          l           s             sum")
                    print("  x " + fmt.format(0.0)           + fmt.format(_s(lx_es)) + fmt.format(_s(sx_es)) + fmt.format(_s(check_x_es)))
                    print("  y " + fmt.format(_s(R_Gamma_y_es)) + fmt.format(_s(ly_es)) + fmt.format(_s(sy_es)) + fmt.format(_s(check_gamma_y_es)))
                    print("  z " + fmt.format(_s(R_Gamma_z_es)) + fmt.format(_s(lz_es)) + fmt.format(_s(sz_es)) + fmt.format(_s(check_gamma_z_es)))
                    
                    print("gs:<sx>^2 + <sy>^2 + <sz>^2", _s(sx_gs)**2 + _s(sy_gs)**2 + _s(sz_gs**2))
                    print("es:<sx>^2 + <sy>^2 + <sz>^2", _s(sx_es)**2 + _s(sy_es)**2 + _s(sz_es**2))
                    print("gs:<lx>^2 + <ly>^2 + <lz>^2", _s(lx_gs)**2 + _s(ly_gs)**2 + _s(lz_gs**2))
                    print("es:<lx>^2 + <ly>^2 + <lz>^2", _s(lx_es)**2 + _s(ly_es)**2 + _s(lz_es**2))
                    print("gs:l(l+1):<lx^2> + <ly^2> + <lz^2>", _s(lx2_gs) + _s(ly2_gs) + _s(lz2_gs))
                    print("es:l(l+1):<lx^2> + <ly^2> + <lz^2>", _s(lx2_es) + _s(ly2_es) + _s(lz2_es))
                    del Hx_ps
                    free_gpu_cache()

                    
                    
                    
    #EPS = xp.loadtxt("rij_matrix.txt")
    #ivalload = xp.loadtxt("ri_values.txt")
    #ival = ivalload.reshape([NR,1])
    #Ad_n= ivalload
    #print("EPSg",EPSg)
    #print("Ad_nsg",Ad_nsg)

    ### Vibrational energies for PS using BO and Weyl

    if args.splits == 0:
        Hbo_g = +1/(2*H.mu12)*(-H.ddR2 + xp.diag(H.Pphi**2/H.R**2)+ xp.diag(H.Ptheta**2/H.R**2)) +xp.diag(Ad_nsg)
        Ad_vn_g = batch_eigvalsh(Hbo_g)
        e_bo_g = xp.sort(Ad_vn_g.flatten())
        print("e_bo_new g.s.",e_bo_g[0:10])
        bo_vib_ggap = e_bo_g[1] - e_bo_g[0]
        print("BO new vib gap g.s.",bo_vib_ggap,flush=True)
        
        Hbo_e = +1/(2*H.mu12)*(-H.ddR2 + xp.diag(H.Pphi**2/H.R**2)+ xp.diag(H.Ptheta**2/H.R**2)) +xp.diag(Ad_nse)
        Ad_vn_e = batch_eigvalsh(Hbo_e)
        e_bo_e = xp.sort(Ad_vn_e.flatten())
        print("e_bo_new e.s.",e_bo_e[0:10])
        bo_vib_egap = e_bo_e[1] - e_bo_e[0]
        print("BO new vib gap e.s.",bo_vib_egap,flush=True)
        
        EPSg += 1/(2*H.mu12)*(Pval**2+H.Pphi**2/Rval**2+H.Ptheta**2/Rval**2)
        HPSg = inverse_weyl_transform(EPSg, H.shape[0], H.R, H.P_R)
        EPSvg, evecs_vg = xp.linalg.eigh(HPSg)
        print("EPSv g.s.",EPSvg[0:10])
        print("PS vib gap g.s.",EPSvg[1]-EPSvg[0],flush=True)
        
        EPSe += 1/(2*H.mu12)*(Pval**2+H.Pphi**2/Rval**2+H.Ptheta**2/Rval**2)
        HPSe = inverse_weyl_transform(EPSe, H.shape[0], H.R, H.P_R)
        EPSve, evecs_ve = xp.linalg.eigh(HPSe)
        print("EPSv e.s.",EPSve[0:10])
        print("PS vib gap e.s.",EPSve[1]-EPSve[0],flush=True)

        H_expectations = {
            name: inverse_weyl_transform(values, NR, H.R, H.P_R)
            for name, values in expectations.items()
        }

        vg = evecs_vg[:, 0]
        ve = evecs_ve[:, 1]
        
        v_expectations = {}
        
        for name, H_op in H_expectations.items():
            if name.endswith("_gs"):
                vec = vg
            elif name.endswith("_es"):
                vec = ve
            else:
                continue
        
            v_expectations[name] = xp.conj(vec).T @ (H_op @ vec)

    else:
        (
            exp_RxGamma_y_gs, exp_RxGamma_y_es,
            exp_RxGamma_z_gs, exp_RxGamma_z_es,
            exp_sx_gs, exp_sx_es, exp_sy_gs, exp_sy_es, exp_sz_gs, exp_sz_es,
            exp_lx_gs, exp_lx_es, exp_ly_gs, exp_ly_es, exp_lz_gs, exp_lz_es,
            exp_lx2_gs, exp_lx2_es, exp_ly2_gs, exp_ly2_es,
            exp_lz2_gs, exp_lz2_es,
        ) = (expectations[name] for name in expectation_names)

        spin_tag = 'no_spin_terms' if args.no_spin_terms else 'spin'
        gammasq_tag = '_110' if args.Gammasq else ''
        matrix_prefix = (
            f'matrix_{spin_tag}{gammasq_tag}_{args.potential}_J_{args.J}_'
            f'Pth_{args.Ptheta}_Pph_{args.Pphi}_a_{args.alpha}_m_{args.M_2}'
        )
        matrix_outputs = {
            'Ad_nsg': Ad_nsg,
            'Ad_nse': Ad_nse,
            'EPSg': EPSg,
            'EPSe': EPSe,
            'exp_RxGamma_y_gs': exp_RxGamma_y_gs,
            'exp_RxGamma_y_es': exp_RxGamma_y_es,
            'exp_RxGamma_z_gs': exp_RxGamma_z_gs,
            'exp_RxGamma_z_es': exp_RxGamma_z_es,
            'exp_sx_gs': exp_sx_gs,
            'exp_sx_es': exp_sx_es,
            'exp_sy_gs': exp_sy_gs,
            'exp_sy_es': exp_sy_es,
            'exp_sz_gs': exp_sz_gs,
            'exp_sz_es': exp_sz_es,
            'exp_lx_gs': exp_lx_gs,
            'exp_lx_es': exp_lx_es,
            'exp_ly_gs': exp_ly_gs,
            'exp_ly_es': exp_ly_es,
            'exp_lz_gs': exp_lz_gs,
            'exp_lz_es': exp_lz_es,
            'exp_lx2_gs': exp_lx2_gs,
            'exp_lx2_es': exp_lx2_es,
            'exp_ly2_gs': exp_ly2_gs,
            'exp_ly2_es': exp_ly2_es,
            'exp_lz2_gs': exp_lz2_gs,
            'exp_lz2_es': exp_lz2_es,
        }

        for name, value in matrix_outputs.items():
            np.save(
                os.path.join(
                    folder,
                    f'{matrix_prefix}_{name}_split_{args.split_idx}.npy',
                ),
                value,
            )
