#!/usr/bin/env python3
"""
Combine split .npy outputs from ps_cart_spin_mpi.py and run the same
post-processing (BO/PS eigenvalues, Weyl transform, vib gaps).

Single run:
  python S_3D/combine_splits.py --folder . --splits 4 --potential erf_coulomb \\
    -J 0.5 -Ptheta 0.7071 -Pphi 0.5 -alpha 1e4 -M_1 2000 -M_2 2000 -R 89

Sweep (same parameter grid as run_sweep.sh; combine all and save to file):
  python S_3D/combine_splits.py --sweep --folder . --splits 4 --out results_sweep.json \\
    --potential erf_coulomb -R 89
  Missing split files are skipped by default; use --no-skip-missing to raise on first missing.
  Output: JSON list of result dicts (or .npz if --out ends with .npz).
  Edit SWEEP_* at top of this file to match run_sweep.sh (masses, alphas, J/Ptheta/Pphi).

Files expected:
  matrix_spin_{potential}_J_{J}_Pth_{Ptheta}_Pph_{Pphi}_a_{alpha}_m_{M_2}_{name}_split_{k}.npy
for k in 1..splits.
"""
import os
import sys
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

sys.path.append(os.path.abspath("lib"))
from constants import *
from functools import partial
import potentials
from hamiltonian import KE, KE_FFT

#sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib'))

import numpy as np


# Use numpy backend for combine (no GPU needed)
import xp
from hamiltonian import inverse_weyl_transform, KE
import warnings
warnings.filterwarnings('ignore', message=r'\[cupy\.linalg\.(eigh|eigvalsh)\].*Attempting to override with torch')

# Parameter grid matching run_sweep.sh (used when --sweep; edit to match your run_sweep.sh)
#SWEEP_MASSES = [2000.0]
#SWEEP_ALPHAS = [0.0, 100.0, 1000.0]
#SWEEP_J = [0.0, 0.5, 1.0, 1.0, 1.5, 1.5]
#SWEEP_PTHETA = [0.0, 0.7071, 1.0, 1.4142, 1.2247, 1.8708]
#SWEEP_PPHI = [0.0, 0.5, 1.0, 0.0, 1.5, 0.5]

#SWEEP_MASSES = [2000.0]
#SWEEP_ALPHAS = [0.0, 100.0, 200.0, 400.0, 800.0, 1000.0, 1500.0, 2000.0, 4000.0]
#SWEEP_J = [0.5, 1.5, 1.5]
#SWEEP_PTHETA = [0.7071, 1.2247, 1.8708]
#SWEEP_PPHI = [0.5, 1.5, 0.5]

SWEEP_MASSES = [5.0]
SWEEP_ALPHAS = [0.0, 100000.0, 500000.0, 1000000.0, 5000000.0, 10000000.0,50000000.0,100000000.0]
SWEEP_J = [0.5, 1.5, 1.5]
SWEEP_PTHETA = [0.7071, 1.2247, 1.8708]
SWEEP_PPHI = [0.5, 1.5, 0.5]



def parse_args():
    parser = argparse.ArgumentParser(description='Combine split MPI outputs and diagonalize.')
    parser.add_argument('--folder', default='.', type=str, help='Directory containing split .npy files')
    parser.add_argument('--splits', type=int, required=True, help='Number of splits (split_idx 1..splits)')
    parser.add_argument('--potential', default='borgis', choices=['erf_coulomb', 'borgis'])
    parser.add_argument('-J', type=float, help='J (required unless --sweep)')
    parser.add_argument('-Pphi', type=float, help='Pphi (required unless --sweep)')
    parser.add_argument('-Ptheta', type=float, help='Ptheta (required unless --sweep)')
    parser.add_argument('-alpha', default=0.0, type=float)
    parser.add_argument('-M_1', type=float, help='M_1 (required unless --sweep)')
    parser.add_argument('-M_2', type=float, help='M_2 (required unless --sweep)')
    parser.add_argument('-R', dest='NR', default=101, type=int)
    parser.add_argument('--backend', default='cupy')
    parser.add_argument('--sweep', action='store_true', help='Loop over run_sweep parameter grid and combine all')
    parser.add_argument('--out', default='', type=str, help='Output file for results (required with --sweep). Use .json or .npz')
    parser.add_argument('--no-skip-missing', action='store_true', dest='no_skip_missing',
                        help='With --sweep: raise on first missing file (default is to skip and continue)')
    return parser.parse_args()


def file_prefix(args):
    return (
        f'matrix_spin_{args.potential}_J_{args.J}_Pth_{args.Ptheta}_Pph_{args.Pphi}_a_{args.alpha}_m_{args.M_2}_'
    )


def load_splits(folder, prefix, splits, names):
    """Load arrays for split_idx 1..splits and sum (splits are disjoint)."""
    out = {}
    folder = os.path.realpath(folder)  # resolve symlinks so we look in the actual directory
    for name in names:
        path1 = os.path.join(folder, f'{prefix}{name}_split_1.npy')
        if not os.path.isfile(path1):
            raise FileNotFoundError(
                f'Missing {path1}\n'
                f'Folder (resolved): {folder}\n'
                f'Tip: ps_cart_spin_mpi.py writes to the job cwd. Find files with:\n'
                f'  find /path/to/root -name "matrix_spin_*_split_1.npy"'
            )
        first = np.load(path1)
        out[name] = np.asarray(first, dtype=first.dtype).copy()
        for k in range(2, splits + 1):
            #print("Loading split", k)
            path = os.path.join(folder, f'{prefix}{name}_split_{k}.npy')
            if not os.path.isfile(path):
                raise FileNotFoundError(f'Missing {path}')
            out[name] += np.load(path)
    return out


def _tofloat(x):
    """Convert array scalar to Python float for JSON. Handles CuPy arrays via .get()."""
    if hasattr(x, 'get'):
        x = x.get()  # CuPy: transfer to CPU
    return float(np.asarray(x).ravel()[0].real)


def compute_one(args):
    """
    Load splits for one parameter set, run post-processing, return a dict of
    parameters and scalar results. Uses global batch_eigvalsh.
    Raises FileNotFoundError if any split file is missing.
    """
    folder = os.path.abspath(args.folder)
    prefix = file_prefix(args)
    NR = args.NR
    result = {
        'alpha': args.alpha, 'M_1': args.M_1, 'M_2': args.M_2,
        'J': args.J, 'Ptheta': args.Ptheta, 'Pphi': args.Pphi,
        'potential': args.potential,
    }

    names = [
        'Ad_nsg', 'Ad_nse', 'EPSg', 'EPSe',
        'exp_RxGamma_y_gs', 'exp_RxGamma_y_es',
        'exp_RxGamma_z_gs', 'exp_RxGamma_z_es',
        'exp_sx_gs', 'exp_sx_es',
        'exp_sy_gs', 'exp_sy_es',
        'exp_sz_gs', 'exp_sz_es',
        'exp_lx_gs', 'exp_lx_es',
        'exp_ly_gs', 'exp_ly_es',
        'exp_lz_gs', 'exp_lz_es',
        'exp_lx2_gs', 'exp_lx2_es',
        'exp_ly2_gs', 'exp_ly2_es',
        'exp_lz2_gs', 'exp_lz2_es',
    ]
    print('Loading split files from', folder, '...')
    data = load_splits(folder, prefix, args.splits, names)
    Ad_nsg = xp.asarray(data['Ad_nsg'])
    Ad_nse = xp.asarray(data['Ad_nse'])
    EPSg = xp.asarray(data['EPSg'])
    EPSe = xp.asarray(data['EPSe'])
    exp_RxGamma_y_gs = xp.asarray(data['exp_RxGamma_y_gs'])
    exp_RxGamma_y_es = xp.asarray(data['exp_RxGamma_y_es'])
    exp_RxGamma_z_gs = xp.asarray(data['exp_RxGamma_z_gs'])
    exp_RxGamma_z_es = xp.asarray(data['exp_RxGamma_z_es'])
    exp_sx_gs  = xp.asarray(data['exp_sx_gs'])
    exp_sx_es  = xp.asarray(data['exp_sx_es'])
    exp_sy_gs  = xp.asarray(data['exp_sy_gs'])
    exp_sy_es  = xp.asarray(data['exp_sy_es'])
    exp_sz_gs  = xp.asarray(data['exp_sz_gs'])
    exp_sz_es  = xp.asarray(data['exp_sz_es'])
    exp_lx_gs  = xp.asarray(data['exp_lx_gs'])
    exp_lx_es  = xp.asarray(data['exp_lx_es'])
    exp_ly_gs  = xp.asarray(data['exp_ly_gs'])
    exp_ly_es  = xp.asarray(data['exp_ly_es'])
    exp_lz_gs  = xp.asarray(data['exp_lz_gs'])
    exp_lz_es  = xp.asarray(data['exp_lz_es'])
    exp_lx2_gs = xp.asarray(data['exp_lx2_gs'])
    exp_lx2_es = xp.asarray(data['exp_lx2_es'])
    exp_ly2_gs = xp.asarray(data['exp_ly2_gs'])
    exp_ly2_es = xp.asarray(data['exp_ly2_es'])
    exp_lz2_gs = xp.asarray(data['exp_lz2_gs'])
    exp_lz2_es = xp.asarray(data['exp_lz2_es'])
    extent_func = potentials.extents_borgis if args.potential == 'borgis' else potentials.extents_erf_coulomb
    M_1 = args.M_1
    M_2 = args.M_2
    m_e = 1

    if args.potential == 'borgis':
        print(f"Waring: All masses scaled to AMU for {args.potential}!")
        m_e *= AMU_TO_AU
        M_1 *= AMU_TO_AU
        M_2 *= AMU_TO_AU
    
    mu   = xp.sqrt(M_1*M_2*m_e/(M_1+M_2+m_e))
    mur  = (M_1+M_2)*m_e/(M_1+M_2+m_e)
    mu12 = M_1*M_2/(M_1+M_2)
    
    extent = extent_func(mu12)
    R_min, R_max = extent[0], extent[1]
    R = xp.linspace(R_min, R_max, args.NR)
    dR = R[1] - R[0]
    P_R = xp.fft.fftshift(xp.fft.fftfreq(args.NR, dR)) * 2 * xp.pi
    Rval, Pval = xp.meshgrid(R, P_R, indexing='ij')

    Pphi = args.Pphi
    Ptheta = args.Ptheta
    #ddR2 = KE(NR, dR, bare=True, cyclic=False)
    ddR2  = KE_FFT(NR, P_R, R)

    Hbo_g = 1/(2*mu12)*(-ddR2 + xp.diag(Pphi**2/R**2) + xp.diag(Ptheta**2/R**2) + xp.diag(1/(2*R)**2)) + xp.diag(Ad_nsg)
    Ad_vn_g = batch_eigvalsh(Hbo_g)
    e_bo_g = xp.sort(Ad_vn_g.flatten())
    bo_vib_ggap = e_bo_g[1] - e_bo_g[0]
    result['e_bo_g0'] = _tofloat(e_bo_g[0])
    result['bo_vib_ggap'] = _tofloat(bo_vib_ggap)

    Hbo_e = 1/(2*mu12)*(-ddR2 + xp.diag(Pphi**2/R**2) + xp.diag(Ptheta**2/R**2) + xp.diag(1/(2*R)**2)) + xp.diag(Ad_nse)
    Ad_vn_e = batch_eigvalsh(Hbo_e)
    e_bo_e = xp.sort(Ad_vn_e.flatten())
    bo_vib_egap = e_bo_e[1] - e_bo_e[0]
    result['e_bo_e0'] = _tofloat(e_bo_e[0])
    result['bo_vib_egap'] = _tofloat(bo_vib_egap)

    EPSg += 1/(2*mu12)*(Pval**2 + Pphi**2/Rval**2 + Ptheta**2/Rval**2 + 1/(2*Rval)**2)
    HPSg = inverse_weyl_transform(EPSg, NR, R, P_R)
    EPSvg, evecs_vg = xp.linalg.eigh(HPSg)
    ps_vib_ggap = _tofloat(EPSvg[1] - EPSvg[0])
    print("EPSg",EPSvg[0:10])
    result['EPSvg0'] = _tofloat(EPSvg[0])
    result['ps_vib_ggap'] = ps_vib_ggap

    EPSe += 1/(2*mu12)*(Pval**2 + Pphi**2/Rval**2 + Ptheta**2/Rval**2 + 1/(2*Rval)**2)
    HPSe = inverse_weyl_transform(EPSe, NR, R, P_R)
    EPSve, evecs_ve = xp.linalg.eigh(HPSe)
    ps_vib_egap = _tofloat(EPSve[1] - EPSve[0])
    print("EPSe",EPSve[0:10])
    result['EPSve0'] = _tofloat(EPSve[0])
    result['ps_vib_egap'] = ps_vib_egap

    HGamma_RRy_gs = inverse_weyl_transform(exp_RxGamma_y_gs, NR, R, P_R)
    HGamma_RRy_es = inverse_weyl_transform(exp_RxGamma_y_es, NR, R, P_R)
    HGamma_RRz_gs = inverse_weyl_transform(exp_RxGamma_z_gs, NR, R, P_R)
    HGamma_RRz_es = inverse_weyl_transform(exp_RxGamma_z_es, NR, R, P_R)
    Hsx_gs  = inverse_weyl_transform(exp_sx_gs, NR, R, P_R)
    Hsx_es  = inverse_weyl_transform(exp_sx_es, NR, R, P_R)
    Hsy_gs  = inverse_weyl_transform(exp_sy_gs, NR, R, P_R)
    Hsy_es  = inverse_weyl_transform(exp_sy_es, NR, R, P_R)
    Hsz_gs  = inverse_weyl_transform(exp_sz_gs, NR, R, P_R)
    Hsz_es  = inverse_weyl_transform(exp_sz_es, NR, R, P_R)
    Hlx_gs  = inverse_weyl_transform(exp_lx_gs, NR, R, P_R)
    Hlx_es  = inverse_weyl_transform(exp_lx_es, NR, R, P_R)
    Hly_gs  = inverse_weyl_transform(exp_ly_gs, NR, R, P_R)
    Hly_es  = inverse_weyl_transform(exp_ly_es, NR, R, P_R)
    Hlz_gs  = inverse_weyl_transform(exp_lz_gs, NR, R, P_R)
    Hlz_es  = inverse_weyl_transform(exp_lz_es, NR, R, P_R)
    Hlx2_gs = inverse_weyl_transform(exp_lx2_gs, NR, R, P_R)
    Hlx2_es = inverse_weyl_transform(exp_lx2_es, NR, R, P_R)
    Hly2_gs = inverse_weyl_transform(exp_ly2_gs, NR, R, P_R)
    Hly2_es = inverse_weyl_transform(exp_ly2_es, NR, R, P_R)
    Hlz2_gs = inverse_weyl_transform(exp_lz2_gs, NR, R, P_R)
    Hlz2_es = inverse_weyl_transform(exp_lz2_es, NR, R, P_R)

    RxGamma_y_gs = xp.conj(evecs_vg[:,0]).T @ (HGamma_RRy_gs @ evecs_vg[:,0])
    RxGamma_y_es = xp.conj(evecs_ve[:,0]).T @ (HGamma_RRy_es @ evecs_ve[:,0])
    RxGamma_z_gs = xp.conj(evecs_vg[:,0]).T @ (HGamma_RRz_gs @ evecs_vg[:,0])
    RxGamma_z_es = xp.conj(evecs_ve[:,0]).T @ (HGamma_RRz_es @ evecs_ve[:,0])

    v_sx_gs = xp.conj(evecs_vg[:,0]).T @ (Hsx_gs @ evecs_vg[:,0])
    v_sx_es = xp.conj(evecs_ve[:,0]).T @ (Hsx_es @ evecs_ve[:,0])
    v_sy_gs = xp.conj(evecs_vg[:,0]).T @ (Hsy_gs @ evecs_vg[:,0])
    v_sy_es = xp.conj(evecs_ve[:,0]).T @ (Hsy_es @ evecs_ve[:,0])
    v_sz_gs = xp.conj(evecs_vg[:,0]).T @ (Hsz_gs @ evecs_vg[:,0])
    v_sz_es = xp.conj(evecs_ve[:,0]).T @ (Hsz_es @ evecs_ve[:,0])

    v_lx_gs = xp.conj(evecs_vg[:,0]).T @ (Hlx_gs @ evecs_vg[:,0])
    v_lx_es = xp.conj(evecs_ve[:,0]).T @ (Hlx_es @ evecs_ve[:,0])
    v_ly_gs = xp.conj(evecs_vg[:,0]).T @ (Hly_gs @ evecs_vg[:,0])
    v_ly_es = xp.conj(evecs_ve[:,0]).T @ (Hly_es @ evecs_ve[:,0])
    v_lz_gs = xp.conj(evecs_vg[:,0]).T @ (Hlz_gs @ evecs_vg[:,0])
    v_lz_es = xp.conj(evecs_ve[:,0]).T @ (Hlz_es @ evecs_ve[:,0])

    v_lx2_gs = xp.conj(evecs_vg[:,0]).T @ (Hlx2_gs @ evecs_vg[:,0])
    v_lx2_es = xp.conj(evecs_ve[:,0]).T @ (Hlx2_es @ evecs_ve[:,0])
    v_ly2_gs = xp.conj(evecs_vg[:,0]).T @ (Hly2_gs @ evecs_vg[:,0])
    v_ly2_es = xp.conj(evecs_ve[:,0]).T @ (Hly2_es @ evecs_ve[:,0])
    v_lz2_gs = xp.conj(evecs_vg[:,0]).T @ (Hlz2_gs @ evecs_vg[:,0])
    v_lz2_es = xp.conj(evecs_ve[:,0]).T @ (Hlz2_es @ evecs_ve[:,0])

    def _s(x):
        return _tofloat(x)

    # Store expectation values for output
    result['RxGamma_y_gs'] = _s(RxGamma_y_gs)
    result['RxGamma_y_es'] = _s(RxGamma_y_es)
    result['RxGamma_z_gs'] = _s(RxGamma_z_gs)
    result['RxGamma_z_es'] = _s(RxGamma_z_es)
    result['v_sx_gs'] = _s(v_sx_gs)
    result['v_sx_es'] = _s(v_sx_es)
    result['v_sy_gs'] = _s(v_sy_gs)
    result['v_sy_es'] = _s(v_sy_es)
    result['v_sz_gs'] = _s(v_sz_gs)
    result['v_sz_es'] = _s(v_sz_es)
    result['v_lx_gs'] = _s(v_lx_gs)
    result['v_lx_es'] = _s(v_lx_es)
    result['v_ly_gs'] = _s(v_ly_gs)
    result['v_ly_es'] = _s(v_ly_es)
    result['v_lz_gs'] = _s(v_lz_gs)
    result['v_lz_es'] = _s(v_lz_es)
    result['v_lx2_gs'] = _s(v_lx2_gs)
    result['v_lx2_es'] = _s(v_lx2_es)
    result['v_ly2_gs'] = _s(v_ly2_gs)
    result['v_ly2_es'] = _s(v_ly2_es)
    result['v_lz2_gs'] = _s(v_lz2_gs)
    result['v_lz2_es'] = _s(v_lz2_es)
    result['L_x_gs'] = _s(-v_lx_gs - v_sx_gs)
    result['L_y_gs'] = _s(Ptheta - v_ly_gs - v_sy_gs)
    result['L_z_gs'] = _s(Pphi - v_lz_gs - v_sz_gs)
    result['L_x_es'] = _s(-v_lx_es - v_sx_es)
    result['L_y_es'] = _s(Ptheta - v_ly_es - v_sy_es)
    result['L_z_es'] = _s(Pphi - v_lz_es - v_sz_es)
    # Single plottable angle in y-z plane (deg), measured from +y axis: 0=+y, 90=+z, ±180=-y, -90=-z
    s_yz_gs = np.sqrt(result['v_sy_gs']**2 + result['v_sz_gs']**2)
    s_yz_es = np.sqrt(result['v_sy_es']**2 + result['v_sz_es']**2)
    result['theta_yz_gs'] = np.degrees(np.arctan2(result['v_sz_gs'], result['v_sy_gs'])) if s_yz_gs > 1e-15 else float('nan')
    result['theta_yz_es'] = np.degrees(np.arctan2(result['v_sz_es'], result['v_sy_es'])) if s_yz_es > 1e-15 else float('nan')

    result['report'] = format_result_report(result)
    return result


def format_result_report(r):
    """Format Rxgamma / l / s / sum table and <sx>, <lx>, <lx^2> block like ps_cart_spin_mpi.py."""
    fmt = "  {:>12.7f}"
    lines = []
    # Header line with run params
    lines.append("alpha=%s  M=%s  J=%s  Ptheta=%s  Pphi=%s" % (r['alpha'], r['M_2'], r['J'], r['Ptheta'], r['Pphi']))
    lines.append("")
    # Rxgamma, l, s, sum table
    lines.append("gs")
    lines.append("         Rxgamma          l           s             sum              N")
    check_x_gs = r['v_lx_gs'] + r['v_sx_gs']
    check_gamma_y_gs = r['RxGamma_y_gs'] + r['v_ly_gs'] + r['v_sy_gs']
    check_gamma_z_gs = r['RxGamma_z_gs'] + r['v_lz_gs'] + r['v_sz_gs']
    lines.append("  x " + fmt.format(0.0) + fmt.format(r['v_lx_gs']) + fmt.format(r['v_sx_gs']) + fmt.format(check_x_gs) + fmt.format(r['L_x_gs']))
    lines.append("  y " + fmt.format(r['RxGamma_y_gs']) + fmt.format(r['v_ly_gs']) + fmt.format(r['v_sy_gs']) + fmt.format(check_gamma_y_gs) + fmt.format(r['L_y_gs']))
    lines.append("  z " + fmt.format(r['RxGamma_z_gs']) + fmt.format(r['v_lz_gs']) + fmt.format(r['v_sz_gs']) + fmt.format(check_gamma_z_gs) + fmt.format(r['L_z_gs']))
    s2_gs = r['v_sx_gs']**2 + r['v_sy_gs']**2 + r['v_sz_gs']**2
    l2_gs = r['v_lx_gs']**2 + r['v_ly_gs']**2 + r['v_lz_gs']**2  
    l2op_gs = r['v_lx2_gs'] + r['v_ly2_gs'] + r['v_lz2_gs']
    
    lines.append("")
    lines.append("  gs: <sx>^2 + <sy>^2 + <sz>^2   " + fmt.format(s2_gs))
    lines.append("  gs: <lx>^2 + <ly>^2 + <lz>^2   " + fmt.format(l2_gs))    
    lines.append("  gs: <lx^2> + <ly^2> + <lz^2>   " + fmt.format(l2op_gs))
    #lines.append("  gs: L_x= " + fmt.format(r['L_x_gs']) + "  L_y= " + fmt.format(r['L_y_gs']) + "  L_z= " + fmt.format(r['L_z_gs']))
    lines.append("")
    lines.append("--------------------------------")
    lines.append("es")
    lines.append("         Rxgamma          l           s             sum              N")
    check_x_es = r['v_lx_es'] + r['v_sx_es']
    check_gamma_y_es = r['RxGamma_y_es'] + r['v_ly_es'] + r['v_sy_es']
    check_gamma_z_es = r['RxGamma_z_es'] + r['v_lz_es'] + r['v_sz_es']
    lines.append("  x " + fmt.format(0.0) + fmt.format(r['v_lx_es']) + fmt.format(r['v_sx_es']) + fmt.format(check_x_es) + fmt.format(r['L_x_es']))
    lines.append("  y " + fmt.format(r['RxGamma_y_es']) + fmt.format(r['v_ly_es']) + fmt.format(r['v_sy_es']) + fmt.format(check_gamma_y_es) + fmt.format(r['L_y_es']))
    lines.append("  z " + fmt.format(r['RxGamma_z_es']) + fmt.format(r['v_lz_es']) + fmt.format(r['v_sz_es']) + fmt.format(check_gamma_z_es) + fmt.format(r['L_z_es']))
    lines.append("")
    # <sx>^2, <lx>^2, <lx^2> sums
    s2_es = r['v_sx_es']**2 + r['v_sy_es']**2 + r['v_sz_es']**2
    l2_es = r['v_lx_es']**2 + r['v_ly_es']**2 + r['v_lz_es']**2
    l2op_es = r['v_lx2_es'] + r['v_ly2_es'] + r['v_lz2_es']
    lines.append("  es: <sx>^2 + <sy>^2 + <sz>^2   " + fmt.format(s2_es))
    lines.append("  es: <lx>^2 + <ly>^2 + <lz>^2   " + fmt.format(l2_es))
    lines.append("  es: <lx^2> + <ly^2> + <lz^2>   " + fmt.format(l2op_es))
    lines.append("  es: L_x= " + fmt.format(r['L_x_es']) + "  L_y= " + fmt.format(r['L_y_es']) + "  L_z= " + fmt.format(r['L_z_es']))
    
    lines.append("EPSg= " + fmt.format(r['EPSvg0']))
    lines.append("EPSe= " + fmt.format(r['EPSve0']))
    # y-z plane: angles from +y and +z axes (can add to 90° or 180° depending on quadrant)
    s_yz = np.sqrt(r['v_sy_gs']**2 + r['v_sz_gs']**2)
    if s_yz > 1e-15:
        sy_deg = np.degrees(np.arccos(np.clip(r['v_sy_gs'] / s_yz, -1.0, 1.0)))
        sz_deg = np.degrees(np.arccos(np.clip(r['v_sz_gs'] / s_yz, -1.0, 1.0)))
        lines.append("sy angle (from +y)= " + fmt.format(sy_deg) + "  sz angle (from +z)= " + fmt.format(sz_deg))
        lines.append("theta_yz (deg, from +y toward +z)= " + fmt.format(r['theta_yz_gs']) + "  [use for 1D plots]")
    return "\n".join(lines)


def print_results(result):
    """Print result dict in human-readable form (single-run mode)."""
    print("e_bo g.s. [0]", result['e_bo_g0'], "  bo_vib_ggap", result['bo_vib_ggap'])
    print("e_bo e.s. [0]", result['e_bo_e0'], "  bo_vib_egap", result['bo_vib_egap'])
    print("EPSvg[0]", result['EPSvg0'], "  ps_vib_ggap", result['ps_vib_ggap'])
    print("EPSve[0]", result['EPSve0'], "  ps_vib_egap", result['ps_vib_egap'])
    print("")
    print(result['report'])


def run_sweep(args):
    """Loop over parameter grid (matching run_sweep.sh), combine splits, save to --out."""
    masses = getattr(args, 'sweep_masses', SWEEP_MASSES)
    alphas = getattr(args, 'sweep_alphas', SWEEP_ALPHAS)
    J_vals = getattr(args, 'sweep_J', SWEEP_J)
    Ptheta_vals = getattr(args, 'sweep_Ptheta', SWEEP_PTHETA)
    Pphi_vals = getattr(args, 'sweep_Pphi', SWEEP_PPHI)

    results = []
    n_skip = 0
    for a in alphas:
        for m in masses:
            for i in range(len(J_vals)):
                J, Pth, Pph = J_vals[i], Ptheta_vals[i], Pphi_vals[i]
                one = SimpleNamespace(
                    folder=args.folder, splits=args.splits, potential=args.potential,
                    J=J, Ptheta=Pth, Pphi=Pph, alpha=a, M_1=m, M_2=m, NR=args.NR,
                )
                try:
                    r = compute_one(one)
                    results.append(r)
                    print("OK alpha=%s M=%s J=%s Pth=%s Pph=%s" % (a, m, J, Pth, Pph), flush=True)
                except FileNotFoundError as e:
                    if not getattr(args, 'no_skip_missing', False):
                        n_skip += 1
                        print("SKIP (missing) alpha=%s M=%s J=%s: %s" % (a, m, J, e), flush=True)
                    else:
                        raise

    print("Total: %d computed, %d skipped" % (len(results), n_skip), flush=True)

    out = args.out
    if not out:
        raise ValueError("--out is required with --sweep")
    out = os.path.abspath(out)
    if out.endswith('.npz'):
        # Store as structured arrays for numpy
        keys = list(results[0].keys()) if results else []
        data = {k: np.array([r[k] for r in results]) for k in keys}
        np.savez(out, **data)
        print("Wrote %s (%d rows)" % (out, len(results)))
    else:
        # Default: JSON
        with open(out, 'w') as f:
            json.dump(results, f, indent=2)
        print("Wrote %s (%d entries)" % (out, len(results)))

    # Always write a human-readable report file for easy comparison
    report_path = os.path.splitext(out)[0] + "_report.txt"
    with open(report_path, 'w') as f:
        for i, r in enumerate(results, 1):
            header = "========== Run %d/%d: alpha=%s  M=%s  J=%s  Ptheta=%s  Pphi=%s ==========" % (
                i, len(results), r['alpha'], r['M_2'], r['J'], r['Ptheta'], r['Pphi'])
            f.write(header + "\n")
            f.write(r['report'] + "\n")
            f.write("\n")
    print("Wrote %s (readable report for comparison)" % report_path)
    return results


if __name__ == '__main__':
    args = parse_args()

    if xp.backend != args.backend:
        xp.backend = args.backend

    batch_eigvalsh = xp.linalg.eigvalsh

    if xp.backend == 'cupy':
        try:
            import torch
            torch.cuda.current_device()
            def torch_eigvalsh(H):
                return xp.asarray(torch.linalg.eigvalsh(torch.from_dlpack(H)))
            batch_eigvalsh = torch_eigvalsh
        except (ModuleNotFoundError, AssertionError):
            pass

    if args.sweep:
        if not args.out:
            sys.exit("With --sweep you must specify --out (e.g. --out results_sweep.json)")
        run_sweep(args)
    else:
        for attr in ('J', 'Pphi', 'Ptheta', 'M_1', 'M_2'):
            if getattr(args, attr.replace('-', '_'), None) is None:
                sys.exit("Single-run mode requires -J, -Pphi, -Ptheta, -M_1, -M_2")
        print(args)
        result = compute_one(args)
        print_results(result)
