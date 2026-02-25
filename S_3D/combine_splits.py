#!/usr/bin/env python3
"""
Combine split .npy outputs from ps_cart_spin_mpi.py and run the same
post-processing (BO/PS eigenvalues, Weyl transform, vib gaps).

Usage: pass the same potential, J, Pphi, alpha, M_2 (and other H args) as used
when generating the splits, plus --splits and --folder. Files are expected as:
  matrix_{potential}_j_{J}_m_{Pphi}_a_{alpha}_m_{M_2}_{Ad_nsg|Ad_nse|ivalg|ivale|EPSg|EPSe}_split_{k}.npy
for k in 1..splits.
"""
import os
import sys
import argparse
from pathlib import Path
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


def parse_args():
    parser = argparse.ArgumentParser(description='Combine split MPI outputs and diagonalize.')
    parser.add_argument('--folder', default='.', type=str, help='Directory containing split .npy files')
    parser.add_argument('--splits', type=int, required=True, help='Number of splits (split_idx 1..splits)')
    parser.add_argument('--potential', default='borgis', choices=['erf_coulomb', 'borgis'])
    parser.add_argument('-J', required=True, type=float)
    parser.add_argument('-Pphi', required=True, type=float)
    parser.add_argument('-Ptheta', required=True, type=float)
    parser.add_argument('-alpha', default=0.0, type=float)
    parser.add_argument('-M_1', required=True, type=float)
    parser.add_argument('-M_2', required=True, type=float)
    parser.add_argument('-R', dest='NR', default=101, type=int)
    parser.add_argument('--backend', default='numpy')
    return parser.parse_args()


def file_prefix(args):
    return (
        f'matrix_{args.potential}_j_{args.J}_m_{args.Pphi}_a_{args.alpha}_m_{args.M_2}_'
    )


def load_splits(folder, prefix, splits, names):
    """Load arrays for split_idx 1..splits and sum (splits are disjoint)."""
    out = {}
    for name in names:
        first = np.load(os.path.join(folder, f'{prefix}{name}_split_1.npy'))
        out[name] = np.asarray(first, dtype=first.dtype).copy()
        for k in range(2, splits + 1):
            #print("Loading split", k)
            path = os.path.join(folder, f'{prefix}{name}_split_{k}.npy')
            if not os.path.isfile(path):
                raise FileNotFoundError(f'Missing {path}')
            out[name] += np.load(path)
    return out


def main():
    args = parse_args()
    folder = os.path.abspath(args.folder)
    prefix = file_prefix(args)
    NR = args.NR

    names = ['Ad_nsg', 'Ad_nse', 'ivalg', 'ivale', 'EPSg', 'EPSe']
    print('Loading split files from', folder, '...')
    data = load_splits(folder, prefix, args.splits, names)
    Ad_nsg = xp.asarray(data['Ad_nsg'])
    Ad_nse = xp.asarray(data['Ad_nse'])
    ivalg = xp.asarray(data['ivalg'])
    ivale = xp.asarray(data['ivale'])
    EPSg = xp.asarray(data['EPSg'])
    EPSe = xp.asarray(data['EPSe'])
    print("EPSg",EPSg)
    print("Ad_nsg",Ad_nsg)

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

    Hbo_g = (
        +1/(2*mu12) * (-ddR2 + xp.diag(Pphi**2/R**2) + xp.diag(Ptheta**2/R**2) + xp.diag(1/(2*R)**2))
        + xp.diag(Ad_nsg)
    )
    Hbo_e = (
        +1/(2*mu12) * (-ddR2 + xp.diag(Pphi**2/R**2) + xp.diag(Ptheta**2/R**2) + xp.diag(1/(2*R)**2))
        + xp.diag(Ad_nse)
    )
    Ad_vn_g = xp.linalg.eigvalsh(Hbo_g)
    Ad_vn_e = xp.linalg.eigvalsh(Hbo_e)
    e_bo_g = xp.sort(Ad_vn_g.flatten())
    e_bo_e = xp.sort(Ad_vn_e.flatten())
    print('e_bo_new g.s.', e_bo_g[:10], flush=True)
    print('BO new vib gap g.s.', e_bo_g[1] - e_bo_g[0], flush=True)
    print('e_bo_new e.s.', e_bo_e[:10], flush=True)
    print('BO new vib gap e.s.', e_bo_e[1] - e_bo_e[0], flush=True)

    # #region agent log
    try:
        with open("/home/mb3835/ps-model-exact/.cursor/debug.log", "a") as _f:
            import json
            _f.write(json.dumps({"hypothesisId": "H1", "location": "combine_splits.py:125", "message": "types before EPSg+=", "data": {"type_EPSg": type(EPSg).__module__ + "." + type(EPSg).__name__, "type_Pval": type(Pval).__module__ + "." + type(Pval).__name__, "type_Rval": type(Rval).__module__ + "." + type(Rval).__name__, "type_mu12": str(type(mu12)), "xp_backend": getattr(xp, 'backend', '?')}, "timestamp": __import__("time").time() * 1000}) + "\n")
    except Exception:
        pass
    # #endregion

    EPSg += 1/(2*mu12) * (Pval**2 + Pphi**2/Rval**2 + Ptheta**2/Rval**2 + 1/(2*Rval)**2)
    EPSe += 1/(2*mu12) * (Pval**2 + Pphi**2/Rval**2 + Ptheta**2/Rval**2 + 1/(2*Rval)**2)

    HPSg = inverse_weyl_transform(EPSg, NR, R, P_R)
    HPSe = inverse_weyl_transform(EPSe, NR, R, P_R)
    EPSvg = xp.linalg.eigvalsh(HPSg)
    EPSve = xp.linalg.eigvalsh(HPSe)
    print('EPSv g.s.', xp.sort(EPSvg.flatten())[:10])
    epsvg = xp.sort(EPSvg.flatten())
    print('PS vib gap g.s.', epsvg[1] - epsvg[0])
    print('EPSv e.s.', xp.sort(EPSve.flatten())[:10])
    epsve = xp.sort(EPSve.flatten())
    print('PS vib gap e.s.', epsve[1] - epsve[0])

    # BO-level Weyl (ivalg / ivale)
    EPS_bog = xp.repeat(ivalg, NR, axis=1)
    EPS_bog += 1/(2*mu12) * (Pval**2 + Pphi**2/Rval**2 + Ptheta**2/Rval**2 + 1/(2*Rval)**2)
    EPS_boe = xp.repeat(ivale, NR, axis=1)
    EPS_boe += 1/(2*mu12) * (Pval**2 + Pphi**2/Rval**2 + Ptheta**2/Rval**2 + 1/(2*Rval)**2)
    HPS_bog = inverse_weyl_transform(EPS_bog, NR, R, P_R)
    HPS_boe = inverse_weyl_transform(EPS_boe, NR, R, P_R)
    EPSv_bog = xp.linalg.eigvalsh(HPS_bog)
    EPSv_boe = xp.linalg.eigvalsh(HPS_boe)
    print('EPSv_bo g.s.', xp.sort(EPSv_bog.flatten())[:10])
    print('EPSv_bo e.s.', xp.sort(EPSv_boe.flatten())[:10])

    # Optional: save combined arrays
    out_folder = folder
    np.save(os.path.join(out_folder, f'{prefix}Ad_nsg_combined.npy'), Ad_nsg)
    np.save(os.path.join(out_folder, f'{prefix}Ad_nse_combined.npy'), Ad_nse)
    np.save(os.path.join(out_folder, f'{prefix}EPSg_combined.npy'), EPSg)
    np.save(os.path.join(out_folder, f'{prefix}EPSe_combined.npy'), EPSe)
    print('Saved combined arrays to', out_folder)


if __name__ == '__main__':
    args = parse_args()
    print(args)

    if xp.backend != args.backend:
        xp.backend = args.backend

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

    main()
