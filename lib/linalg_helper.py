#!/usr/bin/env python
# Copyright 2014-2020 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

#
# Heavily modified by Vale Cofer-Shabica <vale.cofershabica@gmail.com>, 2025
#


'''
Extension to scipy.linalg module
'''

import sys
import inspect
import warnings
from functools import reduce
import numpy
import numpy as np
import scipy.linalg
from pyscf.lib import logger
from pyscf.lib import numpy_helper
from pyscf.lib import misc
from pyscf.lib.exceptions import LinearDependencyError
from pyscf import __config__

from debug import timer, timer_ctx

from time import perf_counter

DAVIDSON_LINDEP = getattr(__config__, 'lib_linalg_helper_davidson_lindep', 1e-14)
DSOLVE_LINDEP = getattr(__config__, 'lib_linalg_helper_dsolve_lindep', 1e-13)
MAX_MEMORY = getattr(__config__, 'lib_linalg_helper_davidson_max_memory', 4000)  # 4GB

# def _fill_heff_hermitian(heff, xs, ax, xt, axt, dot):
#     nrow = len(axt)
#     row1 = len(ax)
#     row0 = row1 - nrow
#     for ip, i in enumerate(range(row0, row1)):
#         for jp, j in enumerate(range(row0, i)):
#             heff[i,j] = dot(xt[ip].conj(), axt[jp])
#             heff[j,i] = heff[i,j].conj()
#         heff[i,i] = dot(xt[ip].conj(), axt[ip]).real

#     for i in range(row0):
#         axi = numpy.asarray(ax[i])
#         for jp, j in enumerate(range(row0, row1)):
#             heff[j,i] = dot(xt[jp].conj(), axi)
#             heff[i,j] = heff[j,i].conj()
#         axi = None
#     return heff



def _fill_heff_hermitian(heff, xs, ax, xt, axt, _):
    nrow = len(axt)
    row1 = len(ax)
    row0 = row1 - nrow
    #print("fill_heff", nrow, row1, row0)

    # Stack active blocks
    XT = np.stack(xt)         # shape (nrow, dim)
    AXT = np.stack(axt)       # shape (nrow, dim)

    # === Block A: lower-right (nrow x nrow), symmetric ===
    block_A = XT @ AXT.T.conj()  # shape (nrow, nrow)
    heff[row0:row1, row0:row1] = (block_A + block_A.T.conj()) / 2

    # === Block B: off-diagonal (row0 x nrow), symmetric ===
    if row0 > 0:
        AX = np.stack(ax[:row0])  # shape (row0, dim)
        block_B = XT @ AX.T.conj()  # shape (nrow, row0)
        heff[row0:row1, :row0] = block_B
        heff[:row0, row0:row1] = block_B.T.conj()

    return heff


__lasttime = None

def tic(label):
    global __lasttime
    printing = False
    if printing and __lasttime:
        print(f"EEElapsed {label}", perf_counter() - __lasttime)
    __lasttime = perf_counter()


def davidson1(aop, x0, precond, tol=1e-12, max_cycle=50, max_space=12,
              lindep=DAVIDSON_LINDEP, max_memory=MAX_MEMORY,
              dot=numpy.dot, callback=None,
              nroots=1, verbose=logger.WARN,
              tol_residual=None,
              fill_heff=_fill_heff_hermitian
              ):
    r'''Davidson diagonalization method to solve  a c = e c.  Ref
    [1] E.R. Davidson, J. Comput. Phys. 17 (1), 87-94 (1975).
    [2] http://people.inf.ethz.ch/arbenz/ewp/Lnotes/chapter11.pdf

    Note: This function has an overhead of memory usage ~4*x0.size*nroots

    Args:
        aop : function([x]) => [array_like_x]
            Matrix vector multiplication :math:`y_{ki} = \sum_{j}a_{ij}*x_{jk}`.
        x0 : 1D array or a list of 1D arrays or a function to generate x0 array(s)
            Initial guess.  The initial guess vector(s) are just used as the
            initial subspace bases.  If the subspace is smaller than "nroots",
            eg 10 roots and one initial guess, all eigenvectors are chosen as
            the eigenvectors during the iterations.  The first iteration has
            one eigenvector, the next iteration has two, the third iteration
            has 4, ..., until the subspace size > nroots.
        precond : diagonal elements of the matrix or  function(dx, e, x0) => array_like_dx
            Preconditioner to generate new trial vector.
            The argument dx is a residual vector ``a*x0-e*x0``; e is the current
            eigenvalue; x0 is the current eigenvector.

    Kwargs:
        tol : float
            Convergence tolerance.
        max_cycle : int
            max number of iterations.
        max_space : int
            space size to hold trial vectors.
        lindep : float
            Linear dependency threshold.  The function is terminated when the
            smallest eigenvalue of the metric of the trial vectors is lower
            than this threshold.
        max_memory : int or float
            Allowed memory in MB.
        dot : function(x, y) => scalar
            Inner product
        callback : function(envs_dict) => None
            callback function takes one dict as the argument which is
            generated by the builtin function :func:`locals`, so that the
            callback function can access all local variables in the current
            environment.
        nroots : int
            Number of eigenvalues to be computed.  When nroots > 1, it affects
            the shape of the return value

    Returns:
        conv : bool
            Converged or not
        e : list of floats
            The lowest :attr:`nroots` eigenvalues.
        c : list of 1D arrays
            The lowest :attr:`nroots` eigenvectors.

    Examples:

    >>> from pyscf import lib
    >>> a = numpy.random.random((10,10))
    >>> a = a + a.T
    >>> aop = lambda xs: [numpy.dot(a,x) for x in xs]
    >>> precond = lambda dx, e, x0: dx/(a.diagonal()-e)
    >>> x0 = a[0]
    >>> e, c = lib.davidson(aop, x0, precond, nroots=2)
    >>> len(e)
    2
    '''
    tic("start")
    if isinstance(verbose, logger.Logger):
        log = verbose
    else:
        log = logger.Logger(misc.StreamObject.stdout, verbose)

    if tol_residual is None:
        toloose = numpy.sqrt(tol)
    else:
        toloose = tol_residual
    log.debug1('tol %g  toloose %g', tol, toloose)

    if not callable(precond):
        precond = make_diag_precond(precond)

    if callable(x0):  # lazy initialization to reduce memory footprint
        x0 = x0()
    if isinstance(x0, numpy.ndarray) and x0.ndim == 1:
        x0 = [x0]
    #max_cycle = min(max_cycle, x0[0].size)
    max_space = max_space + (nroots-1) * 4
    # max_space*2 for holding ax and xs, nroots*2 for holding axt and xt
    log.debug1('max_cycle %d  max_space %d  max_memory %d',
               max_cycle, max_space, max_memory)
    dtype = None
    heff = None
    fresh_start = True
    e = None
    v = None
    conv = numpy.zeros(nroots, dtype=bool)
    emin = None

    tic("init")
    for icyc in range(max_cycle):
        tic("cycle start")
        if fresh_start:
            # FIXME: need to convert all of this memory to arrays and
            # keep a counter of their length so I can pass views of
            # them around; e.g.: ax[:space] or what-have-you.
            xs = []
            ax = []
            space = 0
# Orthogonalize xt space because the basis of subspace xs must be orthogonal
# but the eigenvectors x0 might not be strictly orthogonal
            xt = None
            x0len = len(x0)
            xt = _qr(x0, dot, lindep)[0]
            if len(xt) != x0len:
                log.warn('QR decomposition removed %d vectors.', x0len - len(xt))
                if len(xt) == 0:
                    if icyc == 0:
                        msg = 'Initial guess is empty or zero'
                    else:
                        msg = ('No more linearly independent basis were found. '
                               'Unless loosen the lindep tolerance (current value '
                               f'{lindep}), the diagonalization solver is not able '
                               'to find eigenvectors.')
                    raise LinearDependenceError(msg)
            x0 = None
            max_dx_last = 1e9
            tic("fresh start")
        elif len(xt) > 1:
            xt = _qr(xt, dot, lindep)[0]
            xt = xt[:40]  # 40 trial vectors at most
            tic("QR")

        axt = aop(xt)
        tic("aop(xt)")
        for k, xi in enumerate(xt):
            xs.append(xt[k])
            ax.append(axt[k])
        rnow = len(xt)
        head, space = space, space+rnow


        tic("build xs,ax")
        
        if dtype is None:
            try:
                dtype = numpy.result_type(axt[0], xt[0])
            except IndexError:
                raise LinearDependenceError('No linearly independent basis found '
                                            'by the diagonalization solver.')
        if heff is None:  # Lazy initialize heff to determine the dtype
            heff = numpy.empty((max_space+nroots,max_space+nroots), dtype=dtype)
        else:
            heff = numpy.asarray(heff, dtype=dtype)

        elast = e
        vlast = v
        conv_last = conv

        fill_heff(heff, xs, ax, xt, axt, dot)
        tic("fill_heff")
        xt = axt = None
        w, v = scipy.linalg.eigh(heff[:space,:space])
        tic("eigh(subspace)")
        
        e = w[:nroots]
        v = v[:,:nroots]
        conv = numpy.zeros(e.size, dtype=bool)
        if not fresh_start:
            elast, conv_last = _sort_elast(elast, conv_last, vlast, v, log)

        tic("sort_elast")
            
        if elast is None:
            de = e
        elif elast.size != e.size:
            log.debug('Number of roots different from the previous step (%d,%d)',
                      e.size, elast.size)
            de = e
        else:
            de = e - elast

        tic("de")
        
        x0 = None
        x0 = _gen_x0(v, xs)
        ax0 = _gen_x0(v, ax)
        tic("_gen_x0(v,)")

        xt = ax0 - e[:,None]*x0
        dx_norm = scipy.linalg.norm(xt, axis=1)
        conv = (np.abs(de) < tol) & (dx_norm < toloose)
        
        for k, ek in enumerate(e):
            if conv[k] and not conv_last[k]:
                log.debug('root %d converged  |r|= %4.3g  e= %s  max|de|= %4.3g',
                          k, dx_norm[k], ek, de[k])
        ax0 = None
        max_dx_norm = max(dx_norm)
        ide = numpy.argmax(abs(de))
        
        if all(conv):
            log.debug('converged %d %d  |r|= %4.3g  e= %s  max|de|= %4.3g',
                      icyc, space, max_dx_norm, e, de[ide])
            break

        tic("covergence")

        # remove subspace linear dependencies
        keep = ~conv & (dx_norm > np.sqrt(lindep))
        xt = xt[keep]
        tic("norms")
        xt = np.stack([precond(xt_, e[0], x0_) for xt_, x0_ in zip(xt, x0[keep])])
        tic("preconditioner")
        norms = scipy.linalg.norm(xt, axis=1)
        xt /= norms[:, None]
                
        tic("lindep")
        xt, norm_min = _normalize_xt_(xt, xs, lindep)
        tic("normalize")
        log.debug('davidson %d %d  |r|= %4.3g  e= %s  max|de|= %4.3g  lindep= %4.3g',
                  icyc, space, max_dx_norm, e, de[ide], norm_min)

        if icyc > 3:
            mydiff = e[0] - mylast
            #print(f"Last delta: {mydiff}")
            if abs(mydiff) > 1e-12 and mydiff > 0:
                print(f"Hold up! Why isn't e0 monotonic??? {mydiff}")
                #raise RuntimeError("Hold up! Why isn't e0 monotonic???")
        mylast = e[0]
        
        if len(xt) == 0:
            log.debug('Linear dependency in trial subspace. |r| for each state %s',
                      dx_norm)
            conv = dx_norm < toloose
            break

        max_dx_last = max_dx_norm
        fresh_start = space+nroots > max_space

        tic("end")
        # useful, e.g.: for restarts
        if callable(callback):
            callback(locals())

        tic("callback")

    x0 = list(x0)  # nparray -> list

    # Check whether the solver finds enough eigenvectors.
    h_dim = x0[0].size
    if len(x0) < min(h_dim, nroots):
        # Two possible reasons:
        # 1. All the initial guess are the eigenvectors. No more trial vectors
        # can be generated.
        # 2. The initial guess sits in the subspace which is smaller than the
        # required number of roots.
        log.warn(f'Not enough eigenvectors (len(x0)={len(x0)}, nroots={nroots})')

    return numpy.asarray(conv), e, x0


def make_diag_precond(diag, level_shift=1e-3):
    '''Generate the preconditioner function with the diagonal function.'''
    # For diagonal matrix A, precond (Ax-x*e)/(diag(A)-e) is not able to
    # generate linearly independent basis (see issue 1362). Use level_shift to
    # break the correlation between Ax-x*e and diag(A)-e.
    def precond(dx, e, *args):
        diagd = diag - (e - level_shift)
        diagd[abs(diagd)<1e-8] = 1e-8
        return dx/diagd
    return precond


def _qr(xs, dot, lindep=1e-14):
    '''QR decomposition for a list of vectors (for linearly independent vectors only).
    xs = (r.T).dot(qs)
    '''
    nvec = len(xs)
    xs = qs = numpy.array(xs, copy=True)
    rmat = numpy.eye(nvec, order='F', dtype=xs.dtype)

    nv = 0
    for i in range(nvec):
        xi = xs[i]
        for j in range(nv):
            prod = dot(qs[j].conj(), xi)
            xi -= qs[j] * prod
            rmat[:,nv] -= rmat[:,j] * prod
        innerprod = dot(xi.conj(), xi).real
        norm = numpy.sqrt(innerprod)
        if innerprod > lindep:
            qs[nv] = xi/norm
            rmat[:nv+1,nv] /= norm
            nv += 1
    return qs[:nv], numpy.linalg.inv(rmat[:nv,:nv])


def _outprod_to_subspace(v, xs):
    v = numpy.asarray(v)
    ndim = v.ndim
    if ndim == 1:
        v = v[:, None]  # shape: (space, 1)
    # FIXME: the majority of the time is spent building xs; move to
    # everything as arrays for big speedup
    xs = numpy.asarray(xs)  # shape: (space, n)
    x0 = numpy.einsum('ik,ij->kj', v, xs, optimize=True)

    if ndim == 1:
        x0 = x0[0]
    return x0

_gen_x0 = _outprod_to_subspace


def _sort_elast(elast, conv_last, vlast, v, log):
    '''
    Eigenstates may be flipped during the Davidson iterations.  Reorder the
    eigenvalues of last iteration to make them comparable to the eigenvalues
    of the current iterations.
    '''
    head, nroots = vlast.shape
    ovlp = abs(numpy.dot(v[:head].conj().T, vlast))
    mapping = numpy.argmax(ovlp, axis=1)
    found = numpy.any(ovlp > .5, axis=1)

    if log.verbose >= logger.DEBUG:
        ordering_diff = (mapping != numpy.arange(len(mapping)))
        if any(ordering_diff & found):
            log.debug('Old state -> New state')
            for i in numpy.where(ordering_diff)[0]:
                log.debug('  %3d     ->   %3d ', mapping[i], i)

    conv = conv_last[mapping]
    e = elast[mapping]
    conv[~found] = False
    e[~found] = 0.
    return e, conv


def _normalize_xt_(xt, xs, threshold):
    '''Projects out existing basis vectors xs. Also checks whether the precond
    function is ill-conditioned'''

    # Project: xt_mat -= xs.T @ (xs @ xt_mat.T)
    # In detail: subtract each xi's projection onto the span of xs
    proj = xs @ xt.T     # shape: (nbasis, nvecs)
    xt -= (proj.T @ xs)  # shape: (nvecs, ndim)

    # Compute norms
    norms = scipy.linalg.norm(xt, axis=1)
    keep = norms**2 > threshold

    # Normalize and select
    xt = xt[keep]
    norms = norms[keep]
    if xt.size == 0:
        return [], 1
    xt /= norms[:, None]

    # FIXME: I have no idea why, but list() doubles our performance
    return list(xt), norms.min()


LinearDependenceError = LinearDependencyError
