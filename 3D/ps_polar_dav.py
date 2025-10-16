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
        'R', 'P_R', 'R_grid', 'r', 'ph', 'r_grid','p_grid', 'th', 'pg', 't_grid','RP_grid', 'ddph2', 'ddph1',
        'ddR2', 'ddr2', 'ddth2', 'ddth1', 'ddr1','axes','Vgrid','rb_grid','tb_grid','pb_grid',
        'Rinv2', 'rinv2', 'diag', '_preconditioner_data','Pg','Pphi','Ptheta',
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
            'borgis': (potentials.borgis, potentials.extents_borgis),
            'erf_coulomb':(potentials.erf_coulomb, potentials.extents_erf_coulomb)
            }[args.potential]

        extent = extent_func(self.mu12)

        print(f"Potential: {args.potential}")

        if hasattr(args, "extent") and args.extent is not None:
            extent = args.extent

        R_range = extent[:2]
        r_max   = extent[-1]
        print("r_max",r_max)
        print("R_range",R_range)
        print("unscaled coords:", R_range, r_max)

        if r_max < R_range[-1]/2:
            raise RuntimeError("r_max should be at least R_max/2")

        print("  scaled coords:", R_range, r_max)

        self.R = xp.linspace(*R_range, args.NR)
        self.r = xp.linspace(r_max/args.Nr, r_max, args.Nr)

        # require Ng to be even
        if args.Nph % 2 != 0:
            raise RuntimeError(f"Ng must be even!")

        # N.B.: It is essential that we not include the endpoint in
        # gamma lest our cyclic grid be ill-formed and 2nd derivatives
        # all over the place
        self.th = xp.linspace(xp.pi/args.Nth, xp.pi-xp.pi/args.Nth, args.Nth, endpoint=True)
        #self.ph = xp.linspace(xp.pi/args.Nph, 2*xp.pi+xp.pi/args.Nph, args.Nph, endpoint=False)##XXXXX  check this        
        #self.th = xp.linspace(xp.pi/(2*(args.Nth+1)), xp.pi-xp.pi/(2*(args.Nth+1)), args.Nth, endpoint=True)
        #self.th = xp.linspace(xp.pi/(2*(args.Nth+1)), xp.pi+xp.pi/(2*(args.Nth+1)), args.Nth, endpoint=False)
        #sinthiinv = xp.sin(self.th)
        #print("th1",self.th)
        #print("sinthiinv",sinthiinv)
#
        #self.th = xp.linspace(xp.pi/(2*(args.Nth+1)), xp.pi-xp.pi/(2*(args.Nth+1)), args.Nth, endpoint=True)
        #sinthiinv = xp.sin(self.th)
        #print("th2",self.th)
        #print("sinthiinv",sinthiinv)
##
        #exit()

        self.ph = xp.linspace(0, 2*xp.pi, args.Nph, endpoint=False)
       
        self.axes = (self.R, self.r, self.ph, self.th)

        self.shape = (args.NR, args.Nr, args.Nph, args.Nth)
        self.boshape = (args.Nr, args.Nph, args.Nth)
        self.size = args.NR * args.Nr * args.Nph * args.Nth

        dR = self.R[1] - self.R[0]
        dr = self.r[1] - self.r[0]
        dth = self.th[1] - self.th[0]
        dph = self.ph[1] - self.ph[0] 

        self.P_R  = xp.fft.fftshift(xp.fft.fftfreq(args.NR, dR)) * 2 * xp.pi
        self.RP_grid = xp.meshgrid(self.R, self.P_R, indexing='ij')
        # N.B.: These all lack the factor of -1/(2 * mu)
        # We also are throwing away the returned jacobian of R/r
        #self.ddR2, _ = KE_Borisov(self.R, bare=True)
        self.ddR2    = KE(args.NR, dR, bare=True, cyclic=False)
        #self.ddr2, _ = KE_Borisov_3D(self.r, bare=True)
        #self.ddr1, _ = KE_Borisov_3D(self.r, bare=True, order=1)
        self.ddr2 = KE(args.Nr, dr, bare=True, cyclic=False)
        self.ddr1 = KE(args.Nr, dr, bare=True, cyclic=False, order=1)

        # Part of the reason for using a cyclic *stencil* for gamma
        # rather than KE_FFT is that it wasn't immediately obvious how
        # I would represent ∂/∂γ. (∂²/∂γ² was clear.)  N.B.: The
        # default stencil degree is 11
        self.ddth2 = KE(args.Nth, dth, bare=True, cyclic=True)
        self.ddth1 = KE(args.Nth, dth, bare=True, cyclic=True, order=1)

        self.ddph2 = KE(args.Nph, dph, bare=True, cyclic=True)
        self.ddph1 = KE(args.Nph, dph, bare=True, cyclic=True, order=1)
    
        self.R_grid, self.rb_grid, self.pb_grid, self.tb_grid = xp.meshgrid(self.R, self.r, self.ph, self.th, indexing='ij')
        self.r_grid, self.p_grid, self.t_grid,  = xp.meshgrid(self.r, self.ph, self.th, indexing='ij')
        self.Vgrid = self.V(self.R_grid, self.rb_grid, self.pb_grid, self.tb_grid)

        # since we need these in Hx; maybe fine to compute on the fly?
        self.rinv2 = 1.0/(self.r)**2

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

    def V(self, R, r, phi, theta, spitvals=False):

        mu12 = self.mu12
        M_1 = self.M_1
        M_2 = self.M_2

        kappa2 = r*R*xp.sin(theta)*xp.cos(phi)

        r1e2 = (r)**2 + (R)**2*(mu12/M_1)**2 - 2*kappa2*mu12/M_1
        r2e2 = (r)**2 + (R)**2*(mu12/M_2)**2 + 2*kappa2*mu12/M_2

        r1e = xp.sqrt(xp.where(r1e2 < 0, 0, r1e2))
        r2e = xp.sqrt(xp.where(r2e2 < 0, 0, r2e2))
        
        if spitvals == True:
            return r1e2,r2e2
        else:
            return self._Vfunc(R, r1e, r2e, (self.g_1, self.g_2))
            

    
    def compute_EPS(info):
    
        Rval, Pval, Htot_bo, gammacoeff_R, gammacoeff_phi, gammacoeff_theta, \
        Gammatotr, Gammatotp, Gammatott, Gammasqtotr, Gammasqtotp, Gammasqtott, mu12 = info
        
        #print("i,j",Rval,Pval,gammacoeff_R[Rval,Pval],flush=True)           
        
        Htot = Htot_bo[Rval]+(gammacoeff_R[Rval]*Gammatotr)+(gammacoeff_phi[Rval]*Gammatotp)+(gammacoeff_theta[Rval]*Gammatott)
        Htotsq = Htot - (Gammasqtotr +Gammasqtotp+ Gammasqtott)/(2*mu12)
        
        e_approx = xp.linalg.eigvalsh(Htot)
        e_approxsq = xp.linalg.eigvalsh(Htotsq)
        
        return Rval,Pval,e_approx[0],e_approxsq[0]

def Gamma_etf(R,r,phi,theta,ddr,ddph,ddth,M_1,M_2,mu12,r1e2,r2e2,*xdav):

    if len(xdav)==1:
        xdavx1 = xdavy1 = xdavz1 = xdavx2 = xdavy2 = xdavz2  = xdav[0]
    else:
        xdavx1, xdavy1, xdavz1, xdavx2, xdavy2, xdavz2 = xdav   

    Nth = len(ddth)
    Nr = len(ddr)
    Nph = len(ddph)
    
    theta1 = xp.exp(-r1e2)
    theta2 = xp.exp(-r2e2)
    partition = theta1 + theta2

    costheta = xp.diag(xp.cos(theta)[0,0,:])
    sintheta = xp.diag(xp.sin(theta)[0,0,:])
    cosphi = xp.diag(xp.cos(phi)[0,:,0])
    sinphi = xp.diag(xp.sin(phi)[0,:,0])

    costheta_diag = (xp.cos(theta)[0,0,:])
    sintheta_diag = (xp.sin(theta)[0,0,:])
    cosphi_diag = (xp.cos(phi)[0,:,0])
    sinphi_diag = (xp.sin(phi)[0,:,0])
    
    re = r[:,0,0]
    invr = xp.diag(1/re)

    t1 = theta1/partition
    t2 = theta2/partition

    t1pxr = xp.einsum('ikl,ij,k,l,Bjkl->Bikl',t1,(ddr-invr),cosphi_diag,sintheta_diag,xdavx1)
    t1pxt = xp.einsum('ijk,i,j,kl,Bijl->Bijk',t1,1/re,cosphi_diag,(costheta@ddth-xp.diag(costheta_diag/(2*sintheta_diag))),xdavx1)
    t1pxp = -xp.einsum('ijl,i,jk,l,Bikl->Bijl',t1,1/re,sinphi@ddph,1/sintheta_diag,xdavx1)
    t1px = t1pxr + t1pxt + t1pxp

    t2pxr = xp.einsum('ikl,ij,k,l,Bjkl->Bikl',t2,(ddr-invr),cosphi_diag,sintheta_diag,xdavx2)
    t2pxt = xp.einsum('ijk,i,j,kl,Bijl->Bijk',t2,1/re,cosphi_diag,(costheta@ddth-xp.diag(costheta_diag/(2*sintheta_diag))),xdavx2)
    t2pxp = -xp.einsum('ijl,i,jk,l,Bikl->Bijl',t2,1/re,sinphi@ddph,1/sintheta_diag,xdavx2)
    t2px = t2pxr + t2pxt + t2pxp

    t1pyr = xp.einsum('ikl,ij,k,l,Bjkl->Bikl',t1,(ddr-invr),sinphi_diag,sintheta_diag,xdavy1)
    t1pyt = xp.einsum('ijk,i,j,kl,Bijl->Bijk',t1,1/re,sinphi_diag,(costheta@ddth-xp.diag(costheta_diag/(2*sintheta_diag))),xdavy1)
    t1pyp = -xp.einsum('ijl,i,jk,l,Bikl->Bijl',t1,1/re,cosphi@ddph,1/sintheta_diag,xdavy1)
    t1py = t1pyr + t1pyt + t1pyp

    t2pyr = xp.einsum('ikl,ij,k,l,Bjkl->Bikl',t2,(ddr-invr),sinphi_diag,sintheta_diag,xdavy2)
    t2pyt = xp.einsum('ijk,i,j,kl,Bijl->Bijk',t2,1/re,sinphi_diag,(costheta@ddth-xp.diag(costheta_diag/(2*sintheta_diag))),xdavy2)
    t2pyp = -xp.einsum('ijl,i,jk,l,Bikl->Bijl',t2,1/re,cosphi@ddph,1/sintheta_diag,xdavy2)
    t2py = t2pyr + t2pyt + t2pyp

    t1pzr = xp.einsum('ilk,ij,k,Bjlk->Bilk', t1, (ddr-invr),costheta_diag,xdavz1)
    t1pzt = xp.einsum('ilj,i,jk,Bilk->Bilj', t1, 1/re,(xp.eye(Nth,Nth)-sintheta@ddth),xdavz1)
    t1pz = t1pzr + t1pzt

    t2pzr = xp.einsum('ilk,ij,k,Bjlk->Bilk', t2, (ddr-invr),costheta_diag,xdavz2)
    t2pzt = xp.einsum('ilj,i,jk,Bilk->Bilj', t2, 1/re,(xp.eye(Nth,Nth)-sintheta@ddth),xdavz2)
    t2pz = t2pzr + t2pzt
    
    pxrt1 = xp.einsum('ij,k,l,jkl,Bjkl->Bikl',(ddr-invr),cosphi_diag,sintheta_diag,t1,xdavx1)
    pxtt1 = xp.einsum('i,j,kl,ijl,Bijl->Bijk',1/re,cosphi_diag,(costheta@ddth-xp.diag(costheta_diag/(2*sintheta_diag))),t1,xdavx1)
    pxpt1 = -xp.einsum('i,jk,l,ikl,Bikl->Bijl',1/re,sinphi@ddph,1/sintheta_diag,t1, xdavx1)
    pxt1 = pxrt1 + pxtt1 + pxpt1

    pxrt2 = xp.einsum('ij,k,l,jkl,Bjkl->Bikl',(ddr-invr),cosphi_diag,sintheta_diag,t2,xdavx2)
    pxtt2 = xp.einsum('i,j,kl,ijl,Bijl->Bijk',1/re,cosphi_diag,(costheta@ddth-xp.diag(costheta_diag/(2*sintheta_diag))),t2,xdavx2)
    pxpt2 = -xp.einsum('i,jk,l,ikl,Bikl->Bijl',1/re,sinphi@ddph,1/sintheta_diag,t2, xdavx2)
    pxt2 = pxrt2 + pxtt2 + pxpt2

    pyrt1 = xp.einsum('ij,k,l,jkl,Bjkl->Bikl',(ddr-invr),sinphi_diag,sintheta_diag,t1,xdavy1)
    pytt1 = xp.einsum('i,j,kl,ijl,Bijl->Bijk',1/re,sinphi_diag,(costheta@ddth-xp.diag(costheta_diag/(2*sintheta_diag))),t1,xdavy1)
    pypt1 = -xp.einsum('i,jk,l,ikl,Bikl->Bijl',1/re,cosphi@ddph,1/sintheta_diag,t1,xdavy1)
    pyt1 = pyrt1 + pytt1 + pypt1

    pyrt2 = xp.einsum('ij,k,l,jkl,Bjkl->Bikl',(ddr-invr),sinphi_diag,sintheta_diag,t2,xdavy2)
    pytt2 = xp.einsum('i,j,kl,ijl,Bijl->Bijk',1/re,sinphi_diag,(costheta@ddth-xp.diag(costheta_diag/(2*sintheta_diag))),t2,xdavy2)
    pypt2 = -xp.einsum('i,jk,l,ikl,Bikl->Bijl',1/re,cosphi@ddph,1/sintheta_diag,t2,xdavy2)
    pyt2 = pyrt2 + pytt2 + pypt2

    pzrt1 = xp.einsum('ij,k,jlk,Bjlk->Bilk',(ddr-invr),costheta_diag,t1,xdavz1)
    pztt1 = xp.einsum('i,jk,ilk,Bilk->Bilj',1/re,(xp.eye(Nth,Nth)-sintheta@ddth),t1,xdavz1)
    pzt1 = pzrt1 + pztt1

    pzrt2 = xp.einsum('ij,k,jlk,Bjlk->Bilk',(ddr-invr),costheta_diag,t2,xdavz2)
    pztt2 = xp.einsum('i,jk,ilk,Bilk->Bilj',1/re,(xp.eye(Nth,Nth)-sintheta@ddth),t2,xdavz2)
    pzt2 = pzrt2 + pztt2
    
    gammaetf1x = -0.5*(t1px + pxt1)
    gammaetf1y = -0.5*(t1py + pyt1)
    gammaetf1z = -0.5*(t1pz + pzt1)

    gammaetf2x = -0.5*(t2px + pxt2)   
    gammaetf2y = -0.5*(t2py + pyt2)
    gammaetf2z = -0.5*(t2pz + pzt2)

    return gammaetf1x, gammaetf1y, gammaetf1z, gammaetf2x, gammaetf2y, gammaetf2z


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
    parser.add_argument('-r', dest="Nr", metavar="Nr", default=400, type=int)
    parser.add_argument('-theta', dest="Nth", metavar="Nth", default=250, type=int)
    parser.add_argument('-phi', dest="Nph", metavar="Nph", default=250, type=int)
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
    
    NR,Nr,Nph,Nth = H.shape
    Nelec = Nr*Nth*Nph 
    
    xdav = xp.random.rand(H.shape[1],H.shape[2],H.shape[3])
    xdot = xdav.flatten()

    def Tx(xdav):

        sinthiinv = 1/xp.sin(H.th)
        sinthiinv[xp.abs(sinthiinv)>1e6]=0       
        sinthiinvsq = xp.square(sinthiinv)
        xdav = xdav.reshape((-1,) + H.boshape)
        
        Hel_dav = -1/(2*H.mur)*(           
            xp.einsum('ij,Bjkl->Bikl',H.ddr2,xdav)\
            +xp.einsum('i,jk,Bilk->Bilj',H.rinv2,H.ddth2,xdav)\
            +xp.einsum('i,Bikj->Bikj',0.5*H.rinv2,xdav)
            +xp.einsum('i,j,Bikj->Bikj',0.25*H.rinv2,(xp.cos(H.th)**2/xp.sin(H.th)**2),xdav)
            +xp.einsum('i,j,kl,Bilj->Bikj',H.rinv2, sinthiinvsq, H.ddph2,xdav)
            )
        
        return Hel_dav.reshape(xdav.shape)
    
    def _preconditioner_naive(H, dx, e, x0, Ri):
        diagH = buildDiag(H,Ri)
        diagd = diagH - (e - 1e-5)
        return dx/diagd

    #print("check",H.ph)
    #print("check",1/xp.sin(H.ph)**2)
    #exit()
    #print("check3",(xp.diag(H.ddth2)[None,:,None]).shape)
    #print("check2",(1/xp.sin(H.p_grid)**2)[0,:,0])
    
    def buildDiag(H,Ri):
        rinv2 = 1/(H.r_grid)**2
        sinsqthinv = 1/xp.sin(H.t_grid)**2
        sinsqthinv[xp.abs(sinsqthinv)>1e6]=0
        #sinthinv = 1/xp.sin(H.th)
        sinthinv = 1/xp.sin(H.t_grid)
        sinthinv[xp.abs(sinthinv)>1e6]=0
        thbig = (H.ddth1@(xp.diag(xp.sin(H.th))@H.ddth1))

        ke  = xp.zeros([Nr,Nph,Nth])
        ke += xp.diag(H.ddr2)[:,None,None]
        ke += rinv2*xp.diag(H.ddth2)[None,None,:]
        ke += 0.5*H.rinv2[:,None,None]
        ke += 0.25*rinv2*((xp.cos(H.th)**2/(xp.sin(H.th)**2)))[None,None,:]
        ke += rinv2*sinsqthinv*xp.diag(H.ddph2)[None,:,None]
        ke *= -1 / (2*H.mur)
        diag = H.Vgrid[Ri] + ke
        return diag.ravel()

    ival = xp.zeros([NR,1])
    Ad_n = xp.zeros(NR)
    gammacoeff_R = -1j*H.P_R/H.mu12 
    gammacoeff_phi = -1j*(H.Pphi/H.R)/H.mu12
    gammacoeff_theta = +1j*(H.Ptheta/H.R)/H.mu12

    sinthiinv = 1/xp.sin(H.th)
    sinthiinv[xp.abs(sinthiinv)>1e6]=0
    sinthiinvsq = xp.square(sinthiinv)

    
    for i in range(NR):

        print("Atom Ri",i)
        diag = buildDiag(H,i)

        def Hbo_dav(xdav):
            x = xdav.reshape((-1,)+H.boshape)
            Hbodav = H.Vgrid[i]*x + Tx(x)
            return Hbodav.reshape(xdav.shape)
        
        guess_bo = xp.exp(-(H.Vgrid[i] - xp.min(H.Vgrid[i]))**2/27.211**2).ravel()
        
        with timer_ctx(f"Davidson of size {H.size}"):
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
                tol=1e-10,
            )
        print("Davidson:", e_approx)
        print(conv)
        Ad_n[i] = e_approx[0]
        ival[i,0] = e_approx[0]
        r1e2, r2e2 = H.V(H.R[i], H.r_grid, H.p_grid, H.t_grid, spitvals=True)
    
        for j in range(NR):

            print("i,j",i,j)

            def ps_ham(xdav):

                x = xdav.reshape((-1,)+H.boshape)

                Tx = -1/(2*H.mur)*(           
                    xp.einsum('ij,Bjkl->Bikl',H.ddr2,x)\
                    +xp.einsum('i,jk,Bilk->Bilj',H.rinv2,H.ddth2,x)\
                    +xp.einsum('i,Bikj->Bikj',0.5*H.rinv2,x)
                    +xp.einsum('i,j,Bikj->Bikj',0.25*H.rinv2,(xp.cos(H.th)**2/xp.sin(H.th)**2),x)
                    +xp.einsum('i,j,kl,Bilj->Bikj',H.rinv2, sinthiinvsq, H.ddph2,x)
                )
                
                gammaetf1x, gammaetf1y, gammaetf1z, gammaetf2x, gammaetf2y, gammaetf2z = Gamma_etf(H.R[i],H.r_grid,H.p_grid,H.t_grid,H.ddr1,H.ddph1,H.ddth1,H.M_1,H.M_2,H.mu12,r1e2,r2e2,x)
                gamma1x = gammaetf1x
                gamma2x = gammaetf2x
                gamma1y = gammaetf1y
                gamma2y = gammaetf2y
                gamma1z = gammaetf1z
                gamma2z = gammaetf2z
                Gammatotx = (H.M_2*gamma1x-H.M_1*gamma2x)/(H.M_1+H.M_2)
                Gammatoty = (H.M_2*gamma1y-H.M_1*gamma2y)/(H.M_1+H.M_2)
                Gammatotz = (H.M_2*gamma1z-H.M_1*gamma2z)/(H.M_1+H.M_2)

                gammasq1x, gammasq1y, gammasq1z, gammasq2x, gammasq2y, gammasq2z = Gamma_etf(H.R[i],H.r_grid,H.p_grid,H.t_grid,H.ddr1,H.ddph1,H.ddth1,H.M_1,H.M_2,H.mu12,r1e2,r2e2, gammaetf1x, gammaetf1y, gammaetf1z, gammaetf2x, gammaetf2y, gammaetf2z)
                gamma1x2x, gamma1y2y, gamma1z2z, gamma2x1x, gamma2y1y, gamma2z1z = Gamma_etf(H.R[i],H.r_grid,H.p_grid,H.t_grid,H.ddr1,H.ddph1,H.ddth1,H.M_1,H.M_2,H.mu12,r1e2,r2e2, gammaetf2x, gammaetf2y, gammaetf2z, gammaetf1x, gammaetf1y, gammaetf1z)
                
                Gammasqtotx = ((H.M_2**2*gammasq1x)+(H.M_1**2*gammasq2x)-(H.M_1*H.M_2*gamma1x2x)-(H.M_1*H.M_2*gamma2x1x))/(H.M_1+H.M_2)**2
                Gammasqtoty = ((H.M_2**2*gammasq1y)+(H.M_1**2*gammasq2y)-(H.M_1*H.M_2*gamma1y2y)-(H.M_1*H.M_2*gamma2y1y))/(H.M_1+H.M_2)**2
                Gammasqtotz = ((H.M_2**2*gammasq1z)+(H.M_1**2*gammasq2z)-(H.M_1*H.M_2*gamma1z2z)-(H.M_1*H.M_2*gamma2z1z))/(H.M_1+H.M_2)**2
                
                #Hbodav = H.Vgrid[i]*x + Tx
                Hbodav = H.Vgrid[i]*x + Tx + (gammacoeff_R[j]*Gammatotx)+(gammacoeff_phi[i]*Gammatoty)+(gammacoeff_theta[i]*Gammatotz)
                Htotsq = Hbodav - (Gammasqtotx +Gammasqtoty + Gammasqtotz)/(2*H.mu12) 

                return Htotsq.reshape(xdav.shape)

            guess_ps = xp.exp(-(H.Vgrid[i] - xp.min(H.Vgrid[i]))**2/27.211**2).ravel()
            with timer_ctx(f"Davidson of size {H.size}"):
                conv, e_approx, evecs = lib.davidson1(
                    ps_ham,
                    guess_ps,
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

