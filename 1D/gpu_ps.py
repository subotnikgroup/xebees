import sys
import fcntl
#import numpy as xp

import os, sys
sys.path.append(os.path.abspath("lib"))
from hamiltonian import  inverse_weyl_transform
import potentials
from constants import *

import xp
import math

# Constants
amu_to_au = 1822.888486209
angstrom_to_bohr = 1.8897259886

def VO(R_au, r1e_au, r2e_au, charges):
    """
    Potential energy in Hartree.
    """

    Q1, Q2 = charges

    R   = R_au   / ANGSTROM_TO_BOHR
    r1e = r1e_au / ANGSTROM_TO_BOHR
    r2e = r2e_au / ANGSTROM_TO_BOHR

    D, d, a, c = 60, 0.95, 2.52, 1
    A, B, C = 2.32e5, 3.15, 2.31e4

    D2 = Q2 * D * (     xp.exp(-2*a * (r2e-d))
                    - 2*xp.exp(  -a * (r2e-d))
                    + 1)
    D1 = Q1 * D * c**2 * (     xp.exp(-(2*a/c) * (r1e-d))
                           - 2*xp.exp(-(  a/c) * (r1e-d)))

    VN = Q1 * Q2 * (A*xp.exp(-B*R) - C/R**6)

    return 0.00159362 * (D1 + D2 + VN)

def get_stencil_coefficients(stencil_size, derivative_order):
    """
    Get finite-difference stencil coefficients
    """
    if stencil_size % 2 == 0:
        raise ValueError("Stencil size must be odd.")
    
    half_size = stencil_size // 2
    A = xp.zeros((stencil_size, stencil_size))

    x_vals = xp.arange(-half_size, half_size + 1)
    for i in range(stencil_size):
        A[i] = x_vals**i

    b = xp.zeros(stencil_size)
    b[derivative_order] = math.factorial(derivative_order)

    # Solve for coefficients
    return xp.linalg.solve(A, b)

def KE(n, dx, mass, stencil_size=7):
    """
    Kinetic energy operator
    """
    stencil = get_stencil_coefficients(stencil_size, 2) / dx**2
    I = xp.eye(n)
    # Build T by convolving each basis vector with the stencil
    T = xp.array([xp.convolve(I[i], stencil, mode='same') for i in range(n)])
    T *= -1.0 / (2.0 * mass)
    return T

def PO(n, r, stencil_size=7):
    """
    First-order momentum operator 
    """
    dr = r[1] - r[0]
    stencil = get_stencil_coefficients(stencil_size, 1) / dr
    I = xp.eye(n)
    pe = xp.array([xp.convolve(I[i], stencil, mode='same') for i in range(n)], dtype=complex)
    return -1j * pe

def PO2(n, r, stencil_size=7):
    """
    Second-order momentum operator (p^2 -> -d^2/dx^2).
    """
    dr = r[1] - r[0]
    stencil = get_stencil_coefficients(stencil_size, 2) / dr**2
    I = xp.eye(n)
    pe2 = xp.array([xp.convolve(I[i], stencil, mode='same') for i in range(n)], dtype=complex)
    return -pe2

def Gamma(r, R_val, pe, M1, M2, sigma=1, w=1):
    """
    Gamma operator
    """
    mu12 = M1*M2/(M1+M2)
    #r1e2, r2e2 = H.V(H.R[i], H.x_grid, H.y_grid, H.z_grid, spitvals=True)
    kappa2 = r*R_val
    r1e2 = r**2 + (R_val)**2*(mu12/M1)**2 - 2*kappa2*mu12/M1
    r2e2 = r**2 + (R_val)**2*(mu12/M2)**2 + 2*kappa2*mu12/M2
    theta1 = xp.exp(-r1e2)
    theta2 = xp.exp(-r2e2)
    partition = theta1 + theta2
    t1 = xp.diag(theta1 / partition)
    t2 = xp.diag(theta2 / partition)

    gamma1 = (t1 @ pe + pe @ t1) / (2j)
    gamma2 = (t2 @ pe + pe @ t2) / (2j)
    return (M2*gamma1 - M1*gamma2) / (M1 + M2)

def Gamma2(r, R_val, pe, pe2, M1, M2, sigma=1, w=1):
    """
    Second-order Gamma operator
    """

    kappa2 = r*R_val
    r1e2 = r**2 + (R_val)**2*(mu12/M1)**2 - 2*kappa2*mu12/M1
    r2e2 = r**2 + (R_val)**2*(mu12/M2)**2 + 2*kappa2*mu12/M2
    theta1 = xp.exp(-r1e2)
    theta2 = xp.exp(-r2e2)
    partition = theta1 + theta2
    t1 = theta1 / partition
    t2 = theta2 / partition

    def get_gamma2(a, b, op1, op2):
        da = xp.gradient(a, r)
        db = xp.gradient(b, r)
        d2b = xp.gradient(db, r)
        da_diag = xp.diag(da)
        db_diag = xp.diag(db)
        d2b_diag = xp.diag(d2b)
        a_diag = xp.diag(a)
        b_diag = xp.diag(b)

        a1 = -1j * a_diag @ db_diag @ op1 + a_diag @ b_diag @ op2
        a2 = -a_diag @ d2b_diag - 2j * a_diag @ db_diag @ op1 + a_diag @ b_diag @ op2
        a3 = -1j * da_diag @ b_diag @ op1 - 1j * a_diag @ db_diag @ op1 + a_diag @ b_diag @ op2
        a4 = (-da_diag @ db_diag - 1j * da_diag @ b_diag @ op1 - a_diag @ d2b_diag
              - 2j * a_diag @ db_diag @ op1 + a_diag @ b_diag @ op2)

        return -0.25 * (a1 + a2 + a3 + a4)

    gamma11 = get_gamma2(t1, t1, pe, pe2)
    gamma12 = get_gamma2(t1, t2, pe, pe2)
    gamma21 = get_gamma2(t2, t1, pe, pe2)
    gamma22 = get_gamma2(t2, t2, pe, pe2)
    return (M2**2*gamma11 - M1*M2*gamma12 - M1*M2*gamma21 + M2**2*gamma22) / (M1 + M2)**2

def Gammasq(r, R_val, M1, M2, pe2, sigma=1, w=1):
    """
    Second-order Gamma operator
    """
    mu12 = M1*M2/(M1+M2)
    #r1e2, r2e2 = H.V(H.R[i], H.x_grid, H.y_grid, H.z_grid, spitvals=True)
    kappa2 = r*R_val
    r1e2 = r**2 + (R_val)**2*(mu12/M1)**2 - 2*kappa2*mu12/M1
    r2e2 = r**2 + (R_val)**2*(mu12/M2)**2 + 2*kappa2*mu12/M2
    theta1 = xp.exp(-r1e2)
    theta2 = xp.exp(-r2e2)
    partition = theta1 + theta2
    t1 = xp.diag(theta1 / partition)
    t2 = xp.diag(theta2 / partition)

    gamma1 = (t1 @ pe2 + pe2 @ t1) / (2j)
    gamma2 = (t2 @ pe2 + pe2 @ t2) / (2j)

    return (M2*gamma1**2 - M1*gamma2**2) / (M1 + M2)**2

def solve_EPS(NR, Nr, R, r, Mtotal,mu12, mur, M1, M2, P, charges):
    """
    Solve for EPS 
    """
    EPS = xp.zeros((NR, NR))
    rPS = xp.zeros((NR, NR), dtype=complex)
    pPS = xp.zeros((NR, NR), dtype=complex)
    r2PS = xp.zeros((NR, NR), dtype=complex)
    p2PS = xp.zeros((NR, NR), dtype=complex)
    EPSsquare = xp.zeros((NR, NR))
    pPS_square = xp.zeros((NR, NR), dtype=complex)
    p2PS_square = xp.zeros((NR, NR), dtype=complex)
    psiPS_square = xp.zeros((NR, NR, Nr), dtype=complex)
    piPS_square = xp.zeros((NR, NR), dtype=complex)
    pi2PS_square = xp.zeros((NR, NR), dtype=complex)
    EBO = xp.zeros([NR,1])
    pBO = xp.zeros([NR,1],dtype=complex)
    p2BO = xp.zeros([NR,1],dtype=complex)
    EPSsquare_new = xp.zeros((NR, NR))
    pPS_square_new = xp.zeros((NR, NR), dtype=complex)
    p2PS_square_new = xp.zeros((NR, NR), dtype=complex)
    sigma = 1
    g = 1
    w = 1


   
    dr = r[1] - r[0]
    ke = KE(Nr, dr, m, stencil_size=7)
    pe = PO(Nr, r, stencil_size=7)
    pe2 = PO2(Nr, r, stencil_size=7)

    for i in range(NR):
        if i % 10 == 0: print("i",i,flush=True)
        gamma = Gamma(r, R[i], pe, M1, M2, sigma, w)
        gamma2 = Gamma2(r, R[i], pe, pe2, M1, M2, sigma, w)
        gammasq = Gammasq(r, R[i], M1, M2, pe2, sigma, w)
        
        kappa2 = r*R[i]

        r1e2 = r**2 + (R[i])**2*(mu12/M1)**2 - 2*kappa2*mu12/M1
        r2e2 = r**2 + (R[i])**2*(mu12/M2)**2 + 2*kappa2*mu12/M2

        r1e = xp.sqrt(xp.where(r1e2 < 0, 0, r1e2))
        r2e = xp.sqrt(xp.where(r2e2 < 0, 0, r2e2))
        v_diag = xp.diag(VO(R[i], r1e, r2e, charges))

        Helbo = ke + v_diag
        valbo, vecbo = xp.linalg.eigh(Helbo)
        EBO[i] = valbo[0]
        #print("EBO",EBO[i],flush=True)
        gsbo = vecbo[:, 0]
        pbo = xp.conj(gsbo).T @ (pe @ gsbo)
        p2bo = xp.conj(gsbo).T @ (pe2 @ gsbo)
        pBO[i] = pbo
        p2BO[i] = p2bo
        
        for j in range(NR):
            #print("i,j",i,j,flush=True)
            Hel_square = ke + v_diag - 1j * gamma * P[j] / mu12 - gamma2 / (2 * mu12)
            Hel = ke + v_diag - 1j * gamma * P[j] / mu12
            Hel_square_new = ke + v_diag - 1j * gamma * P[j] / mu12 - gammasq / (2 * mu12)

            val_square, vec_square = xp.linalg.eigh(Hel_square)
            val, vec = xp.linalg.eigh(Hel)
            val_square_new, vec_square_new = xp.linalg.eigh(Hel_square_new)

            EPSsquare[i, j] = val_square[0] + 0.5 * P[j]**2 / mu12
            EPS[i,j] = val[0] + 0.5 * P[j]**2 / mu12
            EPSsquare_new[i, j] = val_square_new[0] + 0.5 * P[j]**2 / mu12

            # Expectation values for ground state
            gs_square = vec_square[:, 0]
            gs = vec[:, 0]
            gs_square_new = vec_square_new[:, 0]
            #rPS_square[i, j] = gs_square.conj().T @ xp.diag(r) @ gs_square
            #r2PS_square[i, j] = gs_square.conj().T @ xp.diag(r**2) @ gs_square
            #rPS[i, j] = gs.conj().T @ xp.diag(r) @ gs
            #r2PS[i, j] = gs.conj().T @ xp.diag(r**2) @ gs

            psip = xp.fft.fftshift(xp.fft.fft(gs)) / xp.sqrt(Nr)
            p_vals = xp.fft.fftshift(xp.fft.fftfreq(Nr, dr)) * 2.0 * xp.pi
            pPS[i, j] = psip.conj().T @ xp.diag(p_vals) @ psip
            p2PS[i, j] = psip.conj().T @ xp.diag(p_vals**2) @ psip

            psip_square = xp.fft.fftshift(xp.fft.fft(gs_square)) / xp.sqrt(Nr)           
            pPS_square[i, j] = psip_square.conj().T @ xp.diag(p_vals) @ psip_square
            p2PS_square[i, j] = psip_square.conj().T @ xp.diag(p_vals**2) @ psip_square

            psip_square_new = xp.fft.fftshift(xp.fft.fft(gs_square_new)) / xp.sqrt(Nr)
            pPS_square_new[i, j] = psip_square_new.conj().T @ xp.diag(p_vals) @ psip_square_new
            p2PS_square_new[i, j] = psip_square_new.conj().T @ xp.diag(p_vals**2) @ psip_square_new

            #piPS[i, j] = P[j] - 1j * (gs.conj().T @ gamma @ gs)
            #pi2PS[i, j] = (P[j]**2 
            #               - 2j * P[j] * (gs.conj().T @ gamma @ gs)
            #               - (gs.conj().T @ gamma2 @ gs))
#
            #psiPS[i, j] = vec[:, 0]

    KE_BO = KE(NR, dR, mu12, stencil_size=7)

    #return EPS, EPSsquare, pPS, p2PS, pPS_square, p2PS_square
    return EBO, EPS, EPSsquare, EPSsquare_new, pPS, p2PS, pPS_square, p2PS_square, pPS_square_new, p2PS_square_new, pBO, p2BO, KE_BO



if __name__ == "__main__":
    M1 = float(sys.argv[1]) * amu_to_au
    M2 = float(sys.argv[2]) * amu_to_au
    #identifier = sys.argv[2]
    #sigma_val = float(sys.argv[3])
    #output_dir = sys.argv[4]  
    NR = int(sys.argv[3]) + 1
    Nr = int(sys.argv[4])
    Q1 = float(sys.argv[5])
    Q2 = float(sys.argv[6])
    #g_val = float(sys.argv[7])
    #w_val = float(sys.argv[8])
    sigma_val = 1
    g_val = 1
    w_val = 1
    xp.backend = 'cupy'

    m = 1 * amu_to_au        
    Mtotal = M1 + M2                
    mu12 = M1 * M2 / (M1 + M2)    
    mur = (M1 + M2) * m / (M1 + M2 + m)

    extent = potentials.extents_borgis(mu12)
    print("extent",extent)
    R_min = extent[0]
    R_max = extent[1]
    r_min = -extent[2]
    r_max = extent[2]
    R = xp.linspace(R_min, R_max, NR)
    r = xp.linspace(r_min, r_max, Nr)

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

    # Masses
    
    
    dR = R[1] - R[0]
    P = xp.fft.fftshift(xp.fft.fftfreq(NR, dR)) * 2 * xp.pi
    charges = (Q1, Q2)

    #EPS, EPSsquare, pPS, p2PS, pPS_square, p2PS_square= solve_EPS(NR, Nr, R, r, Mtotal,mu12, mur, M1, M2, P, charges)
    EBO, EPS, EPSsquare, EPSsquare_new, pPS, p2PS, pPS_square, p2PS_square, pPS_square_new, p2PS_square_new, pBO, p2BO, KE_BO = solve_EPS(NR, Nr, R, r, Mtotal,mu12, mur, M1, M2, P, charges)
    
    HPS = inverse_weyl_transform(EPS, NR, R, P)
    HpPS = inverse_weyl_transform(pPS, NR, R, P)
    Hp2PS = inverse_weyl_transform(p2PS, NR, R, P)
    EPSv, psiPSv = xp.linalg.eigh(HPS)
    v_p00 = xp.conj(psiPSv[:,0]).T @ (HpPS @ psiPSv[:,0])
    v_p01 = xp.conj(psiPSv[:,1]).T @ (HpPS @ psiPSv[:,0])
    v_p200 = xp.conj(psiPSv[:,0]).T @ (Hp2PS @ psiPSv[:,0])
    v_p201 = xp.conj(psiPSv[:,1]).T @ (Hp2PS @ psiPSv[:,0])


    HPS_square = inverse_weyl_transform(EPSsquare, NR, R, P)
    HpPS_square = inverse_weyl_transform(pPS_square, NR, R, P)
    Hp2PS_square = inverse_weyl_transform(p2PS_square, NR, R, P)
    EPSv_square, psiPSv_square = xp.linalg.eigh(HPS_square)
    v_p00_square = xp.conj(psiPSv_square[:,0]).T @ (HpPS_square @ psiPSv_square[:,0])
    v_p01_square = xp.conj(psiPSv_square[:,1]).T @ (HpPS_square @ psiPSv_square[:,0])
    v_p200_square = xp.conj(psiPSv_square[:,0]).T @ (Hp2PS_square @ psiPSv_square[:,0])
    v_p201_square = xp.conj(psiPSv_square[:,1]).T @ (Hp2PS_square @ psiPSv_square[:,0])

    HPS_square_new = inverse_weyl_transform(EPSsquare_new, NR, R, P)
    HpPS_square_new = inverse_weyl_transform(pPS_square_new, NR, R, P)
    Hp2PS_square_new = inverse_weyl_transform(p2PS_square_new, NR, R, P)
    EPSv_square_new, psiPSv_square_new = xp.linalg.eigh(HPS_square_new)
    v_p00_square_new = xp.conj(psiPSv_square_new[:,0]).T @ (HpPS_square_new @ psiPSv_square_new[:,0])
    v_p01_square_new = xp.conj(psiPSv_square_new[:,1]).T @ (HpPS_square_new @ psiPSv_square_new[:,0])
    v_p200_square_new = xp.conj(psiPSv_square_new[:,0]).T @ (Hp2PS_square_new @ psiPSv_square_new[:,0])
    v_p201_square_new = xp.conj(psiPSv_square_new[:,1]).T @ (Hp2PS_square_new @ psiPSv_square_new[:,0])
    
    print("EPSv",EPSv[0:10])
    print("EPSv_square",EPSv_square[0:10])
    print("EPSv_square_new",EPSv_square_new[0:10])
    print("v_p00",v_p00)
    print("v_p00_square",v_p00_square)
    print("v_p00_square_new",v_p00_square_new)
    print("v_p01",v_p01)
    print("v_p01_square",v_p01_square)
    print("v_p01_square_new",v_p01_square_new)
    print("v_p200",v_p200)
    print("v_p200_square",v_p200_square)
    print("v_p200_square_new",v_p200_square_new)
    print("v_p201",v_p201)
    print("v_p201_square",v_p201_square)
    print("v_p201_square_new",v_p201_square_new)
    

    Rval, Pval = xp.meshgrid(R, P, indexing='ij')
    EBO_check = xp.zeros((NR, NR))
    Helmatg = xp.repeat(EBO,NR,axis=1)
    EBO_check += Helmatg   
    EBO_check += 1/(2*mu12)*(Pval**2)
    HBO_check = inverse_weyl_transform(EBO_check, NR, R, P)
    EBO_weyl_check, vecBO_weyl_check = xp.linalg.eigh(HBO_check)

    #HBO = KE_BO + xp.diag(EBO)
    #EBO_stencil, vecBO_stencil = xp.linalg.eigh(HBO)
    #print("EBO",EBO_stencil[0:10])
    print("EBO_check",EBO_weyl_check[0:10])

    pBORP = xp.repeat(pBO,NR,axis=1)
    p2BORP = xp.repeat(p2BO,NR,axis=1)

    HpBO = inverse_weyl_transform(pBORP, NR, R, P)
    Hp2BO = inverse_weyl_transform(p2BORP, NR, R, P)
    #v_pBO_stencil = xp.conj(vecBO_stencil[:,0]).T @ (HpBO @ vecBO_stencil[:,0])
    #v_p2BO_stencil = xp.conj(vecBO_stencil[:,0]).T @ (Hp2BO @ vecBO_stencil[:,0])

    v_pBO_check = xp.conj(vecBO_weyl_check[:,0]).T @ (HpBO @ vecBO_weyl_check[:,0])
    v_p2BO_check = xp.conj(vecBO_weyl_check[:,0]).T @ (Hp2BO @ vecBO_weyl_check[:,0])

    #print("v_pBO",v_pBO_stencil)
    #print("v_p2BO",v_p2BO_stencil)
    print("v_pBO_check",v_pBO_check)
    print("v_p2BO_check",v_p2BO_check)

    
    #with open(f"EPS_{identifier}.dat", "a") as f:
    #    fcntl.flock(f, fcntl.LOCK_EX)
    #    print(M_val, end=' ', file=f)
    #    print(" ".join(f"{E:.8f}" for E in EPSv), file=f)
    #    fcntl.flock(f, fcntl.LOCK_UN)
