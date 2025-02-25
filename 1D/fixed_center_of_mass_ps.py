import sys
import fcntl
import numpy as np

# Constants
amu_to_au = 1822.888486209
angstrom_to_bohr = 1.8897259886

def VO(R, r, g=1):
    """
    Potential energy in Hartree.
    """
    R_bohr = R / angstrom_to_bohr
    r_bohr = r / angstrom_to_bohr
    D = 60
    d = 0.95
    a = 2.52
    c = 1
    A = 2.32e5
    B = 3.15
    C = 2.31e4

    D1 = g * D * (np.exp(-2*a * (R_bohr / 2 + r_bohr - d)) 
                  - 2*np.exp(-a * (R_bohr / 2 + r_bohr - d)) 
                  + 1)
    D2 = D * c**2 * (np.exp(- (2*a / c) * (R_bohr / 2 - r_bohr - d))
                     - 2*np.exp(- (a / c) * (R_bohr / 2 - r_bohr - d)))
    return 0.00159362 * (D1 + D2 + A * np.exp(-B*R_bohr) - C / (R_bohr**6))

def get_stencil_coefficients(stencil_size, derivative_order):
    """
    Get finite-difference stencil coefficients
    """
    if stencil_size % 2 == 0:
        raise ValueError("Stencil size must be odd.")
    
    half_size = stencil_size // 2
    A = np.zeros((stencil_size, stencil_size))

    x_vals = np.arange(-half_size, half_size + 1)
    for i in range(stencil_size):
        A[i] = x_vals**i

    b = np.zeros(stencil_size)
    b[derivative_order] = np.math.factorial(derivative_order)

    # Solve for coefficients
    return np.linalg.solve(A, b)

def KE(n, dx, mass, stencil_size=7):
    """
    Kinetic energy operator
    """
    stencil = get_stencil_coefficients(stencil_size, 2) / dx**2
    I = np.eye(n)
    # Build T by convolving each basis vector with the stencil
    T = np.array([np.convolve(I[i], stencil, mode='same') for i in range(n)])
    T *= -1.0 / (2.0 * mass)
    return T

def PO(n, r, stencil_size=7):
    """
    First-order momentum operator 
    """
    dr = r[1] - r[0]
    stencil = get_stencil_coefficients(stencil_size, 1) / dr
    I = np.eye(n)
    pe = np.array([np.convolve(I[i], stencil, mode='same') for i in range(n)], dtype=complex)
    return -1j * pe

def PO2(n, r, stencil_size=7):
    """
    Second-order momentum operator (p^2 -> -d^2/dx^2).
    """
    dr = r[1] - r[0]
    stencil = get_stencil_coefficients(stencil_size, 2) / dr**2
    I = np.eye(n)
    pe2 = np.array([np.convolve(I[i], stencil, mode='same') for i in range(n)], dtype=complex)
    return -pe2

def Gamma(r, R_val, pe, sigma=1, w=1):
    """
    Gamma operator
    """
    theta1 = w * np.exp(-((r + R_val/2)**2) / sigma**2)
    theta2 = np.exp(-((r - R_val/2)**2) / sigma**2)
    partition = theta1 + theta2
    t1 = np.diag(theta1 / partition)
    t2 = np.diag(theta2 / partition)

    gamma1 = (t1 @ pe + pe @ t1) / (2j)
    gamma2 = (t2 @ pe + pe @ t2) / (2j)
    return (gamma1 - gamma2) / 2

def Gamma2(r, R_val, pe, pe2, sigma=1, w=1):
    """
    Second-order Gamma operator
    """
    theta1 = w * np.exp(-((r + R_val/2)**2) / sigma**2)
    theta2 = np.exp(-((r - R_val/2)**2) / sigma**2)
    partition = theta1 + theta2
    t1 = theta1 / partition
    t2 = theta2 / partition

    def get_gamma2(a, b, op1, op2):
        da = np.gradient(a, r)
        db = np.gradient(b, r)
        d2b = np.gradient(db, r)
        da_diag = np.diag(da)
        db_diag = np.diag(db)
        d2b_diag = np.diag(d2b)
        a_diag = np.diag(a)
        b_diag = np.diag(b)

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
    return (gamma11 - gamma12 - gamma21 + gamma22) / 4

def solve_EPS(NR, Nr, R, M, r, m, P, sigma=1, g=1, w=1):
    """
    Solve for EPS 
    """
    EPS = np.zeros((NR, NR))
    rPS = np.zeros((NR, NR), dtype=complex)
    pPS = np.zeros((NR, NR), dtype=complex)
    r2PS = np.zeros((NR, NR), dtype=complex)
    p2PS = np.zeros((NR, NR), dtype=complex)
    psiPS = np.zeros((NR, NR, Nr), dtype=complex)
    piPS = np.zeros((NR, NR), dtype=complex)
    pi2PS = np.zeros((NR, NR), dtype=complex)

    dr = r[1] - r[0]
    ke = KE(Nr, dr, m, stencil_size=7)
    pe = PO(Nr, r, stencil_size=7)
    pe2 = PO2(Nr, r, stencil_size=7)

    for i in range(NR):
        gamma = Gamma(r, R[i], pe, sigma, w)
        gamma2 = Gamma2(r, R[i], pe, pe2, sigma, w)
        v_diag = np.diag(VO(R[i], r, g))

        for j in range(NR):
            H = ke + v_diag - 1j * gamma * P[j] / M - gamma2 / (2 * M)
            val, vec = np.linalg.eigh(H)

            EPS[i, j] = val[0] + 0.5 * P[j]**2 / M

            # Expectation values for ground state
            gs = vec[:, 0]
            rPS[i, j] = gs.conj().T @ np.diag(r) @ gs
            r2PS[i, j] = gs.conj().T @ np.diag(r**2) @ gs

            psip = np.fft.fftshift(np.fft.fft(gs)) / np.sqrt(Nr)
            p_vals = np.fft.fftshift(np.fft.fftfreq(Nr, dr)) * 2.0 * np.pi
            pPS[i, j] = psip.conj().T @ np.diag(p_vals) @ psip
            p2PS[i, j] = psip.conj().T @ np.diag(p_vals**2) @ psip

            piPS[i, j] = P[j] - 1j * (gs.conj().T @ gamma @ gs)
            pi2PS[i, j] = (P[j]**2 
                           - 2j * P[j] * (gs.conj().T @ gamma @ gs)
                           - (gs.conj().T @ gamma2 @ gs))

            psiPS[i, j] = vec[:, 0]

    return EPS, rPS, pPS, r2PS, p2PS, psiPS, piPS, pi2PS

def inverse_weyl_transform(E, NR, R, P):
    """
    Perform the inverse Weyl transform 
    """
    HPS = np.zeros((NR, NR), dtype=complex)
    EPP = np.zeros((NR, NR), dtype=complex)
    EPS_half = np.zeros((NR + 1, NR), dtype=complex)
    dR = R[1] - R[0]
    R_half = np.linspace(R[0] - dR/2, R[-1] + dR/2, NR + 1)

    # Build EPP
    for i in range(NR):
        for j in range(NR):
            for k in range(NR):
                EPP[j, i] += np.exp(-1j * R[k] * P[j]) * E[k, i] / np.sqrt(NR)

    # Build EPS_half
    for i in range(NR):
        for j in range(NR + 1):
            for k in range(NR):
                EPS_half[j, i] += np.exp(1j * R_half[j] * P[k]) * EPP[k, i] / np.sqrt(NR)

    # Build HPS
    for j in range(NR):
        for q1 in range(NR):
            for q2 in range(NR):
                if (q1 - q2) % 2 == 0:
                    HPS[q1, q2] += (np.exp(-1j * (R[q1] - R[q2]) * P[j])
                                    * E[(q1 + q2) // 2, j] / NR)
                else:
                    idx = (q1 + q2 + 1) // 2
                    HPS[q1, q2] += (np.exp(-1j * (R[q1] - R[q2]) * P[j])
                                    * EPS_half[idx, j] / NR)
    return HPS

if __name__ == "__main__":
    M_val = float(sys.argv[1]) * amu_to_au
    identifier = sys.argv[2]
    sigma_val = float(sys.argv[3])
    output_dir = sys.argv[4]  
    NR = int(sys.argv[5]) + 1
    Nr = int(sys.argv[6])
    g_val = float(sys.argv[7])
    w_val = float(sys.argv[8])

    R = np.linspace(2, 4, NR) * angstrom_to_bohr
    r = np.linspace(-2, 2, Nr) * angstrom_to_bohr

    # Masses
    m = 1 * amu_to_au        
    M = M_val                
    mu = M * M / (2 * M)    
    
    dR = R[1] - R[0]
    P = np.fft.fftshift(np.fft.fftfreq(NR, dR)) * 2 * np.pi

    EPS, rPS, pPS, r2PS, p2PS, psiPS, piPS, pi2PS = solve_EPS(
        NR, Nr, R, mu, r, m, P, sigma_val, g_val, w_val
    )

    HPS = inverse_weyl_transform(EPS, NR, R, P)
    EPSv, psiPSv = np.linalg.eigh(HPS)
    
    with open(f"EPS_{identifier}.dat", "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        print(M_val, end=' ', file=f)
        print(" ".join(f"{E:.8f}" for E in EPSv), file=f)
        fcntl.flock(f, fcntl.LOCK_UN)
