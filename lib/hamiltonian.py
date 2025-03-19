import numpy as np
from scipy.special import factorial
from scipy.signal import convolve

def get_stencil_coefficients(stencil_size, derivative_order):
    if stencil_size % 2 == 0:
        raise ValueError("Stencil size must be odd.")
    
    half_size = stencil_size // 2
    A = np.vander(np.arange(-half_size, half_size + 1), increasing=True).T
    b = np.zeros(stencil_size)
    b[derivative_order] = factorial(derivative_order)
    
    return np.linalg.solve(A, b)

def KE(N, dx, mass, stencil_size=11, order=2):
    stencil = get_stencil_coefficients(stencil_size, order) / dx**order
    T = -1 / (2 * mass) * np.array(
        [convolve(e, stencil, mode='same') for e in np.eye(N)]
    )
    
    return T

def KE_FFT(N, P, R, mass): 
    Tp = np.diag(P**2 / (2 * mass))
    exp_RP = np.exp(1j * np.outer(P, R))
    
    return (exp_RP.T.conj() @ Tp @ exp_RP) / N


def solve_BO_surface(Tr, V):
    return np.array(
       [np.linalg.eigvalsh(Tr + np.diag(v))[0] for v in V])

def solve_BO_surfaces(Tr, V):
    return np.array(
       [np.linalg.eigvalsh(Tr + np.diag(v)) for v in V]).T


# print(
#     solve_BOv(
#         KE(NR, dR, M),
#         KE(Nr, dr, m),
#         VO(*np.meshgrid(R, r, indexing='ij')
#     )
def solve_BOv(TR, Tr, V):
    return np.linalg.eigvalsh(TR + np.diag(solve_BO_surface(Tr,V)))
