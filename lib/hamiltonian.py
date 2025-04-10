import numpy as np
from scipy.special import factorial
import scipy.signal as ssg
import scipy.ndimage as snd

def get_stencil_coefficients(stencil_size, derivative_order):
    if stencil_size % 2 == 0:
        raise ValueError("Stencil size must be odd.")
    
    half_size = stencil_size // 2
    A = np.vander(np.arange(-half_size, half_size + 1), increasing=True).T
    b = np.zeros(stencil_size)
    b[derivative_order] = factorial(derivative_order)
    
    return np.linalg.solve(A, b)

def KE(N, dx, mass=None, stencil_size=11, order=2, cyclic=False, bare=False):
    stencil = get_stencil_coefficients(stencil_size, order) / dx**order
    if cyclic:
        T = np.array(
            [snd.convolve(e, stencil, mode='wrap') for e in np.eye(N)]
        )

    else:
        T = np.array(
            [ssg.convolve(e, stencil, mode='same') for e in np.eye(N)]
        )

    if not bare:
        T *= -1 / (2 * mass)

    return T

def KE_FFT(N, P, R, mass): 
    Tp = np.diag(P**2 / (2 * mass))
    exp_RP = np.exp(1j * np.outer(P, R))
    
    return (exp_RP.T.conj() @ Tp @ exp_RP) / N


def KE_ColbertMiller_zero_inf(N, dx, mass=None, bare=False):
    T = np.zeros((N, N))

    # since we do not include the 0 point i->i+1; i+j-> i+j+2
    for i in range(N):
        for j in range(N):
            if i == j:
                T[i,i] = np.pi**2/3 - 1/2/(i+1)**2
            else:
                T[i,j] = (-1)**(i-j) * (2/(i-j)**2 - 2/(i+j+2)**2)

    if bare:
        T *= -1
    else:
        T *= 1 / (2 * mass)

    return T / dx**2


def KE_ColbertMiller_ab(N, dx, mass=None, bare=False):
    T = np.zeros((N, N))

    # since we do not include the 0 point i->i+1; i+j-> i+j+2
    for i in range(N):
        for j in range(N):
            if i == j:  # A6b
                T[i,i] = (2*(N+1)**2+1)/3 - 1/np.sin(np.pi*(i+1)/(N+1))**2
            else:       # A6a
                T[i,j] = (-1)**(i-j) * (1/np.sin(np.pi*(i-j  )/2/(N+1))**2 -
                                        1/np.sin(np.pi*(i+j+2)/2/(N+1))**2)

    T *= np.pi**2/(N*dx)**2/2

    if bare:
        T *= -1
    else:
        T *= 1 / (2 * mass)

    return T


def KE_FFT_cutoff(N, dx, ecut=30, mass=None, bare=False, cyclic=True):
    if not cyclic:
        raise RuntimeError("Noncyclic KE not implemented; think grid doubling!")

    #kgrid = np.fft.fftshift(np.fft.fftfreq(N, dx)) * 2 * np.pi
    kgrid = np.fft.fftfreq(N, dx) * 2 * np.pi
    k2 = [k**2 for k in kgrid]

    k2cut = np.minimum(k2, ecut*np.ones(N))

    T = np.zeros((N,N))
    for i in range(1,N+1):
        bi = np.zeros(N)
        bi[i-1]= 1
        bk = k2cut*np.fft.fft(bi)
        T[:,i-1] = np.fft.ifft(bk).real

    if bare:
        T *= -1
    else:
        T *= 1 / (2 * mass)

    return T

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
