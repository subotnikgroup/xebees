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


# for equally spaced points; if unequal, pass J.
# tol specifies maximum mean Hermitian deviation
def KE_Borisov(x, tol=1e-6, mass=None, bare=False, order=2):
    # A. G. Borisov, J. Chem. Phys. 114, 7770–7777 (2001)
    # https://doi.org/10.1063/1.1358867

    N = len(x)
    x_max = x[-1]
    J = np.gradient(x) * N / x_max


    bound = lambda a, b: np.arange(a,b+1)
    al = lambda k: np.where((k == 0) | (k == N), 1/np.sqrt(2), 1)

    # Helper function to pre-compute sine and cosine matrices (Asin & Acos above)
    def DTT(N, func):
        k = bound(0, N)
        m = bound(1, N)
        return func(np.outer(2*m-1,k) * np.pi/N/2)

    COS = DTT(N, np.cos)
    SIN = DTT(N, np.sin)

    Ac = COS.T * al(bound(0,N))[:,np.newaxis]
    As = (SIN * al(bound(0,N))).T
    Acv = COS * al(bound(0,N))[np.newaxis, :] * (2/N)
    Asv = SIN * al(bound(0,N))[np.newaxis, :] * (2/N)

    F = np.copy(x)

    b = 1/np.sqrt(F * J)
    R = F / J
    k = np.arange(N+1) * np.pi / x[-1]

    if order == 2:  # L should be symmetric
        L = -b[:,None] * Acv * k @ As * R @ Asv * k @ Ac * b
        deviation = np.mean(np.abs(L-L.T))
        L = (L + L.T)/2
    elif order == 1:  # iL is Hermitian
        L = b[:,None] * (Acv * k @ As - Asv * k @ Ac) * b
        deviation = np.mean(np.abs(L+L.T))
        L = (L - L.T)/2
    else:
        raise RuntimeError(f"Borisov derivatives of order {order} not implemented!")


    if deviation > tol:
        raise RuntimeError("Deviation from Hermitian too large:", deviation)

    if not bare:
        L *= -1 / (2 * mass)

    return L, J

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

@timer
def Gamma_etf(R,r,g,pr,pg,M_1,M_2):
    """
    Gamma operator
    """
    print("building etf")
    mu12 = M_1*M_2/(M_1+M_2)
    sigma = 1
    Ng = len(pg)
    kappa2 = R*r*np.cos(g)    
    r1e = (r)**2 + (R)**2*(mu12/M_1)**2 - 2*kappa2*mu12/M_1
    re2 = (r)**2 + (R)**2*(mu12/M_2)**2 + 2*kappa2*mu12/M_2

    theta1 = np.exp(-r1e / sigma**2)
    theta2 = np.exp(-re2 / sigma**2)
    partition = theta1 + theta2
    
    t1 = np.diag((theta1/partition).ravel())
    t2 = np.diag((theta2/partition).ravel())
    
    pR = np.kron(-1j*pr,np.eye(Ng))
    pt = np.kron(np.diag(1/r[:,0]), -1j*pg)
    
    gamma1R = (t1 @ pR + pR @ t1) / (2j)
    gamma1t = (t1 @ pt + pt @ t1) / (2j)
    gamma2R = (t2 @ pR + pR @ t2) / (2j)
    gamma2t = (t2 @ pt + pt @ t2) / (2j)
    
    return gamma1R, gamma1t, gamma2R, gamma2t

@timer
def Gamma_erf(R,r,g,pr,pg,M_1,M_2):
    print("building erf")

    mu12 = M_1*M_2/(M_1+M_2)
    sigma = 1
    Ng = len(pg)
    Nr = len(pr)
    kappa2 = R*r*np.cos(g)    
    r1e = (r)**2 + (R)**2*(mu12/M_1)**2 - 2*kappa2*mu12/M_1
    re2 = (r)**2 + (R)**2*(mu12/M_2)**2 + 2*kappa2*mu12/M_2

    theta1 = np.exp(-r1e / sigma**2)
    theta2 = np.exp(-re2 / sigma**2)
    partition = theta1 + theta2
    t1 = np.diag((theta1/partition).ravel())
    t2 = np.diag((theta2/partition).ravel())
    
    pR = np.kron(-1j*pr,np.eye(Ng))
    pGdr = np.kron(np.diag(1/r[:,0]),-1j*pg)
    rcosg = np.kron(np.diag(r[:,0]),np.diag(np.cos(g[0,:])))
    rsing = np.kron(np.diag(r[:,0]),np.diag(np.sin(g[0,:])))

    
    J1 = -0.5j*((rcosg-(np.eye(Nr*Ng)*R*mu12/M_1))@(t1@pGdr+pGdr@t1)-rsing@((t1@pR+pR@t1)))
    J2 = -0.5j*((rcosg+(np.eye(Nr*Ng)*R*mu12/M_2))@(t2@pGdr+pGdr@t2)-rsing@((t2@pR+pR@t2)))

    #check signs
    gamma1 = -1/R*(J1+J2)
    gamma2 = 1/R*(J1+J2)
  
    return gamma1,gamma2
