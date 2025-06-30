import xp
from scipy.special import factorial
import numpy
from debug import timer
import nvtx

def get_stencil_coefficients(stencil_size, derivative_order):
    if stencil_size % 2 == 0:
        raise ValueError("Stencil size must be odd.")
    
    half_size = stencil_size // 2
    A = xp.vander(xp.arange(-half_size, half_size + 1.0), increasing=True).T
    b = xp.zeros(stencil_size)
    b[derivative_order] = factorial(derivative_order)
    return xp.linalg.solve(A, b)


def KE(N, dx, mass=None, stencil_size=11, order=2, cyclic=False, bare=False):
    stencil = get_stencil_coefficients(stencil_size, order) / dx**order
    center = stencil_size // 2

    if cyclic:
        fft_size = N
        eye = xp.eye(N)

    else:
        # zero-pad to next power of 2
        fft_size = int(2 ** numpy.ceil(numpy.log2(N + stencil_size - 1)))
        eye = xp.zeros((N, fft_size))
        eye[xp.arange(N), xp.arange(N)] = 1.0

    stencil_k = xp.zeros(fft_size, dtype=xp.complex128)
    stencil_k[:stencil_size] = stencil
    stencil_k = xp.roll(stencil_k, -center)
    stencil_k = xp.fft.fft(stencil_k)

    T = xp.fft.ifft(stencil_k * xp.fft.fft(eye)).real[:, :N]

    if not bare:
        T *= -1 / (2 * mass)

    return T


def KE_FFT(N, P, R): 
    Tp = xp.diag(-P**2)
    exp_RP = xp.exp(1j * xp.outer(P, R))
    
    return (exp_RP.T.conj() @ Tp @ exp_RP) / N

def gamma_grad(N, P, R): 
    Tp = xp.diag(-1j*P)
    exp_RP = xp.exp(1j * xp.outer(P, R))
    
    return (exp_RP.T.conj() @ Tp @ exp_RP) / N


# for equally spaced points; if unequal, pass J.
# tol specifies maximum mean Hermitian deviation
def KE_Borisov(x, tol=1e-6, mass=None, bare=False, order=2):
    # A. G. Borisov, J. Chem. Phys. 114, 7770–7777 (2001)
    # https://doi.org/10.1063/1.1358867

    N = len(x)
    x_max = x[-1]
    g = xp.gradient(x)
    g = g[0] if isinstance(g, tuple) else g
    J = g * N / x_max
    #J = xp.gradient(x) * N / x_max


    bound = lambda a, b: xp.arange(a,b+1)
    al = lambda k: xp.where((k == 0) | (k == N), 1/numpy.sqrt(2), 1)

    # Helper function to pre-compute sine and cosine matrices (Asin & Acos above)
    def DTT(N, func):
        k = bound(0, N)
        m = bound(1, N)
        return func(xp.outer(2*m-1,k) * xp.pi/N/2)

    COS = DTT(N, xp.cos)
    SIN = DTT(N, xp.sin)

    Ac = COS.T * al(bound(0,N))[:,xp.newaxis]
    As = (SIN * al(bound(0,N))).T
    Acv = COS * al(bound(0,N))[xp.newaxis, :] * (2/N)
    Asv = SIN * al(bound(0,N))[xp.newaxis, :] * (2/N)

    F = x

    b = 1/xp.sqrt(F * J)
    R = F / J
    k = xp.arange(N+1) * xp.pi / x[-1]

    if order == 2:  # L should be symmetric
        L = -b[:,None] * Acv * k @ As * R @ Asv * k @ Ac * b
        deviation = xp.mean(xp.abs(L-L.T))
        L = (L + L.T)/2
    elif order == 1:  # iL is Hermitian
        L = b[:,None] * (Acv * k @ As - Asv * k @ Ac) * b
        deviation = xp.mean(xp.abs(L+L.T))
        L = (L - L.T)/2
    else:
        raise RuntimeError(f"Borisov derivatives of order {order} not implemented!")


    if deviation > tol:
        raise RuntimeError("Deviation from Hermitian too large:", deviation)

    if not bare:
        L *= -1 / (2 * mass)

    return L, J

def KE_ColbertMiller_zero_inf(N, dx, mass=None, bare=False):
    T = xp.zeros((N, N))

    # since we do not include the 0 point i->i+1; i+j-> i+j+2
    for i in range(N):
        for j in range(N):
            if i == j:
                T[i,i] = xp.pi**2/3 - 1/2/(i+1)**2
            else:
                T[i,j] = (-1)**(i-j) * (2/(i-j)**2 - 2/(i+j+2)**2)

    if bare:
        T *= -1
    else:
        T *= 1 / (2 * mass)

    return T / dx**2

def KE_ColbertMiller_ab(N, dx, mass=None, bare=False):
    T = xp.zeros((N, N))

    # since we do not include the 0 point i->i+1; i+j-> i+j+2
    for i in range(N):
        for j in range(N):
            if i == j:  # A6b
                T[i,i] = (2*(N+1)**2+1)/3 - 1/xp.sin(xp.pi*(i+1)/(N+1))**2
            else:       # A6a
                T[i,j] = (-1)**(i-j) * (1/xp.sin(xp.pi*(i-j  )/2/(N+1))**2 -
                                        1/xp.sin(xp.pi*(i+j+2)/2/(N+1))**2)

    T *= xp.pi**2/(N*dx)**2/2

    if bare:
        T *= -1
    else:
        T *= 1 / (2 * mass)

    return T


def KE_FFT_cutoff(N, dx, ecut=xp.inf, mass=None, bare=False, cyclic=True, order=2):
    if not cyclic:
        raise RuntimeError("Noncyclic KE not implemented; think grid doubling!")

    kgrid = xp.fft.fftfreq(N, dx) * 2 * xp.pi
    k2 = [k**2 for k in kgrid]
    # NB: note cut off will only be applied to order=2 derivatives!
    k2cut = xp.minimum(k2, ecut*xp.ones(N))

    T = xp.zeros((N,N))
    for i in range(1,N+1):
        bi = xp.zeros(N)
        bi[i-1]= 1
        if order==2:
            bk = k2cut*xp.fft.fft(bi)
        elif order==1:
            bk = (0+1j)*kgrid*xp.fft.fft(bi)
        else:
            raise RuntimeError("order=1,2 are only valid orders")

        T[:,i-1] = xp.fft.ifft(bk).real

    if bare:
        T *= -1
    else:
        T *= 1 / (2 * mass)

    return T

def solve_BO_surface(Tr, V):
    return xp.asarray(
       [xp.linalg.eigvalsh(Tr + xp.diag(v))[0] for v in V])

def solve_BO_surfaces(Tr, V):
    return xp.asarray(
       [xp.linalg.eigvalsh(Tr + xp.diag(v)) for v in V]).T


# print(
#     solve_BOv(
#         KE(NR, dR, M),
#         KE(Nr, dr, m),
#         VO(*xp.meshgrid(R, r, indexing='ij')
#     )
def solve_BOv(TR, Tr, V):
    return xp.linalg.eigvalsh(TR + xp.diag(solve_BO_surface(Tr,V)))

@nvtx.annotate("gamma_build", color="red")
def Gamma_etf_erf(R,r,g,pr,pg,M_1,M_2,mu12,r1e2,r2e2):

    Ng = len(pg)
    Nr = len(pr)
    
    theta1 = xp.exp(-r1e2)
    theta2 = xp.exp(-r2e2)
    partition = theta1 + theta2

    cosgamma = xp.cos(g)
    singamma = xp.sin(g)
    invr = 1/(r[:,0])

    t1 = xp.diag((theta1/partition).ravel())
    t2 = xp.diag((theta2/partition).ravel())

    #px =  xp.kron(pr,xp.diag(cosgamma[0,:])) - xp.kron(xp.diag(invr),xp.dot(xp.diag(singamma[0,:]),pg))
    #py =  xp.kron(pr,xp.diag(singamma[0,:])) + xp.kron(xp.diag(invr),xp.dot(xp.diag(cosgamma[0,:]),pg))

    #px =  xp.kron(pr,xp.diag(cosgamma[0,:])) - xp.kron(xp.diag(invr),pg)
    #py =  xp.kron(pr,xp.diag(singamma[0,:])) + xp.kron(xp.diag(invr),pg)
    spg = pg.copy()
    cpg = pg.copy()
    xp.fill_diagonal(spg, xp.diag(spg) * singamma[0,:])
    xp.fill_diagonal(cpg, xp.diag(cpg) * cosgamma[0,:])

    px =  xp.kron(pr,xp.diag(cosgamma[0,:])) - xp.kron(xp.diag(invr),spg)
    py =  xp.kron(pr,xp.diag(singamma[0,:])) + xp.kron(xp.diag(invr),cpg)
    
    t1px = xp.dot(t1,px)
    pxt1 = xp.dot(px,t1)
    t2px = xp.dot(t2,px)
    pxt2 = xp.dot(px,t2)
    t1py = xp.dot(t1,py)
    pyt1 = xp.dot(py,t1)
    t2py = xp.dot(t2,py)
    pyt2 = xp.dot(py,t2)

    gammaetf1x = -0.5*(t1px + pxt1)
    gammaetf1y = -0.5*(t1py + pyt1)
    gammaetf2x = -0.5*(t2px + pxt2)
    gammaetf2y = -0.5*(t2py + pyt2)

    #rcosg = xp.diag((r*cosgamma).ravel())
    #rsing = xp.diag((r*singamma).ravel())
#
    #J1 = -0.5*(xp.dot((rcosg-(xp.eye(Nr*Ng)*R*mu12/M_1)),(t1py+pyt1))-xp.dot(rsing,(t1px+pxt1)))
    #J2 = -0.5*(xp.dot((rcosg+(xp.eye(Nr*Ng)*R*mu12/M_2)),(t2py+pyt2))-xp.dot(rsing,(t2px+pxt2)))

    #check signs
    #flip signs because of the cross product
    #gammaerf1y = 1/R*(J1+J2)
    #gammaerf2y = -1/R*(J1+J2)
    gammaerf1y = xp.zeros([Nr*Ng,Nr*Ng])
    gammaerf2y = xp.zeros([Nr*Ng,Nr*Ng])

    return gammaetf1x, gammaetf1y, gammaetf2x, gammaetf2y, gammaerf1y, gammaerf2y


def Gamma_etf_erf_old(R,r,g,pr,pg,M_1,M_2,mu12,r1e2,r2e2):

    Ng = len(pg)
    Nr = len(pr)

    theta1 = xp.exp(-r1e2)
    theta2 = xp.exp(-r2e2)
    partition = theta1 + theta2

    cosgamma = xp.cos(g)
    singamma = xp.sin(g)
    invr = 1/(r[:,0])

    t1 = xp.diag((theta1/partition).ravel())
    t2 = xp.diag((theta2/partition).ravel())

    #px =  xp.kron(pr,xp.diag(cosgamma[0,:])) - xp.kron(xp.diag(invr),xp.dot(xp.diag(singamma[0,:]),pg))
    #py =  xp.kron(pr,xp.diag(singamma[0,:])) + xp.kron(xp.diag(invr),xp.dot(xp.diag(cosgamma[0,:]),pg))

    #px =  xp.kron(pr,xp.diag(cosgamma[0,:])) - xp.kron(xp.diag(invr),pg)
    #py =  xp.kron(pr,xp.diag(singamma[0,:])) + xp.kron(xp.diag(invr),pg)

    px =  xp.kron(pr,xp.eye(Ng))
    py =  xp.kron(xp.diag(invr),pg)
    
    t1px = xp.dot(t1,px)
    pxt1 = xp.dot(px,t1)
    t2px = xp.dot(t2,px)
    pxt2 = xp.dot(px,t2)
    t1py = xp.dot(t1,py)
    pyt1 = xp.dot(py,t1)
    t2py = xp.dot(t2,py)
    pyt2 = xp.dot(py,t2)

    gammaetf1x = -0.5*(t1px + pxt1)
    gammaetf1y = -0.5*(t1py + pyt1)
    gammaetf2x = -0.5*(t2px + pxt2)
    gammaetf2y = -0.5*(t2py + pyt2)

    rcosg = xp.diag((r*cosgamma).ravel())
    rsing = xp.diag((r*singamma).ravel())

    J1 = -0.5*(xp.dot((rcosg-(xp.eye(Nr*Ng)*R*mu12/M_1)),(t1py+pyt1))-xp.dot(rsing,(t1px+pxt1)))
    J2 = -0.5*(xp.dot((rcosg+(xp.eye(Nr*Ng)*R*mu12/M_2)),(t2py+pyt2))-xp.dot(rsing,(t2px+pxt2)))

    #check signs
    #flip signs because of the cross product
    gammaerf1y = 1/R*(J1+J2)
    gammaerf2y = -1/R*(J1+J2)

    return gammaetf1x, gammaetf1y, gammaetf2x, gammaetf2y, gammaerf1y, gammaerf2y

@nvtx.annotate("inverse_weyl_transform", color="pink")
def inverse_weyl_transform(E, NR, R, P):
    """
    Perform the inverse Weyl transform 
    """
    HPS = xp.zeros((NR, NR), dtype=complex)
    EPP = xp.zeros((NR, NR), dtype=complex)
    EPS_half = xp.zeros((NR + 1, NR), dtype=complex)
    dR = R[1] - R[0]
    R_half = xp.linspace(R[0] - dR/2, R[-1] + dR/2, NR + 1)

    # Build EPP
    for i in range(NR):
        for j in range(NR):
            for k in range(NR):
                EPP[j, i] += xp.exp(-1j * R[k] * P[j]) * E[k, i] / xp.sqrt(NR)

    # Build EPS_half
    for i in range(NR):
        for j in range(NR + 1):
            for k in range(NR):
                EPS_half[j, i] += xp.exp(1j * R_half[j] * P[k]) * EPP[k, i] / xp.sqrt(NR)

    # Build HPS
    for j in range(NR):
        for q1 in range(NR):
            for q2 in range(NR):
                if (q1 - q2) % 2 == 0:
                    HPS[q1, q2] += (xp.exp(-1j * (R[q1] - R[q2]) * P[j])
                                    * E[(q1 + q2) // 2, j] / NR)
                else:
                    idx = (q1 + q2 + 1) // 2
                    HPS[q1, q2] += (xp.exp(-1j * (R[q1] - R[q2]) * P[j])
                                    * EPS_half[idx, j] / NR)
    return HPS

def inverse_weyl_transform_vec(E, NR, R, P):
    """
    Perform the inverse Weyl transform 
    """
    HPS = xp.zeros((NR, NR), dtype=complex)
    EPP = xp.zeros((NR, NR), dtype=complex)
    EPS_half = xp.zeros((NR + 1, NR), dtype=complex)
    dR = R[1] - R[0]
    R_half = xp.linspace(R[0] - dR/2, R[-1] + dR/2, NR + 1)

    q1_idx, q2_idx = xp.meshgrid(xp.arange(NR), xp.arange(NR), indexing='ij')
    mid_idx = (q1_idx + q2_idx) // 2
    mask = ((q1_idx - q2_idx) % 2 == 0)
    
    mask2 = ((q1_idx - q2_idx) % 2 == 1)
    mid_idx2 = (q1_idx + q2_idx + 1) // 2
    
    R_diff = R[q1_idx][:, :, None] - R[q2_idx][:, :, None]   # (NR, NR, 1) - (NR, NR, 1) * P[j] → broadcast over j
    phase = xp.exp(-1j * R_diff * P[None, None, :])          # shape (NR, NR, NR)

    # E[mid_idx, j] → shape (NR, NR, NR)
    E_mid = E[mid_idx, :]  # shape (NR, NR, NR)
    E_mid2 = EPS_half[mid_idx2, :] 

    # Einsum contraction over j
    HPS_part1 = xp.einsum('qpj,qpj->qp', phase, E_mid) / NR
    HPS_part2 = xp.einsum('qpj,qpj->qp', phase, E_mid2) / NR

    # Apply parity mask
    HPS = mask*HPS_part1+mask2*HPS_part2

    return HPS
