import numpy as np
from os import sysconf
from debug import timer, timer_ctx
from pyscf import lib
from scipy.interpolate import RegularGridInterpolator
#import linalg_helper as lib

# import opt_einsum

def get_davidson_mem(fraction):
    if fraction > 1 or fraction < 0:
        raise RuntimeError("Fraction of system memory for Davidson must be on [0, 1]")

    try:
        system_memory_mb = (sysconf('SC_PAGE_SIZE') * sysconf('SC_PHYS_PAGES')) / 1024**2
    except ValueError:
        print("Unable to determine system memory!")
        system_memory_mb = 8000
    finally:
        davidson_mem = fraction * system_memory_mb
        print(f"Davidson will consume up to {int(davidson_mem)}MB of memory.")

    return davidson_mem


def get_davidson_guess(guessfile, grid_dims):
    if guessfile is None:
        return

    if not guessfile.exists():
        print(f"WARNING: requested guess-file, {guessfile}, does not exist!")
        return

    guess = np.load(guessfile)['guess']
    if guess.shape[1] == np.prod(grid_dims):
        print("Loaded guess from", guessfile)
        return guess
    else:
        print("WARNING: Loaded guess of improper dimension; discarding!")
        return


# FIXME: There's probably a way to do some kind of interpolation on
# the parameters (given the way that, e.g. r/R, are changed with
# changing masses
def get_interpolated_guess(guessfile, axes, method='cubic'):
    if guessfile is None:
        return

    try:
        with np.load(guessfile, allow_pickle=True) as npz:
            guess = npz['guess']
            H = npz['H'].item()
    except Exception as e:
        print(f"WARNING: Unable to load {guessfile}; the error was:", e, sep='\n')
        return

    target_shape = tuple(len(ax) for ax in axes)
    if H.shape == target_shape:
        print("Loaded guess from", guessfile)
        return guess
    else:
        print("Attempting to interpolate guess on new grid!")
        #FIXME don't build interpolator for each guess!!
        return list(map(
            lambda g: interpolate_guess(g.reshape(H.shape),
                                        H.axes,
                                        axes,
                                        method=method).ravel(),
            guess))


def interpolate_guess(psi, axes, axes_target, method='cubic'):
     # Create interpolator
    interpolator = RegularGridInterpolator(axes, psi, method=method,
                                           bounds_error=False,
                                           fill_value=None)  # extrapolate

    # Mesh for new grid
    mesh = np.meshgrid(*axes_target, indexing='ij')
    points = np.stack([m.ravel() for m in mesh], axis=-1)
    shape = [len(ax) for ax in axes_target]

    # Interpolate
    # FIXME: need to parallelize for large numbers of points
    psi_target = interpolator(points).reshape(shape)
    return psi_target


def eye_lazy(N):
    for i in range(N):
        col = np.zeros(N)
        col[i] = 1.0
        yield col


def phase_match(U):
    N, _, M = U.shape

    if np.iscomplexobj(U):
        phase = lambda x, y: np.exp(-1j*np.angle(np.dot(x.conj(), y)))
    else:
        phase = lambda x, y: np.sign(np.dot(x, y))

    for i in range(1,N):
        for n in range(M):
            U[i,:,n] *= phase(U[i-1,:,n], U[i,:,n])

        
@timer
def build_preconditioner(TR, Tr, Vgrid, min_guess=4):
    NR, Nr = Vgrid.shape

    guess = np.zeros((NR,Nr))

    U_n    = np.zeros((NR,Nr,Nr))
    U_v    = np.zeros((Nr,NR,NR))
    Ad_n   = np.zeros((NR,Nr))
    Ad_vn  = np.zeros((NR,Nr))

    # diagonalize H electronic: r->n
    for i in range(NR):
        Hel = Tr + np.diag(Vgrid[i])
        Ad_n[i], U_n[i] = np.linalg.eigh(Hel)

        # align phases
        if i > 0:
            for j in range(Nr):
                if np.sum(U_n[i,:,j] * U_n[i-1,:,j]) < 0:
                    U_n[i,:,j] *= -1.0

    # diagonalize Born-Oppenheimer Hamiltonian: R->v
    for i in range(Nr):
        Hbo = TR + np.diag(Ad_n[:,i])
        Ad_vn[:,i], U_v[i] = np.linalg.eigh(Hbo)

        # align phases
        if i > 0:
            for j in range(NR):
                if np.sum(U_v[i,:,j] * U_v[i-1,:,j]) < 0:
                    U_v[i,:,j] *= -1.0

    # BO states are like: U_n[:,:,n]
    # vib states are like: U_v[n,:,v]
    # our first guess was the ground state BO wavefuction dressed by the first vibrational state
    # guess = U_n[:,:,0] * U_v[0,:,0,np.newaxis]
    # Now we take something like the first num_guess states
    s = int(np.ceil(np.sqrt(min_guess)))
    guesses = [(U_n[:,:,n] * U_v[n,:,v,np.newaxis]).ravel() for n in range(s) for v in range(s)]

    def precond_Rn(dx, e, x0):
        dx_Rr = dx.reshape((NR,Nr))
        
        #for i in range(NR):
        #    dx_Rn[i] = U_n[i].T @ dx_Rr[i]

        dx_Rn = np.einsum('Rji,Rj->Ri', U_n, dx_Rr)
        tr_Rn = dx_Rn / (Ad_n - e)
        tr_Rr = np.einsum('Rij,Rj->Ri', U_n, tr_Rn)
        
        #for i in range(NR):
        #    tr_Rr[i] = U_n[i] @ tr_Rn[i]
        
        return tr_Rr.ravel()


    # for our simple case, these contractions were no observable help and harder to read
    # to_vn = opt_einsum.contract_expression('nji,jqn,jq->in', U_v, U_n, (NR,Nr), constants=[0,1], optimize='optimal')
    # to_Rr = opt_einsum.contract_expression('Rij,jRq,qj->Ri', U_n, U_v, (NR,Nr), constants=[0,1], optimize='optimal')

    # Elimination of temporaries by merging the contractions powered by opt_einsum_fx.
    # c.f.: https://opt-einsum-fx.readthedocs.io/en/latest/api.html#opt_einsum_fx.fuse_einsums
    def precond_vn(dx, e, x0):
        dx_Rr = dx.reshape((NR,Nr))

        #dx_Rn = np.einsum('Rji,Rj->Ri', U_n, dx_Rr, optimize=True)
        #dx_vn = np.einsum('nji,jn->in', U_v, dx_Rn, optimize=True)

        dx_vn = np.einsum('nji,jqn,jq->in', U_v, U_n, dx_Rr, optimize=True)

        tr_vn = dx_vn / (Ad_vn - e)

        #tr_Rn = np.einsum('nij,jn->in', U_v, tr_vn, optimize=True)
        #tr_Rr = np.einsum('Rij,Rj->Ri', U_n, tr_Rn, optimize=True)

        tr_Rr = np.einsum('Rij,jRq,qj->Ri', U_n, U_v, tr_vn, optimize=True)

        return tr_Rr.ravel()

    return precond_vn, guesses

@timer
def solve_exact(TR, Tr, Vgrid, num_state=10):
    H = (np.kron(TR, np.eye(Nr)) +
         np.kron(np.eye(NR), Tr) +
         np.diag(Vgrid.ravel())
    )

    eigenvalues, eigenvectors = np.linalg.eigh(H)
    return eigenvalues[:num_state]

@timer
def solve_exact_gen(Hx, N, num_state=10):
    with timer_ctx(f"Build H of size {N}"):
        H = np.array([
            Hx(e) for e in np.eye(N)
        ])

    eigenvalues, eigenvectors = np.linalg.eigh(H)
    return eigenvalues[:num_state]



@timer
def solve_davidson(TR, Tr, Vgrid,
                   num_state=10,
                   verbosity=2,
                   iterations=1000,
                   max_subspace=1000,
                   guess=None,):
    def aop_fast(x):
        xa = x.reshape(Vgrid.shape)
        r  = TR @ xa
        r += xa @ (Tr)
        r += xa * Vgrid
        return r.ravel()

    aop = lambda xs: [ aop_fast(x) for x in xs ]

    if guess is None:
        pc_unitary, guess = build_preconditioner(TR, Tr, Vgrid, num_state)
    else:
        pc_unitary, _ = build_preconditioner(TR, Tr, Vgrid)


    conv, eigenvalues, eigenvectors = lib.davidson1(
        aop,
        guess,
        pc_unitary,
        nroots=num_state,
        max_cycle=iterations,
        verbose=verbosity,
        follow_state=False,
        max_space=max_subspace,
        max_memory=get_davidson_mem(0.75),
        tol=1e-12,
    )

    return conv, eigenvalues, eigenvectors
