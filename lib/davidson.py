import xp
# N.B. 2D and later code uses numpy for loading and interpolating. 1D
# code also uses numpy for building and applying the preconditioner
import numpy
from os import sysconf
from debug import timer, timer_ctx
import linalg_helper as lib
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

    guess = numpt.load(guessfile)['guess']
    if guess.shape[1] == numpy.prod(grid_dims):
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
        with numpy.load(guessfile, allow_pickle=True) as npz:
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
        print("Attempting to interpolate guesses on new grid!")
        return xp.asarray(list(map(
            lambda g: interpolate_guess(g.reshape(H.shape),
                                        H.axes,
                                        axes,
                                        method=method).ravel(),
            guess)))


# FIXME: Interpolator only works on CPU with numpy backend; explore GPU options?
def interpolate_guess(psi, axes, axes_target, method='cubic'):
     # Create interpolator
    interpolator = RegularGridInterpolator(axes, psi, method=method,
                                           bounds_error=False,
                                           fill_value=None)  # extrapolate

    # Mesh for new grid
    mesh = numpy.meshgrid(*axes_target, indexing='ij')
    points = numpy.stack([m.ravel() for m in mesh], axis=-1)
    shape = [len(ax) for ax in axes_target]

    # Interpolate
    # FIXME: need to parallelize for large numbers of points
    psi_target = interpolator(points).reshape(shape)
    return psi_target


def eye_lazy(N):
    for i in range(N):
        col = xp.zeros(N)
        col[i] = 1.0
        yield col


def phase_match_orig(U):
    N, _, M = U.shape

    if xp.iscomplexobj(U):
        phase = lambda x, y: xp.exp(-1j*xp.angle(xp.dot(x.conj(), y)))
    else:
        phase = lambda x, y: xp.sign(xp.dot(x, y))

    # FIXME: rewrite in vectorized format
    for i in range(1,N):
        for n in range(M):
            U[i,:,n] *= phase(U[i-1,:,n], U[i,:,n])

def phase_match_mem_constrained(U):
    N, _, M = U.shape

    if xp.iscomplexobj(U):
        # Complex case - vectorized
        for i in range(1, N):
            # Compute dot products for all M vectors at once
            dots = xp.sum(U[i-1].conj() * U[i], axis=1)  # Shape: (M,)
            phases = xp.exp(-1j * xp.angle(dots))        # Shape: (M,)
            U[i] *= phases[None, :]                      # Broadcast to (_, M)
    else:
        # Real case - vectorized
        for i in range(1, N):
            # Compute dot products for all M vectors at once
            dots = xp.sum(U[i-1] * U[i], axis=1)         # Shape: (M,)
            signs = xp.sign(dots)                        # Shape: (M,)
            U[i] *= signs[None, :]                       # Broadcast to (_, M)


def phase_match(U):
    if xp.iscomplexobj(U):
        overlaps = xp.sum(U[:-1].conj() * U[1:], axis=1)
        phases = xp.exp(-1j * xp.angle(overlaps))
    else:
        overlaps = xp.sum(U[:-1] * U[1:], axis=1)
        phases = xp.sign(overlaps)

    # Apply phases cumulatively
    cumulative_phases = xp.cumprod(phases, axis=0)
    U[1:] *= cumulative_phases[:, None, :]

@timer
def build_preconditioner(TR, Tr, Vgrid, min_guess=4):
    NR, Nr = Vgrid.shape

    guess = xp.zeros((NR,Nr))

    U_n    = xp.zeros((NR,Nr,Nr))
    U_v    = xp.zeros((Nr,NR,NR))
    Ad_n   = xp.zeros((NR,Nr))
    Ad_vn  = xp.zeros((NR,Nr))

    # diagonalize H electronic: r->n
    for i in range(NR):
        Hel = Tr + xp.diag(Vgrid[i])
        Ad_n[i], U_n[i] = xp.linalg.eigh(Hel)

        # align phases
        if i > 0:
            for j in range(Nr):
                if xp.sum(U_n[i,:,j] * U_n[i-1,:,j]) < 0:
                    U_n[i,:,j] *= -1.0

    # diagonalize Born-Oppenheimer Hamiltonian: R->v
    for i in range(Nr):
        Hbo = TR + xp.diag(Ad_n[:,i])
        Ad_vn[:,i], U_v[i] = xp.linalg.eigh(Hbo)

        # align phases
        if i > 0:
            for j in range(NR):
                if xp.sum(U_v[i,:,j] * U_v[i-1,:,j]) < 0:
                    U_v[i,:,j] *= -1.0

    # BO states are like: U_n[:,:,n]
    # vib states are like: U_v[n,:,v]
    # our first guess was the ground state BO wavefuction dressed by the first vibrational state
    # guess = U_n[:,:,0] * U_v[0,:,0,xp.newaxis]
    # Now we take something like the first num_guess states
    s = int(xp.ceil(xp.sqrt(min_guess)))
    guesses = xp.asarray([(U_n[:,:,n] * U_v[n,:,v,xp.newaxis]).ravel() for n in range(s) for v in range(s)])

    def precond_Rn(dx, e, x0):
        dx_ = dx.reshape((-1, NR, Nr))
        dx_Rn = xp.einsum('Rji,BRj->BRi', U_n, dx_)
        tr_Rn = dx_Rn / (Ad_n[None, :, :] - e)
        tr_Rr = xp.einsum('Rij,BRj->BRi', U_n, tr_Rn)
        return tr_Rr.reshape(dx.shape)


    # for our simple case, these contractions were no observable help and harder to read
    # to_vn = opt_einsum.contract_expression('nji,jqn,jq->in', U_v, U_n, (NR,Nr), constants=[0,1], optimize='optimal')
    # to_Rr = opt_einsum.contract_expression('Rij,jRq,qj->Ri', U_n, U_v, (NR,Nr), constants=[0,1], optimize='optimal')

    # Elimination of temporaries by merging the contractions powered by opt_einsum_fx.
    # c.f.: https://opt-einsum-fx.readthedocs.io/en/latest/api.html#opt_einsum_fx.fuse_einsums
    def precond_vn(dx, e, x0):
        dx_ = dx.reshape((-1, NR, Nr))
        dx_vn = xp.einsum('nji,jqn,Bjq->Bin', U_v, U_n, dx_, optimize=True)
        tr_vn = dx_vn / (Ad_vn[None, :, :] - e)
        tr_Rr = xp.einsum('Rij,jRq,Bqj->BRi', U_n, U_v, tr_vn, optimize=True)
        return tr_Rr.reshape(dx.shape)

    return precond_vn, guesses

@timer
def solve_exact(TR, Tr, Vgrid, num_state=10):
    H = (xp.kron(TR, xp.eye(Nr)) +
         xp.kron(xp.eye(NR), Tr) +
         xp.diag(Vgrid.ravel())
    )

    eigenvalues, eigenvectors = xp.linalg.eigh(H)
    return eigenvalues[:num_state]

@timer
def solve_exact_gen(Hx, N, num_state=10):
    with timer_ctx(f"Build H of size {N}"):
        H = xp.array([
            Hx(e) for e in xp.eye(N)
        ])

    eigenvalues, eigenvectors = xp.linalg.eigh(H)
    return eigenvalues[:num_state]



@timer
def solve_davidson(TR, Tr, Vgrid,
                   num_state=10,
                   verbosity=2,
                   iterations=1000,
                   max_subspace=1000,
                   guess=None,):

    def Hx(xs):
        xa = xs.reshape((-1,) + Vgrid.shape)
        r = xp.einsum('ij,bjk->bik', TR, xa)
        r += xp.einsum('bjk,kl->bjl', xa, Tr)
        r += xa * Vgrid[None, :, :]
        return r.reshape(xs.shape)

    if guess is None:
        pc_unitary, guess = build_preconditioner(TR, Tr, Vgrid, num_state)
    else:
        pc_unitary, _ = build_preconditioner(TR, Tr, Vgrid)

    conv, eigenvalues, eigenvectors = lib.davidson1(
        Hx,
        guess,
        pc_unitary,
        nroots=num_state,
        max_cycle=iterations,
        verbose=verbosity,
        max_space=max_subspace,
        max_memory=get_davidson_mem(0.75),
        tol=1e-12,
    )

    return conv, eigenvalues, eigenvectors
