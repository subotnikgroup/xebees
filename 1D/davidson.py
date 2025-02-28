import numpy as np
from os import sysconf
from debug import timer
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


@timer
def build_preconditioner(TR, Tr, Vgrid):
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
            if np.sum(U_n[i] * U_n[i-1]) < 0:
                U_n[i] *= -1.0

        guess[i] = U_n[i, 0].T

    # diagonalize Born-Oppenheimer Hamiltonian: R->v
    for i in range(Nr):
        Hbo = TR + np.diag(Ad_n[:,i])
        Ad_vn[:,i], U_v[i] = np.linalg.eigh(Hbo)

        # align phases
        if i > 0:
            if np.sum(U_v[i] * U_v[i-1]) < 0:
                U_v[i] *= -1.0

    # stamp down the vib-ground state
    for i in range(Nr):
        guess[:,i] =  U_v[0].T @ guess[:,i]


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

    return precond_vn, guess.ravel()

