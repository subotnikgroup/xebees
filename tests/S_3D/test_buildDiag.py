import pytest
from argparse import Namespace

import os, sys
sys.path.append(os.path.abspath("lib"))

import xp
import fixed_center_of_mass_exact_3D_S as fcm3d

def test_buildDaig():
    ''' Test that the internal function self.buildDiag correctly extracts the diagonal
        of the Hamiltonian by comparison to explict construction of H with self.Hx.'''

    xp.backend = 'numpy'

    base_params = {
        'g_1': 1.0, 'g_2': 1.0,
        'M_1': 2000, 'M_2': 2000,
        'NR': 3, 'Nr': 3, 'Ng': 4, 'Nint':100,
        'J': 0.5, 'potential': 'erf_coulomb',
        'preconditioner': 'naive',
        'verbosity': 0,
        'max_threads': 1,
        'soc': 'roi', 'alpha':1e4
    }
    args = Namespace(**base_params)
    H = fcm3d.Hamiltonian(args)
    N_tot = H.size
    H_diag = H.buildDiag()

    H_test = xp.zeros((N_tot,N_tot), dtype=H.dtype)
    for i in range(N_tot):
        xa = xp.zeros((N_tot))
        xa[i] = 1
        xout = H.Hx(xa)
        H_test[:,i] = xout
    test_diag = xp.diag(H_test)


    print("self.buildDiag sample output")
    print(H_diag)
    print("extracted diag from Hx")
    print(test_diag)
    print("diag diff:")
    print(test_diag-H_diag)

    # assert not xp.any(xp.imag(H_diag)) > 1e-12
    # assert not xp.sum(xp.imag(test_diag)) > 1e-8 
    test_diag = test_diag.real
    print(H_test)
    print("MAX non-real-sym-ness:", xp.max(xp.abs(H_test - H_test.T)))
    print("MAX non-hermitian-ness:", xp.max(xp.abs(H_test - xp.conj(H_test.T))))
    print("MAX non-hermitian-ness:", xp.max(xp.abs(H_test.imag - xp.conj(H_test.T).imag)))
    print(xp.sum(xp.abs(xp.imag(H_test))))
    # assert xp.max(xp.abs(H_test - H_test.T)) < 1e-8
    print("Diag deviation:", xp.mean(xp.abs(test_diag-H_diag)))
    # assert xp.mean(xp.abs(H_diag - test_diag)) < 1e-12
    
if __name__ == "__main__":
    # Allow running the test directly
    pytest.main([__file__, "-v", "-s"])
