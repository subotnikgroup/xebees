"""
Integration tests for 1D quantum mechanics solvers.
Tests both fixed center of mass and fixed single mass solvers
with known parameters and expected eigenvalues.
"""

import pytest
from argparse import Namespace
from threadpoolctl import ThreadpoolController

import xp

import fixed_center_of_mass_exact as fcm
import fixed_single_mass_exact as fsm
from davidson import solve_davidson


@pytest.fixture(scope="module", autouse=True)
def limit_threads():
    """Limit the number of threads used by numerical libraries during tests."""
    with ThreadpoolController().limit(limits=2):
        yield

class TestFixedCenterOfMass:
    def test_eigenvalues_reference(self):
        xp.backend = 'numpy'

        args = Namespace(
            g_1=1.1, g_2=1.0,
            M_1=2.0, M_2=4.0,
            NR=101, Nr=400,
            k=10
        )

        # Build the Hamiltonian terms
        TR, Tr, Tmp, Vgrid, *_ = fcm.build_terms(args)

        # Solve using Davidson iteration
        conv, e_approx, evecs = solve_davidson(
            TR, Tr + Tmp, Vgrid,
            num_state=args.k,
            verbosity=5,
            iterations=1000,
            max_subspace=500,
            guess=None
        )

        expected = xp.array([
            -0.0551502, -0.05028052, -0.04758383, -0.04527196, -0.04169519,
            -0.04116102, -0.0401033, -0.03656404, -0.03556294, -0.03485186
        ])

        # Verify convergence
        assert all(conv), f"Not all eigenvalues converged: {conv}"
        max_diff = xp.max(xp.abs(e_approx - expected))
        assert max_diff < 1e-6, f"Maximum eigenvalue difference {max_diff} exceeds tolerance"


class TestFixedSingleMass:
    def test_eigenvalues_reference(self):
        xp.backend = 'numpy'

        args = Namespace(
            g_1=1.1, g_2=1.0,
            M=4.0,
            NR=101, Nr=400,
            k=10
        )

        TR, Tr, Vgrid, *_ = fsm.build_terms(args)

        conv, e_approx, evecs = solve_davidson(
            TR, Tr, Vgrid,
            num_state=args.k,
            verbosity=5,
            iterations=500,
            max_subspace=250,
            guess=None
        )

        # Expected eigenvalues from recent successful run
        expected = xp.array([
            -0.0571052,  -0.05322405, -0.05161531, -0.04954828, -0.04789593,
            -0.04614574, -0.04512367, -0.04429429, -0.04293521, -0.04152778])

        # Verify convergence
        assert all(conv), f"Not all eigenvalues converged: {conv}"
        max_diff = xp.max(xp.abs(e_approx - expected))
        assert max_diff < 1e-6, f"Maximum eigenvalue difference {max_diff} exceeds tolerance"


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v", "-s"])
