"""
Ensure that the eigen spectra from Hel and Hel_j are the same
"""

import pytest
from argparse import Namespace

import xp
import fixed_center_of_mass_exact_2D as fcm2d

def test_hel_j_convergence():
    """Test that error between build_Hel and build_Hel_j decreases as
    Ng increases. Specfically, we ensure that there is minimal
    deviation in the lowest several electronic surfaces computed in
    both ways and that deviation decreses with increasing Ng.
    """

    ng_values = [16, 24, 32, 48]
    expected_improvement = 50  # Expect 50x improvement
    check_states = 5           # Number of low-lying surfaces we test

    xp.backend='cupy'
    if not xp.cuda.is_available():
        xp.backend='numpy'

    base_params = {
        'g_1': 1.0, 'g_2': 1.0,
        'M_1': 100, 'M_2': 100,
        'NR': 19, 'Nr': 22,
        'J': 0, 'potential': 'borgis',
        'preconditioner': 'naive',
        'verbosity': 0,
        'max_threads': 1,
    }

    errors = []
    print(f"\nTesting convergence with Ng values: {ng_values}")

    for Ng in ng_values:
        args = Namespace(**base_params, Ng=Ng)
        H = fcm2d.Hamiltonian(args)

        # Compute BO spectra with both methods
        Ev_g, En_g = H.BO_spectrum(0, Hel_func=H.build_Hel)
        Ev_j, En_j = H.BO_spectrum(0, Hel_func=H.build_Hel_j)

        state_errors = xp.max(xp.abs(En_g.T[:check_states] - En_j.T[:check_states]), axis=1)
        avg_error = xp.mean(state_errors)
        errors.append(avg_error)

        print(f"Ng={Ng:2d}: avg error = {avg_error:.2e}")

    # Check that error generally decreases (allow small fluctuations)
    for i in range(1, len(errors)):
        improvement_ratio = errors[i] / errors[i-1]
        assert improvement_ratio <= 1.5, \
            f"Error increased too much: Ng={ng_values[i-1]} -> {ng_values[i]}, " \
            f"ratio = {improvement_ratio:.2f}"
    print("All errors:", xp.asarray(errors))

    # Check overall improvement from first to last
    total_improvement = errors[0] / errors[-1]
    assert total_improvement >= expected_improvement, \
        f"Expected {expected_improvement}x improvement, got {total_improvement:.1f}x " \
        f"(from {errors[0]:.2e} to {errors[-1]:.2e})"

    print(f"Convergence verified: {total_improvement:.1f}x improvement ") \

if __name__ == "__main__":
    # Allow running the test directly
    pytest.main([__file__, "-v", "-s"])
