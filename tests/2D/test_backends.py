"""
Test that different computational backends produce consistent results.
"""

import pytest
import warnings
from argparse import Namespace

import xp
import numpy
import fixed_center_of_mass_exact as fcm2d
import linalg_helper as lib
from davidson import get_davidson_mem


def test_backend_consistency():
    """Test that numpy, cupy, and cupynumeric backends produce the same results."""
    backends = ['numpy', 'torch', 'cupy', 'cupynumeric']
    available_backends = []
    results = {}

    # Test which backends are available
    for backend in backends:
        try:
            xp.backend = backend
        except (ImportError, Exception):
            print(f"Backend {backend} not available, skipping")
            continue
        else:
            available_backends.append(backend)

    if len(available_backends) < 2:
        pytest.skip(f"Need at least 2 backends, only have: {available_backends}")

    print(f"\nTesting backends: {available_backends}")

    # Common parameters for all backends
    base_params = {
        'g_1': 1.0, 'g_2': 1.0,
        'M_1': 100, 'M_2': 100,
        'NR': 15, 'Nr': 18, 'Ng': 20,
        'J': 0, 'potential': 'borgis',
        'preconditioner': 'BO',
        'verbosity': 0,
        'max_threads': 1,
        'k': 4, # 4 eigenvalues
    }

    # Run computation with each available backend
    for backend in available_backends:
        print(f"Testing backend: {backend}")
        xp.backend = backend

        args = Namespace(**base_params, backend=backend)
        H = fcm2d.Hamiltonian(args)

        guess = H.make_guess(args.k)
        conv, e_approx, evecs = lib.davidson1(
            H.Hx,
            guess,
            H.preconditioner,
            nroots=args.k,
            max_cycle=100,
            verbose=10,
            max_space=500,
            max_memory=get_davidson_mem(0.75),
            tol=1e-10,
        )

        if backend == 'cupy':
            results[backend] = e_approx.get() if hasattr(e_approx, 'get') else numpy.asarray(e_approx)
        else:
            results[backend] = numpy.asarray(e_approx)

        print(f"{backend}: eigenvalues = {e_approx}")
        print(f"{backend}: converged = {conv}")
        if not all(conv):
            warnings.warn(f"Not all eigenvalues converged for {backend}: {conv}", UserWarning)

    reference = available_backends[0]

    for backend in available_backends:
        if backend == reference:
            continue

        diff = numpy.max(numpy.abs(results[backend] - results[reference]))
        print(f"  {reference} vs. {backend}: max eigenvalue diff = {diff:.2e}")

        # Assert reasonable agreement (backends may have small numerical differences)
        assert diff < 1e-10, f"Eigenvalue difference too large between {backend} and {reference}: {diff}"

    print(f"All {len(available_backends)} backends produce consistent results")


if __name__ == "__main__":
    # Allow running the test directly
    pytest.main([__file__, "-v", "-s"])
