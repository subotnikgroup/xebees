import functools
import xp

def maybejit(*jit_args, **jit_kwargs):
    # JIT compile function via numba when numpy backend is active.
    # Defaults: nopython=True, parallel=True

    jit_kwargs.setdefault('nopython', True)
    jit_kwargs.setdefault('parallel', True)

    def decorator(func):
        # Try to create JIT version, fallback if numba unavailable
        try:
            import numba
            jit_func = numba.jit(*jit_args, **jit_kwargs)(func)
        except ImportError:
            jit_func = None

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Use JIT version only for numpy backend
            if xp.backend == 'numpy' and jit_func is not None:
                return jit_func(*args, **kwargs)
            else:
                return func(*args, **kwargs)
        return wrapper
    return decorator
