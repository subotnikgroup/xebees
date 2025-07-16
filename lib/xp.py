import sys
import importlib

class XPBackend:
    def __init__(self):
        self._backend_name = None
        self._backend = None
        self.backend = self._detect_backend()

    def _detect_backend(self, backends=['numpy', 'cupy', 'torch',
                                        'cupynumeric', 'jax.numpy']):
        for backend in backends:
            try:
                importlib.import_module(backend)
            except ImportError:
                continue
            else:
                return backend
        raise ImportError("No supported backends found")

    @property
    def backend(self):
        return self._backend_name

    @backend.setter
    def backend(self, name):
        try:
            self._backend = importlib.import_module(name)
        except Exception:
            raise
        else:
            self._backend_name = name
            print(f"[xp] backend set to {self._backend_name}")

        if self._backend_name == 'jax.numpy':
            import jax
            jax.config.update('jax_enable_x64', True)
        elif self._backend_name == 'torch':
            self._backend.set_default_dtype(self._backend.float64)


    def __getattr__(self, attr):
        # Delegate attribute access to the backend module
        return getattr(self._backend, attr)

    # Helper functions for dealing with library inconsistencies
    def size(self, A):
        if self._backend_name == 'torch':
            return A.numel()
        else:
            return A.size

    def iscomplexobj(self, A):
        if self._backend_name == 'torch':
            return self._backend.is_complex(A)
        else:
            return self._backend.iscomplexobj(A)

# Replace the module with an instance of XPBackend (a singleton)
sys.modules[__name__] = XPBackend()
