import sys
import importlib

class XPBackend:
    def __init__(self):
        self._backend_name = None
        self._backend = None
        self.backend = self._detect_backend()

    def _detect_backend(self, backends=['cupy', 'numpy',
                                        'cupynumeric', 'jax.numpy', 'torch']):
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

    def __getattr__(self, attr):
        # Delegate attribute access to the backend module
        return getattr(self._backend, attr)

# Replace the module with an instance of XPBackend (a singleton)
sys.modules[__name__] = XPBackend()
