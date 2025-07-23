import sys
import importlib
import override

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
        # Check if there's an override for this attribute first
        override_func = override.get(self._backend_name, attr)

        if override_func:
            # Try to get the original attribute, but it might not exist (for patch-ins)
            try:
                backend_attr = getattr(self._backend, attr)
            except AttributeError:
                backend_attr = None

            # Create wrapper that calls override
            from functools import wraps
            @wraps(override_func)
            def wrapper(*args, **kwargs):
                return override_func(backend_attr, self._backend, self._backend_name, *args, **kwargs)
            return wrapper

        # No override, get the original attribute (this will raise AttributeError if not found)
        backend_attr = getattr(self._backend, attr)

        # Don't wrap callable functions, even if they have __dict__
        if callable(backend_attr):
            return backend_attr
        # Only wrap non-callable modules/objects that might have nested overrides
        elif hasattr(backend_attr, '__dict__') or hasattr(backend_attr, '__getattr__'):
            return override.Module(backend_attr, self._backend, self._backend_name, attr)
        else:
            # For simple values, return as-is
            return backend_attr

    def list_overrides(self, backend_name=None):
        """List all registered function overrides."""
        if backend_name:
            return override.list_overrides(backend_name)
        else:
            return override.list_overrides(self._backend_name)

# ============================================================================
# Backend-specific overrides
# ============================================================================

@override.register('cupy', 'linalg.eigh', warning="Attempting to override with torch")
def _cupy_eigh_fallback(original_func, backend, backend_name, *args, **kwargs):
    """Use torch eigh for better performance on cupy backend."""
    try:
        import torch
        torch.cuda.current_device()
        H = args[0]
        vals, vecs = torch.linalg.eigh(torch.from_dlpack(H))
        return backend.asarray(vals), backend.asarray(vecs)
    except (ImportError, ModuleNotFoundError, AssertionError, RuntimeError):
        # Fall back to cupy's native implementation
        return original_func(*args, **kwargs)

@override.register('cupy', 'linalg.eigvalsh', "Attempting to override with torch")
def _cupy_eigvalsh_fallback(original_func, backend, backend_name, *args, **kwargs):
    """Use torch eigvalsh for better performance on cupy backend."""
    try:
        import torch
        torch.cuda.current_device()
        H = args[0]
        vals = torch.linalg.eigvalsh(torch.from_dlpack(H))
        return backend.asarray(vals)
    except (ImportError, ModuleNotFoundError, AssertionError, RuntimeError):
        # Fall back to cupy's native implementation
        return original_func(*args, **kwargs)

# Patch in compatibility functions for backends that lack them
@override.register('torch', 'size')
def _torch_size(original_func, backend, backend_name, A):
    """Patch in size() function for torch backend."""
    return A.numel()

@override.register('torch', 'iscomplexobj')
def _torch_iscomplexobj(original_func, backend, backend_name, A):
    """Patch in iscomplexobj() function for torch backend."""
    return backend.is_complex(A)

# Replace the module with an instance of XPBackend (a singleton)
sys.modules[__name__] = XPBackend()
