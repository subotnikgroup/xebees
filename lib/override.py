"""
Module override framework for backend-specific function optimizations.

This module provides a registry-based system for overriding specific functions
in different backends (numpy, cupy, torch, etc.) with optimized implementations.
"""

from functools import wraps
import warnings

class OverrideRegistry:
    """Registry for backend-specific function overrides."""
    def __init__(self):
        self._overrides = {}

    def register(self, backend_name, function_path, override_func, warning=None):
        """
        Register an override function for a specific backend and function path.

        Args:
            backend_name: Name of the backend (e.g., 'cupy', 'numpy')
            function_path: Dot-separated path to function (e.g., 'linalg.eigh', 'fft.fft')
            override_func: Function that takes (original_func, backend, backend_name, *args, **kwargs)
                          and returns the result
            warning: Optional warning message to emit when override is used
        """
        if backend_name not in self._overrides:
            self._overrides[backend_name] = {}
        self._overrides[backend_name][function_path] = (override_func, warning)

    def get_override(self, backend_name, function_path):
        """Get override function for backend and function path, if any."""
        entry = self._overrides.get(backend_name, {}).get(function_path)
        if entry is None:
            return None
        # Handle both old format (just function) and new format (function, warning)
        if isinstance(entry, tuple):
            func, warning = entry
            if warning:
                warnings.warn(f"[{backend_name}.{function_path}] {warning}", UserWarning, stacklevel=4)
            return func
        else:
            # Old format - just the function
            return entry

    def list_overrides(self, backend_name=None):
        """List all registered overrides, optionally filtered by backend."""
        if backend_name:
            return list(self._overrides.get(backend_name, {}).keys())
        return {backend: list(funcs.keys()) for backend, funcs in self._overrides.items()}

# Global registry instance
_override_registry = OverrideRegistry()

def register(backend_name, function_path, warning=None):
    """Decorator to register function overrides."""
    def decorator(override_func):
        _override_registry.register(backend_name, function_path, override_func, warning)
        return override_func
    return decorator

class Module:
    """Wrapper that can override specific functions in a module."""
    def __init__(self, original_module, backend, backend_name, path_prefix=""):
        self._original_module = original_module
        self._backend = backend
        self._backend_name = backend_name
        self._path_prefix = path_prefix

    def __getattr__(self, name):
        current_path = f"{self._path_prefix}.{name}" if self._path_prefix else name

        # Check if this attribute has an override
        override_func = get(self._backend_name, current_path)

        if override_func:
            # Try to get the original attribute, but it might not exist (for patch-ins)
            try:
                attr = getattr(self._original_module, name)
                original_exists = True
            except AttributeError:
                attr = None
                original_exists = False

            if not original_exists or callable(attr):
                # Create wrapper that calls override (works for both existing functions and patch-ins)
                @wraps(override_func)
                def wrapper(*args, **kwargs):
                    return override_func(attr, self._backend, self._backend_name, *args, **kwargs)
                return wrapper

        # No override, get original attribute
        attr = getattr(self._original_module, name)

        # Only wrap modules/objects that aren't callable
        # If it's callable (like functions), return as-is
        if callable(attr):
            return attr
        elif hasattr(attr, '__dict__') or hasattr(attr, '__getattr__'):
            # If it's a module/object, wrap it recursively
            return Module(attr, self._backend, self._backend_name, current_path)
        else:
            # Return as-is for simple attributes
            return attr

# Convenience functions for external access
def list_overrides(backend_name=None):
    """List all registered overrides."""
    return _override_registry.list_overrides(backend_name)

def get(backend_name, function_path):
    """Get override function for backend and function path, if any."""
    return _override_registry.get_override(backend_name, function_path)

def get_registry():
    """Get the global override registry (for advanced usage)."""
    return _override_registry
