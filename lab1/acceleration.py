"""Optional deterministic JIT acceleration; Python remains a usable fallback."""
try:
    from numba import njit
except ImportError:
    def njit(**options):
        def decorate(function):
            function.py_func = function
            return function
        return decorate
