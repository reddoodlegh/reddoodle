"""
Matrix subpackage containing functions performing matrix operations.
"""

from .elementary import rowreplacement, rowscale, rowswap, rref

__all__ = ['rowreplacement', 'rowscale', 'rowswap', 'rref']