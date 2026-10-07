"""TTA methods: one file per method, each a TTAMethod subclass registered by name.

Importing this package imports every method module (that's what populates
the registry), the same way src/reliability/proxies/stats.py does for proxies.
"""

from src.tta.methods.base import (
    DEFAULT_TTA_METHOD, TTAMethod, build_tta_method, register, registered_names, resolve_tta_spec,
)
from src.tta.methods.tent import TentMethod
from src.tta.methods.eata import EataMethod

TTA_METHODS = frozenset(registered_names())

__all__ = [
    "DEFAULT_TTA_METHOD", "TTA_METHODS", "TTAMethod", "TentMethod", "EataMethod",
    "build_tta_method", "register", "registered_names", "resolve_tta_spec",
]
