"""4-connectivity CPU reference, re-exported from ../single_block_shared.

Loaded by file path via importlib: the sibling GPU packages have no
__init__.py and share module basenames, so a plain import would shadow.
spec_from_file_location gives the module a private name and a correct
__file__ without touching sys.modules. @njit(cache=True) keys its cache on
the source file path, so both packages share one compiled reference.
"""
import importlib.util
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_SBS = os.path.abspath(os.path.join(_HERE, os.pardir, "single_block_shared"))


def load_by_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_reference = load_by_path("_sbs_reference", os.path.join(_SBS, "reference.py"))
cpu_flood_fill = _reference.cpu_flood_fill
