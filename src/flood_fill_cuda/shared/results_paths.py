"""
Centralized output-path helper for every chapter's benchmarks/ scripts.

Every chapter writes its generated data (JSON, CSV, HTML, GIFs, PNGs) to
results/<chapter_id>/... instead of a directory next to the script itself,
so the whole project's generated output lives in one predictable,
uniformly-discoverable location — this is what lets the dashboard find any
chapter's data from just its chapter id, with no per-module path exposure.
"""

import os

_PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_ROOT = os.path.join(_PACKAGE_ROOT, "results")


def results_dir(chapter_id, *parts):
    """Absolute path under results/<chapter_id>/<parts...>, created if missing."""
    path = os.path.join(RESULTS_ROOT, chapter_id, *parts)
    os.makedirs(path, exist_ok=True)
    return path
