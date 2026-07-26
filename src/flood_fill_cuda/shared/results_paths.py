"""
Centralized output-path helper for every chapter's benchmarks/ scripts.

Every chapter writes its generated data (JSON, CSV, HTML, GIFs, PNGs) to
results/<chapter_id>/... instead of a directory next to the script itself,
so the whole project's generated output lives in one predictable,
uniformly-discoverable location — this is what lets the dashboard find any
chapter's data from just its chapter id, with no per-module path exposure.
"""

import glob
import os
import sys

_PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_ROOT = os.path.join(_PACKAGE_ROOT, "results")


def results_dir(chapter_id, *parts):
    """Absolute path under results/<chapter_id>/<parts...>, created if missing."""
    path = os.path.join(RESULTS_ROOT, chapter_id, *parts)
    os.makedirs(path, exist_ok=True)
    return path


def newest(pattern, folder):
    """Newest (by sorted basename) file matching pattern in folder.

    Exits with an error if none exist — for a chapter's REQUIRED benchmark
    JSON, where a dashboard genuinely cannot render without it.
    """
    candidates = sorted(glob.glob(os.path.join(folder, pattern)))
    if not candidates:
        sys.exit(f"no benchmark JSON matching {pattern} — run the benchmark first")
    return candidates[-1]


def newest_optional(pattern, folder, exclude=None):
    """Like newest, but returns None instead of exiting.

    For an OPTIONAL dataset (e.g. a later stage's JSON): a dashboard run
    predating it should still render, just omitting that section. exclude
    drops basenames containing the given substring — e.g. dual_blob_*.json
    would otherwise swallow dual_blob_radius2_*.json.
    """
    candidates = sorted(glob.glob(os.path.join(folder, pattern)))
    if exclude:
        candidates = [c for c in candidates
                      if exclude not in os.path.basename(c)]
    return candidates[-1] if candidates else None
