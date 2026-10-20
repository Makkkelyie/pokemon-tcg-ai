"""Locate and import the competition's `cg` engine package.

The engine (C++ library + Python bindings) ships with the competition data and
may not be redistributed, so it is not committed. Locally it lives in
data/raw/.../sample_submission; inside a Kaggle submission it sits next to
main.py. Set PTCG_CG_DIR to override.
"""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CG_DIR = os.path.join(
    REPO_ROOT, "data", "raw", "sample_submission", "sample_submission", "sample_submission"
)


def cg_dir() -> str:
    return os.environ.get("PTCG_CG_DIR", DEFAULT_CG_DIR)


def ensure_cg():
    """Import and return the `cg` package, adding its parent dir to sys.path if needed."""
    try:
        import cg  # noqa: F401  (already importable, e.g. inside a submission)
    except ImportError:
        d = cg_dir()
        if not os.path.isdir(os.path.join(d, "cg")):
            raise ImportError(
                f"cg engine not found in {d}. Download the competition data "
                "into data/raw or set PTCG_CG_DIR."
            )
        sys.path.insert(0, d)
        import cg  # noqa: F401
    import cg.api
    import cg.game

    return sys.modules["cg"]


def load_deck(path: str) -> list[int]:
    with open(path) as f:
        deck = [int(x) for x in f.read().split() if x.strip()]
    if len(deck) != 60:
        raise ValueError(f"{path}: deck has {len(deck)} cards, expected 60")
    return deck


def default_deck_path() -> str:
    return os.path.join(cg_dir(), "deck.csv")
