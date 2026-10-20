import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from ptcg.engine import cg_dir, default_deck_path, load_deck  # noqa: E402

HAVE_ENGINE = os.path.isdir(os.path.join(cg_dir(), "cg"))
requires_engine = pytest.mark.skipif(not HAVE_ENGINE, reason="competition data (cg engine) not downloaded")


@pytest.fixture(scope="session")
def deck():
    return load_deck(default_deck_path())

HAVE_DIPAM = os.path.isdir(os.path.join(ROOT, "data", "vendor", "dipam", "dipam_ptcg"))
requires_dipam = pytest.mark.skipif(not (HAVE_ENGINE and HAVE_DIPAM),
                                    reason="vendored policy network missing (scripts/setup_dipam.py)")
