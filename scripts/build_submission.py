"""Assemble build/submission/ and build/submission.tar.gz.

  python scripts/build_submission.py --deck decks/main.csv --config '{"budget": 0.25}'

The tarball contains main.py + deck.csv at top level, the engine package `cg/`
(from the competition data) and the agent package `ptcg/` (runtime modules only).
"""
import argparse
import json
import os
import shutil
import sys
import tarfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from ptcg.engine import cg_dir, default_deck_path, load_deck  # noqa: E402

RUNTIME_MODULES = [
    "__init__.py", "engine.py", "cards.py", "state.py", "belief.py", "evaluate.py", "search_api.py", "meta.py",
    "agents/__init__.py", "agents/base.py", "agents/heuristic.py", "agents/search.py", "agents/policy_search.py",
]
DIPAM_DIR = os.path.join(ROOT, "data", "vendor", "dipam")  # scripts/setup_dipam.py
META_DIR = os.path.join(ROOT, "decks", "meta")
# Only the Linux x86-64 library is needed on Kaggle; keep the others for local testing.
CG_FILES = ["__init__.py", "api.py", "game.py", "sim.py", "utils.py", "libcg.so", "libcg-arm64.so", "libcg.dylib"]


def build(deck_path, config, out_dir=os.path.join(ROOT, "build")):
    deck = load_deck(deck_path)
    sub = os.path.join(out_dir, "submission")
    shutil.rmtree(sub, ignore_errors=True)
    os.makedirs(os.path.join(sub, "ptcg", "agents"))
    os.makedirs(os.path.join(sub, "cg"))
    for m in RUNTIME_MODULES:
        shutil.copy(os.path.join(ROOT, "ptcg", m), os.path.join(sub, "ptcg", m))
    for f in CG_FILES:
        shutil.copy(os.path.join(cg_dir(), "cg", f), os.path.join(sub, "cg", f))
    with open(os.path.join(sub, "deck.csv"), "w") as f:
        f.write("\n".join(str(c) for c in deck) + "\n")
    if config.get("agent") == "ps":
        if not os.path.isdir(os.path.join(DIPAM_DIR, "dipam_ptcg")):
            raise SystemExit("run scripts/setup_dipam.py first")
        shutil.copytree(os.path.join(DIPAM_DIR, "dipam_ptcg"), os.path.join(sub, "dipam_ptcg"),
                        ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copytree(os.path.join(DIPAM_DIR, "data"), os.path.join(sub, "data"))
        shutil.copy(os.path.join(DIPAM_DIR, "model.pt"), os.path.join(sub, "model.pt"))
        shutil.copy(os.path.join(DIPAM_DIR, "LICENSE"), os.path.join(sub, "LICENSE-dipam"))
        if config.get("guess", "meta") == "meta" and os.path.isdir(META_DIR):
            shutil.copytree(META_DIR, os.path.join(sub, "guess_pool"))
    with open(os.path.join(ROOT, "submission", "main.py")) as f:
        main = f.read()
    main = main.replace("_CONFIG = {}", f"_CONFIG = {json.dumps(config, sort_keys=True)}", 1)
    with open(os.path.join(sub, "main.py"), "w") as f:
        f.write(main)
    tar_path = os.path.join(out_dir, "submission.tar.gz")
    with tarfile.open(tar_path, "w:gz") as tar:
        for name in sorted(os.listdir(sub)):
            tar.add(os.path.join(sub, name), arcname=name,
                    filter=lambda ti: None if "__pycache__" in ti.name else ti)
    return sub, tar_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deck", default=default_deck_path())
    ap.add_argument("--config", default="{}", help="JSON kwargs for MCAgent")
    args = ap.parse_args()
    sub, tar_path = build(args.deck, json.loads(args.config))
    print(f"built {sub}\n      {tar_path} ({os.path.getsize(tar_path) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
