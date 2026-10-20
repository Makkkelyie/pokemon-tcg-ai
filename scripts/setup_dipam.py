"""Vendor the public dipam baseline (MIT) as package `dipam_ptcg` under data/vendor/dipam.

Source: Kaggle dataset cristobaltudela/ptcg-playground-public-baseline (Dipam
Chakraborty's agent, MIT, ported to the Playground card pool by Cristobal
Tudela), downloaded to data/external/bots/dipam_port. The package is renamed
so it can live next to this repo's own `ptcg` package; its data paths are
relative to the package parent, so data/ and model.pt are copied alongside.
The seat-feature fix published by Akmal Xodarev (busyaprime) is applied.

  python scripts/setup_dipam.py [--src data/external/bots/dipam_port]
"""
import argparse
import glob
import os
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEST = os.path.join(ROOT, "data", "vendor", "dipam")
SEAT_OLD = "    g[GlobF.ACTOR_SEAT] = float(seat)\n"
SEAT_NEW = "    g[GlobF.ACTOR_SEAT] = 0.0 if first == -1 else float(first != seat)\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(ROOT, "data", "external", "bots", "dipam_port"))
    args = ap.parse_args()
    shutil.rmtree(DEST, ignore_errors=True)
    os.makedirs(DEST)
    shutil.copytree(os.path.join(args.src, "ptcg"), os.path.join(DEST, "dipam_ptcg"),
                    ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(os.path.join(args.src, "data"), os.path.join(DEST, "data"))
    for f in ("model.pt", "LICENSE", "README.md", "deck.csv"):
        shutil.copy(os.path.join(args.src, f), os.path.join(DEST, f))
    enc = os.path.join(DEST, "dipam_ptcg", "rl", "encoder.py")
    s = open(enc).read()
    assert s.count(SEAT_OLD) == 1, "seat line not found: upstream changed"
    open(enc, "w").write(s.replace(SEAT_OLD, SEAT_NEW))
    # The rl package adds <bundle>/data to sys.path for the SDK; harmless, but keep it explicit.
    n = len(glob.glob(os.path.join(DEST, "dipam_ptcg", "**", "*.py"), recursive=True))
    print(f"vendored {n} modules to {DEST}")


if __name__ == "__main__":
    main()
