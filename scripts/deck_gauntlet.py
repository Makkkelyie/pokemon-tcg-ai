"""Candidate decks vs a share-weighted field: the policy pilots both sides (mode=none, KO guard on).

    python scripts/deck_gauntlet.py --cands A,B --field X=.34,Y=.10 --dir data/derived/cands -n 24

Each (candidate, opponent) result is appended to --out as one JSON line.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from ptcg.match import AgentSpec, run_matches  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cands", required=True, help="comma-separated deck names (files <dir>/<name>.csv)")
    p.add_argument("--field", required=True, help="comma-separated name=share pairs")
    p.add_argument("--dir", default="decks")
    p.add_argument("-n", type=int, default=24, help="BO3 matches per pairing")
    p.add_argument("-w", "--workers", type=int, default=7)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out", default="results/gauntlet.jsonl")
    a = p.parse_args()
    field = {k: float(v) for k, v in (kv.split("=") for kv in a.field.split(","))}
    kw = {"mode": "none", "guess": "none"}
    for c in a.cands.split(","):
        tot = 0.0
        for o, w in field.items():
            t0 = time.time()
            r = run_matches(AgentSpec("ps", f"{a.dir}/{c}.csv", kw), AgentSpec("ps", f"{a.dir}/{o}.csv", kw),
                            a.n, workers=a.workers, seed=a.seed)
            row = {"cand": c, "opp": o, "n": r["matches"], "score": r["match_score"],
                   "game": r["game_win_rate"], "errors": len(r["errors"]), "s": round(time.time() - t0)}
            with open(a.out, "a") as f:
                f.write(json.dumps(row) + "\n")
            print(row, flush=True)
            tot += w * r["match_score"]
        print(f"== {c}: share-weighted match score {tot / sum(field.values()):.3f}", flush=True)


if __name__ == "__main__":
    main()
