"""Play N best-of-3 matches between two agents and report win rate with a 95% CI.

Example:
  python scripts/run_match.py heuristic random -n 200 -w 6
  python scripts/run_match.py mc heuristic -n 100 --a-kw budget=0.2 --deck-a decks/a.csv
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ptcg.engine import default_deck_path  # noqa: E402
from ptcg.match import AgentSpec, run_matches  # noqa: E402


def parse_kw(items):
    kw = {}
    for it in items or []:
        k, v = it.split("=", 1)
        try:
            v = json.loads(v)
        except json.JSONDecodeError:
            pass
        kw[k] = v
    return kw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("-n", type=int, default=100, help="number of BO3 matches")
    ap.add_argument("-w", "--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--deck-a", default=None)
    ap.add_argument("--deck-b", default=None)
    ap.add_argument("--a-kw", nargs="*")
    ap.add_argument("--b-kw", nargs="*")
    ap.add_argument("--bo", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log", default="results/matches.jsonl", help="append summary here")
    args = ap.parse_args()

    deck_a = args.deck_a or default_deck_path()
    deck_b = args.deck_b or deck_a
    a = AgentSpec(args.a, deck_a, parse_kw(args.a_kw))
    b = AgentSpec(args.b, deck_b, parse_kw(args.b_kw))
    t0 = time.time()

    def progress(rows):
        if len(rows) % max(1, args.n // 10) == 0:
            s = sum(r["score"] for r in rows) / len(rows)
            print(f"  {len(rows)}/{args.n}  score={s:.3f}  ({time.time() - t0:.0f}s)", flush=True)

    res = run_matches(a, b, args.n, workers=args.workers, bo=args.bo, seed=args.seed, progress=progress)
    lo, hi = res["match_ci95"]
    glo, ghi = res["game_ci95"]
    print(f"{a.label()} vs {b.label()}: {res['matches']} BO{args.bo} matches")
    print(f"  match score {res['match_score']:.3f}  95% CI [{lo:.3f}, {hi:.3f}]  "
          f"W/L/D {res['wins']}/{res['losses']}/{res['draws']}")
    print(f"  game win rate {res['game_win_rate']:.3f}  95% CI [{glo:.3f}, {ghi:.3f}]")
    print(f"  A: {res['ms_per_decision_a']:.1f} ms/decision, max think time in one game "
          f"{res['max_game_time_a']:.1f}s;  errors: {len(res['errors'])}")
    for e in res["errors"][:5]:
        print("   ", e)
    os.makedirs(os.path.dirname(args.log), exist_ok=True)
    with open(args.log, "a") as f:
        row = {k: v for k, v in res.items() if k != "rows"}
        row.update(a=a.label(), b=b.label(), bo=args.bo, seed=args.seed, wall_s=round(time.time() - t0, 1),
                   errors=res["errors"][:20])
        f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
