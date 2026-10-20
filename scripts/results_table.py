"""Print results/matches.jsonl as a markdown table (one row per experiment)."""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main(path=os.path.join(ROOT, "results", "matches.jsonl")):
    print("| A | B | matches | match score [95% CI] | game win rate [95% CI] | ms/decision (A) | errors |")
    print("|---|---|---:|---|---|---:|---:|")
    for line in open(path):
        r = json.loads(line)
        lo, hi = r["match_ci95"]
        glo, ghi = r["game_ci95"]
        print(f"| {r['a']} | {r['b']} | {r['matches']} | {r['match_score']:.3f} [{lo:.3f}, {hi:.3f}] | "
              f"{r['game_win_rate']:.3f} [{glo:.3f}, {ghi:.3f}] | {r['ms_per_decision_a']:.1f} | {len(r['errors'])} |")


if __name__ == "__main__":
    main(*sys.argv[1:])
