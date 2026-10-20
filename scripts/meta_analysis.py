"""Ladder meta analysis from Kaggle's daily episode export.

Streams every episode JSON out of the downloaded zip (no extraction), and fits

  1. a Bradley-Terry model over *agents* (team, deck) -- who is strong;
  2. a ridge-regularised Bradley-Terry model with separate *pilot* (team) and
     *deck* (archetype) effects, logit P(i beats j) = (t_i + d_i) - (t_j + d_j),
     so a deck is not credited for the skill of the teams that happen to play it;
  3. first-player advantage and thinking-time usage.

Outputs (gitignored, may contain card names / deck lists):
  data/derived/episodes.jsonl   one row per BO3 match
  data/derived/decks.json       deck hash -> sorted 60-card list
  results/private/meta_*.csv    tables

  python scripts/meta_analysis.py data/external/episodes/*.zip
"""
import argparse
import collections
import csv
import glob
import hashlib
import json
import math
import multiprocessing as mp
import os
import sys
import zipfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IS_FIRST_CTX, YES = 41, 1


def deck_hash(deck):
    return hashlib.sha1(",".join(map(str, sorted(deck))).encode()).hexdigest()[:8]


def parse_episode(raw: bytes):
    ep = json.loads(raw)
    steps = ep["steps"]
    info = ep.get("info") or {}
    teams = info.get("TeamNames") or ["?", "?"]
    decks = [steps[1][s].get("action") if len(steps) > 1 else None for s in (0, 1)]
    games = {}
    pools = {}
    decisions = [0, 0]
    for i, step in enumerate(steps):
        for seat in (0, 1):
            a = step[seat]
            obs = a.get("observation") or {}
            cur = obs.get("current")
            if not cur:
                continue
            rnd = cur.get("round") or 1
            g = games.setdefault(rnd, {"first": -1, "turns": 0, "wins": None, "chooser": None, "chose_first": None})
            if cur.get("firstPlayer", -1) != -1:
                g["first"] = cur["firstPlayer"]
            g["turns"] = max(g["turns"], cur.get("turn") or 0)
            if g["wins"] is None:
                g["wins"] = [p.get("win") or 0 for p in cur["players"]]
            if a.get("status") == "ACTIVE":
                decisions[seat] += 1
                t = obs.get("remainingOverageTime")
                if t is not None:
                    pr = pools.setdefault(rnd, [[None, None], [None, None]])  # [first seen, last seen]
                    if pr[0][seat] is None:
                        pr[0][seat] = t
                    pr[1][seat] = t
                sel = obs.get("select")
                if sel and sel.get("context") == IS_FIRST_CTX and i + 1 < len(steps):
                    act = steps[i + 1][seat].get("action")
                    if act:
                        g["chooser"], g["chose_first"] = seat, sel["option"][act[0]]["type"] == YES
    rewards = ep.get("rewards") or [None, None]
    rounds = sorted(games)
    out_games = []
    for k, rnd in enumerate(rounds):
        g = games[rnd]
        if k + 1 < len(rounds):
            nxt = games[rounds[k + 1]]["wins"]
            winner = next((s for s in (0, 1) if nxt[s] > g["wins"][s]), 2)
        else:
            winner = 2 if None in rewards or rewards[0] == rewards[1] else int(rewards[1] > rewards[0])
        out_games.append({"winner": winner, "first": g["first"], "turns": g["turns"],
                          "chooser": g["chooser"], "chose_first": g["chose_first"]})
    ok = [isinstance(d, list) and len(d) == 60 for d in decks]
    return {
        "episode": info.get("EpisodeId"),
        "teams": teams,
        "rewards": rewards,
        "statuses": ep.get("statuses"),
        "decks": [deck_hash(d) if o else None for d, o in zip(decks, ok)],
        "lists": {deck_hash(d): sorted(d) for d, o in zip(decks, ok) if o},
        "games": out_games,
        "pool_left_by_round": {str(r): v for r, v in pools.items()},
        "decisions": decisions,
        "module_version": ep.get("module_version"),
    }


def _work(args):
    path, name = args
    with zipfile.ZipFile(path) as z:
        try:
            return parse_episode(z.read(name))
        except Exception as e:  # a corrupt file should not kill the whole run
            return {"error": f"{name}: {e!r}"}


# ------------------------------------------------------------------ models

def bradley_terry(results, iters=500):
    """MM algorithm (Hunter 2004) with one virtual draw vs a unit-strength reference."""
    wins, games = collections.Counter(), collections.defaultdict(collections.Counter)
    for a, b, s in results:
        wins[a] += s
        wins[b] += 1 - s
        games[a][b] += 1
        games[b][a] += 1
    p = dict.fromkeys(games, 1.0)
    for _ in range(iters):
        p = {i: (wins[i] + 0.5) / (sum(n / (p[i] + p[j]) for j, n in games[i].items()) + 1 / (p[i] + 1)) for i in p}
    return {i: 400 * math.log10(v) for i, v in p.items()}


def ridge_bt(matches, l2_team=1.0, l2_deck=4.0, iters=50):
    """Logistic BT with additive team + deck effects; Newton iterations on the penalised log-lik.

    matches: list of (team_a, deck_a, team_b, deck_b, score_a in {0, .5, 1}).
    Returns ({team: effect}, {deck: effect}, {deck: std err}) on the Elo scale (400/ln10 per logit).
    """
    teams = sorted({m[0] for m in matches} | {m[2] for m in matches})
    decks = sorted({m[1] for m in matches} | {m[3] for m in matches})
    ti = {t: i for i, t in enumerate(teams)}
    di = {d: len(teams) + i for i, d in enumerate(decks)}
    n = len(teams) + len(decks)
    X = np.zeros((len(matches), n))
    y = np.zeros(len(matches))
    for r, (ta, da, tb, db, s) in enumerate(matches):
        X[r, ti[ta]] += 1
        X[r, ti[tb]] -= 1
        X[r, di[da]] += 1
        X[r, di[db]] -= 1
        y[r] = s
    lam = np.array([l2_team] * len(teams) + [l2_deck] * len(decks))
    w = np.zeros(n)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ w))
        grad = X.T @ (y - p) - lam * w
        H = (X * (p * (1 - p))[:, None]).T @ X + np.diag(lam)
        step = np.linalg.solve(H, grad)
        w += step
        if np.max(np.abs(step)) < 1e-8:
            break
    p = 1 / (1 + np.exp(-X @ w))
    H = (X * (p * (1 - p))[:, None]).T @ X + np.diag(lam)
    se = np.sqrt(np.diag(np.linalg.inv(H)))
    k = 400 / math.log(10)
    return ({t: k * w[ti[t]] for t in teams}, {d: k * w[di[d]] for d in decks}, {d: k * se[di[d]] for d in decks})


def wilson(w, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p, d = w / n, 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


# --------------------------------------------------------------- archetypes

def load_card_names():
    names, info = {}, {}
    path = os.path.join(ROOT, "data", "raw", "EN_Card_Data_R2_full.csv")
    with open(path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            cid = int(r["Card ID"])
            if cid in names:
                continue
            names[cid] = r["Card Name"].strip()
            hp = r["HP"].strip()
            stage = r["Stage (Pokémon)/Type (Energy and Trainer)"]
            info[cid] = {"pokemon": hp.isdigit(), "stage": 2 if stage.startswith("Stage 2") else
                         1 if stage.startswith("Stage 1") else 0,
                         "mega": r["Rule"] == "Mega Pokémon ex", "ex": r["Rule"] in ("Pokémon ex", "Mega Pokémon ex"),
                         "hp": int(hp) if hp.isdigit() else 0}
    return names, info


def archetype(ids, names, info):
    """Name a deck by its two headline Pokémon (Mega ex > ex > stage > copies)."""
    counts = collections.Counter(ids)
    poke = [c for c in counts if info.get(c, {}).get("pokemon")]
    ranked = sorted(poke, key=lambda c: (counts[c] >= 2, info[c]["mega"], info[c]["ex"], info[c]["stage"],
                                         counts[c], info[c]["hp"]), reverse=True)
    out = []
    for c in ranked:
        if names[c] not in out:
            out.append(names[c])
        if len(out) == 2:
            break
    return " / ".join(out) or "no Pokémon"


# --------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("zips", nargs="+")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--min-matches", type=int, default=15)
    args = ap.parse_args()

    derived = os.path.join(ROOT, "data", "derived")
    private = os.path.join(ROOT, "results", "private")
    os.makedirs(derived, exist_ok=True)
    os.makedirs(private, exist_ok=True)

    tasks = []
    for zp in args.zips:
        with zipfile.ZipFile(zp) as z:
            tasks += [(zp, n) for n in z.namelist() if n.endswith(".json")]
    print(f"{len(tasks)} episodes")
    matches, lists, errors = [], {}, []
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for k, r in enumerate(pool.imap_unordered(_work, tasks, chunksize=4)):
            if "error" in r:
                errors.append(r["error"])
                continue
            lists.update(r.pop("lists"))
            matches.append(r)
            if (k + 1) % 200 == 0:
                print(f"  parsed {k + 1}", flush=True)
    print(f"parsed {len(matches)} matches, {len(errors)} errors")
    with open(os.path.join(derived, "episodes.jsonl"), "w") as f:
        for m in matches:
            f.write(json.dumps(m) + "\n")
    with open(os.path.join(derived, "decks.json"), "w") as f:
        json.dump(lists, f)

    names, info = load_card_names()
    arch = {h: archetype(ids, names, info) for h, ids in lists.items()}
    valid = [m for m in matches if None not in m["decks"] and None not in m["rewards"]]

    def score(m):
        r = m["rewards"]
        return 1.0 if r[0] > r[1] else 0.0 if r[0] < r[1] else 0.5

    # 1. agent strength
    res = [((m["teams"][0], m["decks"][0]), (m["teams"][1], m["decks"][1]), score(m)) for m in valid]
    bt = bradley_terry(res)
    stat = collections.defaultdict(lambda: {"n": 0, "w": 0.0})
    for a, b, s in res:
        stat[a]["n"] += 1
        stat[a]["w"] += s
        stat[b]["n"] += 1
        stat[b]["w"] += 1 - s
    rows = sorted(((bt[a], a, s) for a, s in stat.items() if s["n"] >= args.min_matches), reverse=True)
    with open(os.path.join(private, "meta_agents.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bt", "team", "deck", "archetype", "matches", "match_win_rate"])
        for r, (team, deck), s in rows:
            w.writerow([round(r), team, deck, arch[deck], s["n"], round(s["w"] / s["n"], 3)])
    print("\nTop agents (BT, team, archetype, matches, win%):")
    for r, (team, deck), s in rows[:15]:
        print(f"  {r:7.0f}  {team[:24]:24s} {arch[deck][:40]:40s} {s['n']:4d} {100 * s['w'] / s['n']:5.1f}")

    # 2. deck effects controlling for pilots, at archetype level
    a_matches = [(m["teams"][0], arch[m["decks"][0]], m["teams"][1], arch[m["decks"][1]], score(m)) for m in valid]
    team_eff, deck_eff, deck_se = ridge_bt(a_matches)
    usage = collections.Counter()
    games_by_arch = collections.defaultdict(lambda: [0, 0])
    for m in valid:
        for s in (0, 1):
            usage[arch[m["decks"][s]]] += 1
            for g in m["games"]:
                if g["winner"] in (0, 1):
                    games_by_arch[arch[m["decks"][s]]][0] += g["winner"] == s
                    games_by_arch[arch[m["decks"][s]]][1] += 1
    arows = sorted(((deck_eff[a], a) for a in usage if usage[a] >= args.min_matches), reverse=True)
    with open(os.path.join(private, "meta_archetypes.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["deck_effect_elo", "se", "archetype", "match_slots", "share", "game_win_rate", "lo95", "hi95"])
        for e, a in arows:
            gw, gn = games_by_arch[a]
            lo, hi = wilson(gw, gn)
            w.writerow([round(e, 1), round(deck_se[a], 1), a, usage[a], round(usage[a] / (2 * len(valid)), 3),
                        round(gw / gn, 3) if gn else "", round(lo, 3), round(hi, 3)])
    print("\nArchetypes: deck effect (Elo, pilot-adjusted) +- se, share, raw game win%")
    for e, a in arows[:15]:
        gw, gn = games_by_arch[a]
        print(f"  {e:+6.0f} +-{deck_se[a]:3.0f}  {a[:44]:44s} share {usage[a] / (2 * len(valid)):5.1%}  "
              f"games {100 * gw / max(gn, 1):5.1f}% (n={gn})")

    # 3. turn order and thinking time
    first_w = first_n = 0
    chose = chose_first = 0
    for m in valid:
        for g in m["games"]:
            if g["winner"] in (0, 1) and g["first"] in (0, 1):
                first_n += 1
                first_w += g["winner"] == g["first"]
            if g["chooser"] is not None:
                chose += 1
                chose_first += bool(g["chose_first"])
    lo, hi = wilson(first_w, first_n)
    print(f"\nFirst player wins {first_w}/{first_n} games = {first_w / max(first_n, 1):.3f} "
          f"[{lo:.3f}, {hi:.3f}]; agents chose to go first {chose_first}/{chose}")
    resets = sum(1 for m in valid for r, v in m["pool_left_by_round"].items()
                 if r != "1" and v[0][0] is not None and v[0][0] > 599.5)
    later = sum(1 for m in valid for r in m["pool_left_by_round"] if r != "1")
    print(f"Games 2/3 starting with a full 600 s pool (per-game reset): {resets}/{later}")
    used = [600 - min(v for fl in m["pool_left_by_round"].values() for v in fl[1] if v is not None)
            for m in valid if m["pool_left_by_round"]]
    print(f"Pool used per match: median {np.median(used):.0f}s, p90 {np.percentile(used, 90):.0f}s, "
          f"max {max(used):.0f}s; module versions {collections.Counter(m['module_version'] for m in valid)}")
    summary = {"matches": len(valid), "first_player_game_win": [first_w, first_n], "chose_first": [chose_first, chose],
               "archetypes": len(usage), "agents": len(stat)}
    with open(os.path.join(private, "meta_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)


if __name__ == "__main__":
    sys.exit(main())
