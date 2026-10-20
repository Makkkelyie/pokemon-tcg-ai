"""Opponent-deck prior from the ladder meta (scripts/meta_analysis.py output).

The pool is the set of decklists seen on the ladder with their usage counts.
Given the opponent's revealed cards, `match` keeps the decks that contain the
most of them (a containment likelihood) and samples among those in
proportion to ladder usage (the prior).
"""
import glob
import json
import os
from collections import Counter

from .cards import card_db
from .engine import REPO_ROOT


def _pool_dirs():
    env = os.environ.get("PTCG_GUESS_POOL")
    return ([env] if env else []) + [os.path.join(REPO_ROOT, "guess_pool"), os.path.join(REPO_ROOT, "decks", "meta")]


class GuessPool:
    def __init__(self, decks, weights):
        self.decks = decks
        self.weights = weights
        db = card_db()
        self.basics = [[c for c in d if db[c].is_pokemon and db[c].basic] or [d[0]] for d in decks]
        self.counters = [Counter(d) for d in decks]

    def match(self, tracker, rng) -> int:
        seen = Counter(tracker.opp_known_hand.values()) + Counter(tracker.opp_known_deck.values())
        obs = tracker._last_obs
        if obs is not None:
            cur = obs["current"]
            opp = cur["players"][1 - cur["yourIndex"]]
            for p in (opp.get("active") or []) + (opp.get("bench") or []):
                if p:
                    seen[p["id"]] += 1
                    for e in p.get("energyCards") or []:
                        seen[e["id"]] += 1
            for d in opp.get("discard") or []:
                seen[d["id"]] += 1
        if seen:
            scores = [sum((seen & c).values()) for c in self.counters]
            hi = max(scores)
            idx = [i for i, s in enumerate(scores) if s == hi]
        else:
            idx = list(range(len(self.decks)))
        w = [self.weights[i] for i in idx]
        return rng.choices(idx, weights=w, k=1)[0]


def load_guess_pool():
    for d in _pool_dirs():
        files = sorted(glob.glob(os.path.join(d, "*.csv")))
        if not files:
            continue
        wpath = os.path.join(d, "weights.json")
        weights = json.load(open(wpath)) if os.path.exists(wpath) else {}
        decks, ws = [], []
        for f in files:
            ids = [int(x) for x in open(f).read().split() if x.strip()]
            if len(ids) == 60:
                decks.append(ids)
                ws.append(float(weights.get(os.path.splitext(os.path.basename(f))[0], 1.0)))
        if decks:
            return GuessPool(decks, ws)
    return None
