"""Local best-of-N match runner mirroring kaggle_environments' cabt interpreter.

Differences from the official env are deliberate and small: no JSON replay
rendering, and an optional per-game overage-time budget (600 s per player per
game on Kaggle) that is reported but only enforced when `enforce_time=True`.
"""
from __future__ import annotations

import math
import multiprocessing as mp
import os
import time
from dataclasses import dataclass, field

from .engine import ensure_cg, load_deck

MAX_STEPS_PER_GAME = 20000


@dataclass
class MatchResult:
    winner: int  # 0, 1, or 2 (draw)
    game_winners: list[int] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    think_time: list[float] = field(default_factory=lambda: [0.0, 0.0])
    max_game_time: list[float] = field(default_factory=lambda: [0.0, 0.0])
    decisions: list[int] = field(default_factory=lambda: [0, 0])


def _bo_decided(res, bo):
    remain = bo - sum(res)
    if res[0] > res[1] + remain:
        return 0
    if res[1] > res[0] + remain:
        return 1
    if remain <= 0:
        return 2
    return -1


def play_match(agents, bo: int = 3, time_limit: float = 600.0, enforce_time: bool = False) -> MatchResult:
    """Play one best-of-`bo` episode between two callables (Kaggle agent signature)."""
    cg = ensure_cg()
    game = cg.game
    out = MatchResult(winner=-1)
    deck_obs = {"select": None, "logs": [], "current": None, "search_begin_input": None}
    decks = []
    for p in range(2):
        try:
            d = agents[p](dict(deck_obs, remainingOverageTime=time_limit))
            if len(d) != 60:
                raise ValueError("deck size")
            decks.append(list(d))
        except Exception as e:  # invalid deck -> lose episode
            out.errors.append(f"p{p} deck: {e!r}")
            out.winner = 1 - p
            return out

    res = [0, 0, 0]
    count = 0
    while True:
        obs, sd = game.battle_start(decks[0], decks[1], count % 2 != 0)
        if obs is None:
            p = sd.errorPlayer if sd.errorPlayer >= 0 else 0
            out.errors.append(f"p{p} battle_start error {sd.errorType}")
            out.winner = 1 - p
            return out
        remaining = [time_limit, time_limit]
        steps = 0
        loser = -1
        while obs["current"]["result"] < 0:
            p = obs["current"]["yourIndex"]
            obs["remainingOverageTime"] = remaining[p]
            t0 = time.perf_counter()
            try:
                action = agents[p](obs)
                err = None
            except Exception as e:
                err = f"p{p} agent exception: {e!r}"
            dt = time.perf_counter() - t0
            remaining[p] -= dt
            out.think_time[p] += dt
            out.decisions[p] += 1
            if err is None and enforce_time and remaining[p] < 0:
                err = f"p{p} timeout"
            if err is None:
                try:
                    obs = game.battle_select([int(a) for a in action])
                except Exception as e:
                    err = f"p{p} invalid action {action!r}: {e!r}"
            if err is not None:
                out.errors.append(err)
                loser = p
                break
            steps += 1
            if steps > MAX_STEPS_PER_GAME:
                out.errors.append("step limit")
                break
        for q in range(2):
            out.max_game_time[q] = max(out.max_game_time[q], time_limit - remaining[q])
        game.battle_finish()
        if loser >= 0:  # error/timeout ends the whole episode, like the env
            out.winner = 1 - loser
            return out
        r = obs["current"]["result"] if steps <= MAX_STEPS_PER_GAME else 2
        res[min(r, 2)] += 1
        out.game_winners.append(r)
        count += 1
        w = _bo_decided(res, bo)
        if w >= 0:
            out.winner = w
            return out


# ---------------------------------------------------------------- statistics

def wilson(k: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


# ------------------------------------------------------------- parallel runs

@dataclass
class AgentSpec:
    """Picklable description of an agent: registry name + kwargs + deck file."""
    name: str
    deck_path: str
    kwargs: dict = field(default_factory=dict)

    def build(self, seed):
        from .agents import make_agent
        return make_agent(self.name, load_deck(self.deck_path), seed=seed, **self.kwargs)

    def label(self):
        kw = ",".join(f"{k}={v}" for k, v in sorted(self.kwargs.items()))
        deck = os.path.splitext(os.path.basename(self.deck_path))[0]
        return f"{self.name}({kw})[{deck}]" if kw else f"{self.name}[{deck}]"


def _run_one(args):
    a, b, i, bo, seed = args
    swap = i % 2 == 1  # alternate seats so neither agent always sits at index 0
    ag_a, ag_b = a.build(seed * 2), b.build(seed * 2 + 1)
    agents = [ag_b, ag_a] if swap else [ag_a, ag_b]
    r = play_match(agents, bo=bo)
    a_idx = 1 if swap else 0
    if r.winner == 2:
        score = 0.5
    else:
        score = 1.0 if r.winner == a_idx else 0.0
    games_a = sum(1 for g in r.game_winners if g == a_idx)
    games_b = sum(1 for g in r.game_winners if g == 1 - a_idx)
    return {
        "i": i,
        "score": score,
        "games_a": games_a,
        "games_b": games_b,
        "errors": r.errors,
        "time_a": r.think_time[a_idx],
        "time_b": r.think_time[1 - a_idx],
        "max_game_time_a": r.max_game_time[a_idx],
        "max_game_time_b": r.max_game_time[1 - a_idx],
        "dec_a": r.decisions[a_idx],
        "dec_b": r.decisions[1 - a_idx],
    }


def run_matches(a: AgentSpec, b: AgentSpec, n: int, workers: int = 4, bo: int = 3, seed: int = 0, progress=None):
    tasks = [(a, b, i, bo, seed * 100003 + i) for i in range(n)]
    rows = []
    if workers <= 1:
        for t in tasks:
            rows.append(_run_one(t))
            if progress:
                progress(rows)
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(workers) as pool:
            for r in pool.imap_unordered(_run_one, tasks):
                rows.append(r)
                if progress:
                    progress(rows)
    return summarize(rows)


def summarize(rows):
    n = len(rows)
    score = sum(r["score"] for r in rows)
    wins = sum(1 for r in rows if r["score"] == 1.0)
    ga = sum(r["games_a"] for r in rows)
    gb = sum(r["games_b"] for r in rows)
    lo, hi = wilson(score, n)
    glo, ghi = wilson(ga, ga + gb)
    dec_a = sum(r["dec_a"] for r in rows) or 1
    return {
        "matches": n,
        "match_score": score / n if n else 0.0,
        "match_ci95": (lo, hi),
        "wins": wins,
        "losses": sum(1 for r in rows if r["score"] == 0.0),
        "draws": sum(1 for r in rows if r["score"] == 0.5),
        "game_win_rate": ga / (ga + gb) if ga + gb else 0.0,
        "game_ci95": (glo, ghi),
        "errors": [e for r in rows for e in r["errors"]],
        "ms_per_decision_a": 1000 * sum(r["time_a"] for r in rows) / dec_a,
        "max_game_time_a": max((r["max_game_time_a"] for r in rows), default=0.0),
        "rows": rows,
    }
