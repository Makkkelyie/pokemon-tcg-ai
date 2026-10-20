"""Determinized Monte-Carlo search over the engine's forward model.

For each decision with more than one candidate action:
  1. candidates = every option for single-choice selects (MAIN, attack, yes/no,
     counts), or the heuristic pick plus sampled alternatives for multi-selects;
  2. repeat until the time/simulation budget runs out: sample a determinization
     of the hidden cards (ptcg.belief), apply a candidate (chosen by UCB1),
     roll out with the noisy heuristic policy for both players until
     `horizon` turn boundaries have passed (or the game ends), and score the
     final state with ptcg.evaluate;
  3. play the candidate with the best mean value.
Any engine/search error falls back to the heuristic pick, so the agent can
never return an illegal action because of the search.
"""
import math
import time

from .. import search_api
from ..belief import Belief
from ..evaluate import DEFAULT_WEIGHTS, evaluate
from . import REGISTRY
from .base import Agent
from .heuristic import C_MAIN, HeuristicAgent, pick, score_all

SINGLE_CHOICE_TYPES = {0, 6, 8, 9}  # MAIN, ATTACK, COUNT, YES_NO


class MCAgent(Agent):
    name = "mc"

    def __init__(self, deck, seed=None, budget=0.25, max_sims=400, horizon=1, temperature=40.0,
                 c_ucb=0.6, max_rollout_steps=400, alt_picks=4, min_remaining=60.0, weights=None,
                 search_non_main=True):
        super().__init__(deck, seed)
        self.budget = float(budget)
        self.max_sims = int(max_sims)
        self.horizon = int(horizon)
        self.temperature = float(temperature)
        self.c_ucb = float(c_ucb)
        self.max_rollout_steps = int(max_rollout_steps)
        self.alt_picks = int(alt_picks)
        self.min_remaining = float(min_remaining)
        self.weights = dict(DEFAULT_WEIGHTS, **(weights or {}))
        self.search_non_main = bool(search_non_main)
        self.belief = Belief(deck)
        self.fallback = HeuristicAgent(deck, seed)
        self.stats = {"searched": 0, "fallback": 0, "sims": 0, "errors": 0}

    def new_match(self):
        self.belief = Belief(self.deck)

    # ------------------------------------------------------------------
    def act(self, obs):
        self.belief.observe(obs)
        sel = obs["select"]
        heur = pick(sel, score_all(obs))
        if len(sel["option"]) <= 1 or sel["minCount"] == sel["maxCount"] == len(sel["option"]):
            return heur
        if sel["context"] != C_MAIN and not self.search_non_main:
            return heur
        if not obs.get("search_begin_input"):
            return heur
        remaining = float(obs.get("remainingOverageTime", 600.0) or 600.0)
        budget = self.budget if remaining > self.min_remaining else min(self.budget, 0.02)
        try:
            cands = self._candidates(obs, heur)
            if len(cands) <= 1:
                return heur
            best = self._search(obs, cands, budget)
            self.stats["searched"] += 1
            return best
        except Exception:
            self.stats["errors"] += 1
            return heur
        finally:
            try:
                search_api.end()
            except Exception:
                pass

    def _candidates(self, obs, heur):
        sel = obs["select"]
        n = len(sel["option"])
        if sel["type"] in SINGLE_CHOICE_TYPES and sel["maxCount"] == 1:
            return [[i] for i in range(n)]
        scores = score_all(obs)
        cands = [list(heur)]
        seen = {tuple(sorted(heur))}
        for _ in range(self.alt_picks * 3):
            if len(cands) > self.alt_picks:
                break
            c = pick(sel, scores, self.rng, temperature=self.temperature)
            key = tuple(sorted(c))
            if key not in seen:
                seen.add(key)
                cands.append(c)
        return cands

    def _search(self, obs, cands, budget):
        me = obs["current"]["yourIndex"]
        root_turn = obs["current"]["turn"]
        n_opt = len(obs["select"]["option"])
        k = len(cands)
        visits = [0] * k
        total = [0.0] * k
        deadline = time.perf_counter() + budget
        sims = 0
        while sims < self.max_sims and (sims < k or time.perf_counter() < deadline):
            if sims < k:
                j = sims
            else:
                logn = math.log(sims)
                j = max(range(k), key=lambda a: total[a] / visits[a] + self.c_ucb * math.sqrt(logn / visits[a]))
            det = self.belief.determinize(obs, self.rng)
            sid, sobs = search_api.begin(obs, **det)
            if len(sobs["select"]["option"]) != n_opt:
                raise search_api.SearchError("determinized root differs from real root")
            v = self._rollout(sid, sobs, cands[j], me, root_turn)
            visits[j] += 1
            total[j] += v
            sims += 1
            if sims % 32 == 0:
                search_api.end()  # free memory of finished simulations
        self.stats["sims"] += sims
        best = max(range(k), key=lambda a: (total[a] / visits[a] if visits[a] else -9, visits[a]))
        return cands[best]

    def _rollout(self, sid, sobs, first_action, me, root_turn):
        sid, sobs = search_api.step(sid, first_action)
        steps = 0
        while True:
            cur = sobs["current"]
            if cur["result"] >= 0:
                break
            if cur["turn"] - root_turn >= self.horizon and sobs["select"]["context"] == C_MAIN:
                break
            if steps >= self.max_rollout_steps:
                break
            sel = sobs["select"]
            act = pick(sel, score_all(sobs), self.rng, temperature=self.temperature)
            sid, sobs = search_api.step(sid, act)
            steps += 1
        return evaluate(sobs, me, self.weights)


REGISTRY["mc"] = MCAgent
