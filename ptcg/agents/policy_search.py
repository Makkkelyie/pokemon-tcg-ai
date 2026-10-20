"""Policy-guided determinized search on top of a pretrained policy/value network.

The network, its observation encoder and its card tracker come from Dipam
Chakraborty's public baseline (MIT; vendored as `dipam_ptcg`, see
scripts/setup_dipam.py). This module is the search driver around it:

  * gating: search only single-pick decisions where the policy is unsure
    (top-1 minus top-2 prior below `gate`), otherwise play the policy argmax;
  * candidates: the top-`k` actions by prior;
  * worlds: `dets` determinizations of the hidden cards per decision, the
    opponent's unknown cards drawn from a deck chosen from a *meta pool* of
    ladder decklists (ptcg/meta) by how well it explains the opponent's
    revealed cards, instead of a fixed guess;
  * allocation: PUCT (prior-weighted UCB, as in AlphaZero) or a uniform grid,
    with common random numbers so candidates are compared on the same worlds;
  * rollouts: `rollout` = "net" (sampled policy), "greedy" (policy argmax),
    "heuristic" (this repo's rule-based scorer, ~50x cheaper) or "none"
    (value head right after the action), stopped after `depth` turn
    boundaries; leaf = the value head;
  * override rule: deviate from the policy argmax only with >= `min_visits`
    rollouts and a mean advantage of at least `margin`;
  * time manager: per-decision budget from the remaining 600 s game pool.
"""
import math
import os
import sys
import time

from ..engine import REPO_ROOT, ensure_cg
from .base import Agent
from .heuristic import pick, score_all


def _vendor_dirs():
    env = os.environ.get("PTCG_DIPAM_DIR")
    out = [env] if env else []
    out.append(os.path.join(REPO_ROOT, "data", "vendor", "dipam"))  # local checkout
    out.append(REPO_ROOT)  # inside a submission bundle: dipam_ptcg/ sits next to ptcg/
    return out


def _import_dipam():
    ensure_cg()
    try:
        import dipam_ptcg  # noqa: F401
    except ImportError:
        for d in _vendor_dirs():
            if os.path.isdir(os.path.join(d, "dipam_ptcg")):
                sys.path.insert(0, d)
                break
    from dipam_ptcg.rl.search_agent import SearchAgent, search_begin_raw, search_end, search_step_raw
    from dipam_ptcg.tracker import InfoTracker
    import dipam_ptcg
    base = os.path.dirname(os.path.dirname(os.path.abspath(dipam_ptcg.__file__)))
    return SearchAgent, InfoTracker, search_begin_raw, search_step_raw, search_end, base


_NET_CACHE = {}


class PolicySearchAgent(Agent):
    name = "ps"

    def __init__(self, deck, seed=None, mode="puct", budget=0.5, k=4, dets=4, gate=1.0, depth=1,
                 rollout="greedy", max_steps=60, c_puct=1.5, margin=0.0, min_visits=2, max_sims=64,
                 guess="meta", time_frac=0.04, reserve=60.0, threads=2, play_temp=0.0):
        super().__init__(deck, seed)
        (SearchAgent, self.InfoTracker, self.sbegin, self.sstep, self.send, base) = _import_dipam()
        import torch
        self.torch = torch
        torch.set_num_threads(int(threads))
        key = base
        if key not in _NET_CACHE:
            _NET_CACHE[key] = SearchAgent(os.path.join(base, "model.pt"), os.path.join(base, "deck.csv"),
                                          cfg=None, seed=0, device="cpu", data_dir=base, temp=0.0)
        self.net = _NET_CACHE[key]
        self.mode = mode
        self.budget = float(budget)
        self.k = int(k)
        self.dets = int(dets)
        self.gate = float(gate)
        self.depth = int(depth)
        self.rollout = rollout
        self.max_steps = int(max_steps)
        self.c_puct = float(c_puct)
        self.margin = float(margin)
        self.min_visits = int(min_visits)
        self.max_sims = int(max_sims)
        self.guess = guess
        self.time_frac = float(time_frac)
        self.reserve = float(reserve)
        self.play_temp = float(play_temp)
        self.pool = self._load_pool() if guess == "meta" else None
        self.stats = {"decisions": 0, "searched": 0, "overrides": 0, "sims": 0, "errors": 0}
        self._reset_game_state()

    # ---------------------------------------------------------------- setup
    def _load_pool(self):
        from ..meta import load_guess_pool
        return load_guess_pool()

    def _reset_game_state(self):
        self.tracker = self.InfoTracker(self.deck)
        self.forced_run = 0
        self.round = None

    def new_match(self):
        self._reset_game_state()

    def _bind_net(self):
        """The cached network object is shared; point its per-agent state at ours."""
        net = self.net
        net.tracker = self.tracker
        net.deck = list(self.deck)
        net.filler = list(self.deck)
        net.rng = self.rng
        if self.pool:
            net.guess_pool = self.pool.decks
            net._guess_basics = self.pool.basics

    # ----------------------------------------------------------------- play
    def act(self, obs):
        cur = obs["current"]
        if cur.get("round") != self.round:  # new game in the BO3: our deck knowledge restarts
            self.round = cur.get("round")
            self.tracker = self.InfoTracker(self.deck)
            self.forced_run = 0
        self._bind_net()
        net = self.net
        self.tracker.update(obs)
        sel = obs["select"]
        self.stats["decisions"] += 1
        fa = net._forced(sel)
        if fa is not None:
            self.forced_run += 1
            return fa
        fr, self.forced_run = self.forced_run, 0
        n = len(sel["option"])
        if self.mode == "none" or sel["maxCount"] != 1 or n < 2 or not obs.get("search_begin_input"):
            return net._decompose(obs, self.tracker, fr, sample=self.play_temp > 0, temp=self.play_temp)
        logits, v_root = net._logits(obs, [], False, fr, self.tracker)
        probs = self.torch.softmax(logits, dim=-1)[:n]
        order = sorted(range(n), key=lambda i: -float(probs[i]))
        cands = order[: self.k]
        if len(cands) < 2 or float(probs[cands[0]]) - float(probs[cands[1]]) >= self.gate:
            return [cands[0]]
        budget = self._decision_budget(obs)
        if budget <= 0:
            return [cands[0]]
        try:
            best = self._search(obs, cands, [float(probs[a]) for a in cands], float(v_root), budget)
        except Exception:
            self.stats["errors"] += 1
            best = cands[0]
        finally:
            try:
                self.send()
            except Exception:
                pass
        self.stats["searched"] += 1
        if best != cands[0]:
            self.stats["overrides"] += 1
        return [best]

    def _decision_budget(self, obs):
        rem = obs.get("remainingOverageTime")
        if rem is None:
            return self.budget
        rem = float(rem)
        if rem <= self.reserve:
            return 0.0
        return min(self.budget, self.time_frac * (rem - self.reserve))

    # --------------------------------------------------------------- search
    def _world(self, obs):
        """One determinization: (search_begin args, guessed opponent deck)."""
        if not self.pool:
            return self.net._determinize(obs)
        net = self.net
        gi = self.pool.match(self.tracker, self.rng)
        orig_match, orig_cfg = net._match_guess, net.cfg

        class _Cfg:
            guess_match = 1
        try:
            net._match_guess = lambda: gi
            net.cfg = _Cfg()
            return net._determinize(obs)
        finally:
            net._match_guess, net.cfg = orig_match, orig_cfg

    def _search(self, obs, cands, priors, v_root, budget):
        me = obs["current"]["yourIndex"]
        root_turn = obs["current"]["turn"]
        deadline = time.perf_counter() + budget
        z = sum(priors) or 1.0
        P = [p / z for p in priors]
        k = len(cands)
        N = [0] * k
        W = [0.0] * k
        worlds = [(self._world(obs), self.rng.randrange(1 << 31)) for _ in range(self.dets)]
        sims = 0
        while sims < self.max_sims:
            if sims >= k and time.perf_counter() >= deadline:
                break
            if self.mode == "uniform" or sims < k:
                j = min(range(k), key=lambda a: (N[a], a))
            else:
                sq = math.sqrt(sims)
                j = max(range(k), key=lambda a: (W[a] / N[a] if N[a] else v_root)
                        + self.c_puct * P[a] * sq / (1 + N[a]))
            (det, guess), wseed = worlds[N[j] % self.dets]
            v = self._simulate(obs, det, guess, cands[j], me, root_turn, wseed)
            N[j] += 1
            W[j] += v
            sims += 1
        self.stats["sims"] += sims
        mean = [W[a] / N[a] if N[a] else -2.0 for a in range(k)]
        if self.mode == "puct":
            best = max(range(k), key=lambda a: (N[a], mean[a]))
        else:
            best = max(range(k), key=lambda a: (mean[a], P[a]))
        if best != 0 and (N[best] < self.min_visits or not N[0] or mean[best] < mean[0] + self.margin):
            best = 0
        return cands[best]

    def _simulate(self, obs, det, guess, action, me, root_turn, wseed):
        net = self.net
        root = self.sbegin(obs, *det)
        try:
            st = self.sstep(root["searchId"], [action])
        except RuntimeError:
            return -1.0
        self.torch.manual_seed(wseed)
        trs = net._sim_trackers(me, guess)
        sobs = st["observation"]
        stop_turn = root_turn + self.depth - 1
        for _ in range(self.max_steps if self.rollout != "none" else 0):
            cur = sobs["current"]
            if cur["result"] != -1:
                break
            if cur["turn"] > stop_turn:
                break
            seat = cur["yourIndex"]
            sel = sobs["select"]
            if self.rollout == "heuristic":
                ans = pick(sel, score_all(sobs))
            else:
                trs[seat].update(sobs)
                fa = net._forced(sel)
                ans = fa if fa is not None else net._decompose(
                    sobs, trs[seat], 0, sample=self.rollout == "net", temp=1.0 if self.rollout == "net" else 0.0)
            try:
                st = self.sstep(st["searchId"], ans)
            except RuntimeError:
                return 0.0
            sobs = st["observation"]
        cur = sobs["current"]
        if cur["result"] != -1:
            return 0.0 if cur["result"] == 2 else (1.0 if cur["result"] == me else -1.0)
        seat = cur["yourIndex"]
        trs[seat].update(sobs)
        _, v = net._logits(sobs, [], False, 0, trs[seat])
        return float(v) if seat == me else -float(v)


def register():
    from . import REGISTRY
    REGISTRY["ps"] = PolicySearchAgent


register()
