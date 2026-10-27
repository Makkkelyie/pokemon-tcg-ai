"""The policy-guided search agent: legal play, real searching, meta guess pool, submission load."""
import os
import subprocess
import sys
import tarfile

from ptcg.agents import make_agent
from ptcg.match import play_match

from .conftest import ROOT, requires_dipam
from .test_agents import LegalityChecker


@requires_dipam
def test_ps_legal_and_searches(deck):
    a = make_agent("ps", deck, seed=0, budget=0.05, max_sims=6, dets=2)
    b = make_agent("heuristic", deck, seed=1)
    r = play_match([LegalityChecker(a), LegalityChecker(b)])
    assert not r.errors, r.errors
    assert a.stats["searched"] > 0
    assert a.stats["errors"] == 0, a.stats


@requires_dipam
def test_ps_policy_only_is_fast(deck):
    a = make_agent("ps", deck, seed=0, mode="none")
    r = play_match([a, make_agent("random", deck, seed=1)])
    assert not r.errors
    assert a.stats["searched"] == 0


@requires_dipam
def test_ko_guard_replaces_end_only_with_a_ko(deck):
    a = make_agent("ps", deck, seed=0, mode="none")
    end, atk, play = {"type": 14}, {"type": 13, "attackId": 1}, {"type": 7, "index": 0}
    obs = {"search_begin_input": "x", "select": {"type": 0, "option": [play, atk, end]}}
    a._attack_kos = lambda o, i: True
    assert a._guard_ko(obs, [2]) == [1]
    assert a._guard_ko(obs, [0]) == [0]  # only an END is ever overridden
    a._attack_kos = lambda o, i: False
    assert a._guard_ko(obs, [2]) == [2]
    assert a.stats["ko_guard"] == 1


@requires_dipam
def test_guess_pool_matches_revealed_cards():
    import random
    from collections import Counter

    from ptcg.meta import GuessPool

    class FakeTracker:
        opp_known_hand, opp_known_deck, _last_obs = {}, {}, None

    decks = [[1] * 60, [2] * 60]
    pool = GuessPool.__new__(GuessPool)
    pool.decks, pool.weights = decks, [1.0, 1000.0]
    pool.counters = [Counter(d) for d in decks]
    t = FakeTracker()
    t.opp_known_hand = {10: 1, 11: 1}  # two revealed copies of card 1
    rng = random.Random(0)
    assert all(pool.match(t, rng) == 0 for _ in range(20))  # evidence beats the usage prior
    t.opp_known_hand = {}
    assert sum(pool.match(t, rng) for _ in range(200)) > 150  # no evidence: prior dominates


@requires_dipam
def test_ps_submission_loads(tmp_path):
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    from build_submission import build
    from ptcg.engine import default_deck_path
    _, tar_path = build(default_deck_path(), {"agent": "ps", "budget": 0.02, "max_sims": 4},
                        out_dir=str(tmp_path))
    agent_dir = tmp_path / "agent"
    with tarfile.open(tar_path) as t:
        t.extractall(agent_dir, filter="data")
    assert (agent_dir / "model.pt").exists() and (agent_dir / "dipam_ptcg").is_dir()
    code = f"""
import sys
src = open(r'{agent_dir / "main.py"}').read()
sys.path.append(r'{agent_dir}')
env = {{}}
exec(compile(src, 'main.py', 'exec'), env)
sys.path.pop()
agent = [v for v in env.values() if callable(v)][-1]
assert agent.__name__ == 'agent'
assert type(env['_AGENT']).__name__ == 'PolicySearchAgent'
assert len(agent({{'select': None}})) == 60
print('LOADED')
"""
    out = subprocess.run([sys.executable, "-I", "-c", code], cwd=str(tmp_path), capture_output=True, text=True)
    assert "LOADED" in out.stdout, out.stderr[-3000:]
