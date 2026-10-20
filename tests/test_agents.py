"""Agents must only ever return legal selections; matches must terminate."""
from ptcg.agents import make_agent
from ptcg.match import play_match

from .conftest import requires_engine


class LegalityChecker:
    def __init__(self, inner):
        self.inner = inner
        self.decisions = 0

    def __call__(self, obs):
        out = self.inner(obs)
        sel = obs.get("select")
        if sel is None:
            assert len(out) == 60
            return out
        n = len(sel["option"])
        assert sel["minCount"] <= len(out) <= sel["maxCount"], (out, sel["minCount"], sel["maxCount"])
        assert len(set(out)) == len(out)
        assert all(isinstance(i, int) and 0 <= i < n for i in out)
        self.decisions += 1
        return out


@requires_engine
def test_card_db(deck):
    from ptcg.cards import card_db
    db = card_db()
    assert len(db.cards) > 1000
    assert all(c in db.cards for c in deck)


@requires_engine
def test_random_match_terminates(deck):
    r = play_match([make_agent("random", deck, seed=1), make_agent("random", deck, seed=2)])
    assert r.winner in (0, 1, 2)
    assert not r.errors


@requires_engine
def test_heuristic_is_legal(deck):
    for s in range(3):
        a = LegalityChecker(make_agent("heuristic", deck, seed=s))
        b = LegalityChecker(make_agent("random", deck, seed=100 + s))
        r = play_match([a, b])
        assert not r.errors, r.errors
        assert a.decisions > 0


@requires_engine
def test_mc_is_legal_and_searches(deck):
    a = make_agent("mc", deck, seed=0, budget=0.01, max_sims=20)
    r = play_match([LegalityChecker(a), LegalityChecker(make_agent("heuristic", deck, seed=1))])
    assert not r.errors, r.errors
    assert a.stats["searched"] > 0
    assert a.stats["errors"] == 0, a.stats


@requires_engine
def test_heuristic_beats_random(deck):
    from ptcg.match import AgentSpec, run_matches
    from ptcg.engine import default_deck_path
    res = run_matches(AgentSpec("heuristic", default_deck_path()), AgentSpec("random", default_deck_path()),
                      n=20, workers=1)
    assert res["match_score"] >= 0.8
