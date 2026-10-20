"""Determinizations must be accepted by the engine's search API."""
import random

from ptcg import search_api
from ptcg.agents import make_agent
from ptcg.belief import Belief
from ptcg.engine import ensure_cg

from .conftest import requires_engine


@requires_engine
def test_determinize_shapes_and_search_begin(deck):
    game = ensure_cg().game
    rng = random.Random(0)
    beliefs = [Belief(deck), Belief(deck)]
    pol = [make_agent("heuristic", deck, seed=0), make_agent("heuristic", deck, seed=1)]
    obs, _ = game.battle_start(deck, deck)
    checked = 0
    try:
        while obs["current"]["result"] < 0 and checked < 60:
            p = obs["current"]["yourIndex"]
            beliefs[p].observe(obs)
            det = beliefs[p].determinize(obs, rng)
            me, op = obs["current"]["players"][p], obs["current"]["players"][1 - p]
            if obs["select"].get("deck") is None:
                assert len(det["my_deck"]) == me["deckCount"]
            assert len(det["my_prize"]) == len(me["prize"])
            assert len(det["op_deck"]) == op["deckCount"]
            assert len(det["op_hand"]) == op["handCount"]
            assert len(det["op_prize"]) == len(op["prize"])
            sid, sobs = search_api.begin(obs, **det)
            assert len(sobs["select"]["option"]) == len(obs["select"]["option"])
            search_api.end()
            checked += 1
            obs = game.battle_select(pol[p](obs))
    finally:
        game.battle_finish()
    assert checked > 10
