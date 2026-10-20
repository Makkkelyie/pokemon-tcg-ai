"""Thin dict-based wrapper over the engine's search functions (forward model).

cg.api.search_begin/step build nested dataclasses on every call; for MCTS we
only need the raw JSON, so this calls the C library directly and returns dicts
shaped exactly like the observations the agent receives.
"""
import ctypes
import json

from .engine import ensure_cg

_agent_ptr = None


def _lib():
    return ensure_cg().sim.lib


def _ptr():
    global _agent_ptr
    if _agent_ptr is None:
        _agent_ptr = _lib().AgentStart()
    return _agent_ptr


class SearchError(Exception):
    pass


def _arr(xs):
    return (ctypes.c_int * len(xs))(*xs)


def begin(obs: dict, my_deck, my_prize, op_deck, op_prize, op_hand, op_active, manual_coin=False):
    """Start a determinized search from `obs`. Returns (search_id, observation dict)."""
    sbi = obs.get("search_begin_input")
    if not sbi:
        raise SearchError("observation has no search_begin_input")
    if obs["select"].get("deck") is not None:
        my_deck = []
    bs = _lib().SearchBegin(
        _ptr(), sbi.encode("ascii"), len(sbi),
        _arr(my_deck), _arr(my_prize), _arr(op_deck), _arr(op_prize),
        _arr(op_hand), _arr(op_active), int(manual_coin),
    )
    res = json.loads(bs.decode())
    if res["error"] != 0:
        raise SearchError(f"SearchBegin error {res['error']}")
    st = res["state"]
    return st["searchId"], st["observation"]


def step(search_id: int, select: list[int]):
    bs = _lib().SearchStep(_ptr(), search_id, _arr(select), len(select))
    res = json.loads(bs.decode())
    if res["error"] != 0:
        raise SearchError(f"SearchStep error {res['error']}")
    st = res["state"]
    return st["searchId"], st["observation"]


def release(search_id: int):
    _lib().SearchRelease(_ptr(), search_id)


def end():
    """Free all search states (call once per decision)."""
    _lib().SearchEnd(_ptr())
