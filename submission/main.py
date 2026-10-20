"""Kaggle entry point (copied to the top level of submission.tar.gz).

kaggle_environments exec()s this file with the agent directory temporarily on
sys.path and uses the *last callable* defined here as the agent, so all
imports happen eagerly at the top and `agent` is defined last.
"""
import os
import sys

import cg.api  # noqa: F401  (engine shipped next to this file)
import cg.game  # noqa: F401
import ptcg

_DIR = os.path.dirname(os.path.dirname(os.path.abspath(ptcg.__file__)))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

with open(os.path.join(_DIR, "deck.csv")) as _f:
    _DECK = [int(x) for x in _f.read().split() if x.strip()][:60]

_CONFIG = {}  # filled in by scripts/build_submission.py
_KIND = _CONFIG.pop("agent", "mc")
if _KIND == "ps":
    from ptcg.agents.policy_search import PolicySearchAgent as _AgentClass
else:
    from ptcg.agents.search import MCAgent as _AgentClass
_AGENT = _AgentClass(_DECK, seed=None, **_CONFIG)


def agent(obs, config=None):
    return _AGENT(obs)
