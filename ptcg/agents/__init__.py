from .base import Agent, FirstAgent, RandomAgent
from .heuristic import HeuristicAgent

REGISTRY = {
    "random": RandomAgent,
    "first": FirstAgent,
    "heuristic": HeuristicAgent,
}


def make_agent(name: str, deck: list[int], seed=None, **kwargs) -> Agent:
    if name not in REGISTRY:
        # Lazy imports keep the heuristic agent free of search dependencies.
        if name.startswith("mc") or name.startswith("search"):
            from . import search  # noqa: F401  (registers itself)
    return REGISTRY[name](deck, seed=seed, **kwargs)
