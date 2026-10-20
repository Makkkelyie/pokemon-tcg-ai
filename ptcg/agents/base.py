"""Agent base class.

Kaggle calls `agent(obs_dict)` once per decision. When obs["select"] is None the
agent must return its 60-card deck; otherwise a list of option indices with
minCount <= len <= maxCount and no duplicates.
"""
import random


class Agent:
    name = "base"

    def __init__(self, deck: list[int], seed: int | None = None):
        if len(deck) != 60:
            raise ValueError("deck must have 60 cards")
        self.deck = list(deck)
        self.rng = random.Random(seed)

    def __call__(self, obs: dict) -> list[int]:
        if obs.get("select") is None:
            self.new_match()
            return list(self.deck)
        return self.act(obs)

    def new_match(self):
        """Hook: called when a new BO3 episode starts (deck request)."""

    def act(self, obs: dict) -> list[int]:
        raise NotImplementedError


def random_choice(select: dict, rng: random.Random) -> list[int]:
    n = len(select["option"])
    k = rng.randint(select["minCount"], select["maxCount"])
    return rng.sample(range(n), k)


class RandomAgent(Agent):
    """Uniformly random legal action (number of picks also random)."""

    name = "random"

    def act(self, obs):
        return random_choice(obs["select"], self.rng)


class FirstAgent(Agent):
    """Always picks the first maxCount options (the env's 'first' builtin)."""

    name = "first"

    def act(self, obs):
        return list(range(obs["select"]["maxCount"]))
