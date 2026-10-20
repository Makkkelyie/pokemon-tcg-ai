"""Static evaluation of a (possibly simulated) game state, in [-1, 1] for `me`."""
import math

from .cards import card_db
from .state import attack_value, best_ready_damage, effective_damage, in_play, prize_value

DEFAULT_WEIGHTS = {
    "prize": 1.0,         # per prize card taken (difference)
    "ko_progress": 0.35,  # damage fraction on opponent Pokémon x prize value
    "board": 0.12,        # per Pokémon in play
    "energy": 0.08,       # per useful attached energy unit
    "stage": 0.10,        # per evolution stage in play
    "ready": 0.25,        # active can attack now
    "can_ko": 0.30,       # active can KO opponent's active (x prize value)
    "threat": 0.30,       # opponent's active can KO ours (x prize value)
    "hand": 0.03,         # per card in hand (capped)
    "no_bench": 0.6,      # penalty: no benched Pokémon
    "deck_low": 0.15,     # per card below 5 left in deck
}


def _side(ps, w):
    s = 0.0
    db = card_db()
    pokes = in_play(ps)
    for p in pokes:
        info = db.get(p["id"])
        s += w["board"]
        if info:
            s += w["stage"] * info.stage
            best = max((len(a.cost) for a in info.attacks), default=0)
            s += w["energy"] * min(len(p["energies"]), best)
    s += w["hand"] * min(ps["handCount"], 8)
    if not ps["bench"]:
        s -= w["no_bench"]
    if ps["deckCount"] < 5:
        s -= w["deck_low"] * (5 - ps["deckCount"])
    return s


def _ko_progress(ps):
    t = 0.0
    for p in in_play(ps):
        if p.get("maxHp"):
            t += (1 - p["hp"] / p["maxHp"]) * prize_value(p)
    return t


def _threat(att, dfd, extra_energy=1):
    """Could `att` (with one more energy attached) KO `dfd` with a printed attack?"""
    if att is None or dfd is None:
        return False
    info = card_db().get(att["id"])
    if not info:
        return False
    n = len(att["energies"]) + extra_energy
    for a in info.attacks:
        if len(a.cost) <= n and effective_damage(att["id"], attack_value(a), dfd) >= dfd["hp"]:
            return True
    return False


def evaluate(obs, me: int, w=DEFAULT_WEIGHTS) -> float:
    cur = obs["current"]
    if cur["result"] >= 0:
        return 0.0 if cur["result"] == 2 else (1.0 if cur["result"] == me else -1.0)
    P, O = cur["players"][me], cur["players"][1 - me]
    s = w["prize"] * ((6 - len(P["prize"])) - (6 - len(O["prize"])))
    s += w["ko_progress"] * (_ko_progress(O) - _ko_progress(P))
    s += _side(P, w) - _side(O, w)
    my_act = P["active"][0] if P["active"] else None
    op_act = O["active"][0] if O["active"] else None
    if my_act:
        dmg = best_ready_damage(my_act)
        if dmg:
            s += w["ready"]
            if op_act and effective_damage(my_act["id"], dmg, op_act) >= op_act["hp"]:
                s += w["can_ko"] * prize_value(op_act)
        if _threat(op_act, my_act):
            s -= w["threat"] * prize_value(my_act)
    if op_act:
        if best_ready_damage(op_act):
            s -= w["ready"]
    return math.tanh(s / 3.0)
