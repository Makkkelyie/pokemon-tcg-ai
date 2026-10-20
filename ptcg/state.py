"""Helpers for reading observation dicts (the raw JSON the engine produces)."""
from .cards import card_db, energy_satisfies, missing_energy

# AreaType
DECK, HAND, DISCARD, ACTIVE, BENCH, PRIZE, STADIUM, ENERGY_AREA, TOOL_AREA, PRE_EVO, PLAYER, LOOKING = range(1, 13)

# OptionType
O_NUMBER, O_YES, O_NO, O_CARD, O_TOOL_CARD, O_ENERGY_CARD, O_ENERGY, O_PLAY, O_ATTACH, O_EVOLVE, \
    O_ABILITY, O_DISCARD, O_RETREAT, O_ATTACK, O_END, O_SKILL, O_SPECIAL = range(17)

# SelectType
S_MAIN, S_CARD, S_ATTACHED, S_CARD_OR_ATTACHED, S_ENERGY, S_SKILL, S_ATTACK, S_EVOLVE, S_COUNT, \
    S_YES_NO, S_SPECIAL = range(11)


def me(obs) -> int:
    return obs["current"]["yourIndex"]


def players(obs):
    cur = obs["current"]
    i = cur["yourIndex"]
    return cur["players"][i], cur["players"][1 - i]


def in_play(ps) -> list[dict]:
    """Active (if face-up) followed by bench Pokémon."""
    return [p for p in ps["active"] if p] + list(ps["bench"])


def card_at(obs, area, index, player):
    """Resolve an (area, index, player) reference to a card/Pokémon dict, or None."""
    try:
        cur = obs["current"]
        ps = cur["players"][player]
        if area == DECK:
            return obs["select"]["deck"][index]
        if area == HAND:
            h = ps["hand"]
            return h[index] if h else None
        if area == DISCARD:
            return ps["discard"][index]
        if area == ACTIVE:
            return ps["active"][index]
        if area == BENCH:
            return ps["bench"][index]
        if area == PRIZE:
            return ps["prize"][index]
        if area == STADIUM:
            return cur["stadium"][index]
        if area == LOOKING:
            return cur["looking"][index]
    except (IndexError, KeyError, TypeError):
        return None
    return None


def prizes_left(ps) -> int:
    return len(ps["prize"])


def best_ready_damage(poke) -> int:
    """Highest printed damage among attacks this Pokémon can currently pay for."""
    info = card_db().get(poke["id"])
    if not info:
        return 0
    best = 0
    for a in info.attacks:
        if energy_satisfies(poke["energies"], a.cost):
            best = max(best, attack_value(a))
    return best


def attack_value(a) -> int:
    """Printed damage, with a rough guess for 0-damage attacks whose text deals damage."""
    if a.damage > 0:
        return a.damage
    t = a.text.lower()
    if "damage" in t and "this attack does" in t:
        return 60
    return 10


def energy_need(poke) -> int:
    """Energy units still missing for this Pokémon's strongest attack."""
    info = card_db().get(poke["id"])
    if not info or not info.attacks:
        return 0
    best = max(info.attacks, key=attack_value)
    return missing_energy(poke["energies"], best.cost)


def effective_damage(attacker_id, dmg, defender) -> int:
    db = card_db()
    a, d = db.get(attacker_id), db.get(defender["id"]) if defender else None
    if not a or not d or dmg <= 0:
        return dmg
    if d.weakness is not None and d.weakness == a.energy_type and a.energy_type != 0:
        dmg *= 2
    if d.resistance is not None and d.resistance == a.energy_type and a.energy_type != 0:
        dmg = max(0, dmg - 30)
    return dmg


def prize_value(poke) -> int:
    info = card_db().get(poke["id"]) if poke else None
    return info.prize_value if info else 1
