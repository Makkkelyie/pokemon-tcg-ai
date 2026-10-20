"""Rule-based agent: scores every legal option, then picks the best ones.

Scores are designed so that options we *want* are > 0 and options we only take
when forced are <= 0; `pick` then chooses between minCount and maxCount of
them. The same scorer is reused as the rollout policy inside the search agent.
"""
import math

from ..cards import BASIC_ENERGY, ITEM, POKEMON, SPECIAL_ENERGY, STADIUM, SUPPORTER, TOOL, card_db
from ..state import (
    ACTIVE, BENCH, DECK, HAND, O_ABILITY, O_ATTACH, O_ATTACK, O_CARD, O_DISCARD, O_END, O_ENERGY,
    O_ENERGY_CARD, O_EVOLVE, O_NO, O_NUMBER, O_PLAY, O_RETREAT, O_SKILL, O_SPECIAL, O_TOOL_CARD, O_YES,
    attack_value, best_ready_damage, card_at, effective_damage, energy_need, in_play, prize_value,
)
from .base import Agent

# SelectContext values (cg/api.py)
C_MAIN, C_SETUP_ACTIVE, C_SETUP_BENCH, C_SWITCH, C_TO_ACTIVE, C_TO_BENCH, C_TO_FIELD, C_TO_HAND, \
    C_DISCARD, C_TO_DECK, C_TO_DECK_BOTTOM, C_TO_PRIZE, C_NOT_MOVE, C_DMG_COUNTER, C_DMG_COUNTER_ANY, \
    C_DAMAGE, C_REMOVE_DMG, C_HEAL, C_EVOLVES_FROM, C_EVOLVES_TO, C_DEVOLVE, C_ATTACH_FROM, C_ATTACH_TO, \
    C_DETACH_FROM, C_LOOK, C_EFFECT_TARGET, C_DISCARD_ENERGY_CARD, C_DISCARD_TOOL, C_SWITCH_ENERGY_CARD, \
    C_DISCARD_CARD_OR_ATTACHED, C_DISCARD_ENERGY, C_TO_HAND_ENERGY, C_TO_DECK_ENERGY, C_SWITCH_ENERGY, \
    C_SKILL_ORDER, C_ATTACK, C_DISABLE_ATTACK, C_EVOLVE, C_DRAW_COUNT, C_DMG_COUNTER_COUNT, \
    C_REMOVE_DMG_COUNT, C_IS_FIRST, C_MULLIGAN, C_ACTIVATE, C_FIRST_EFFECT, C_MORE_DEVOLVE, C_COIN_HEAD, \
    C_AFFECT_SPECIAL, C_RECOVER_SPECIAL = range(49)

# Contexts where moving a card away from us is a cost (prefer minCount, least valuable).
LOSS_CONTEXTS = {C_DISCARD, C_TO_DECK, C_TO_DECK_BOTTOM, C_TO_PRIZE, C_DISCARD_ENERGY_CARD, C_DISCARD_TOOL,
                 C_DISCARD_CARD_OR_ATTACHED, C_DISCARD_ENERGY, C_TO_HAND_ENERGY, C_TO_DECK_ENERGY, C_NOT_MOVE}
# Contexts where we gain the selected cards (prefer maxCount, most valuable).
GAIN_CONTEXTS = {C_TO_HAND, C_TO_BENCH, C_TO_FIELD, C_SETUP_BENCH, C_EVOLVES_TO, C_LOOK}
CONDITION_PRIORITY = {3: 5, 2: 4, 0: 3, 1: 2, 4: 1}  # paralyze > sleep > poison > burn > confuse


class Ctx:
    """Per-decision cached view of the observation."""

    def __init__(self, obs):
        self.obs = obs
        self.db = card_db()
        cur = obs["current"]
        self.cur = cur
        self.i = cur["yourIndex"]
        self.P = cur["players"][self.i]
        self.O = cur["players"][1 - self.i]
        self.select = obs["select"]
        self.hand = self.P["hand"] or []
        self.my_pokes = in_play(self.P)
        self.op_active = self.O["active"][0] if self.O["active"] else None
        self.my_active = self.P["active"][0] if self.P["active"] else None
        names = set()
        for p in self.my_pokes:
            info = self.db.get(p["id"])
            if info:
                names.add(info.name)
        self.my_names = names
        hand_evo_from = set()
        for c in self.hand:
            info = self.db.get(c["id"])
            if info and info.is_pokemon and info.evolves_from:
                hand_evo_from.add(info.evolves_from)
        self.hand_evo_from = hand_evo_from

    def info(self, cid):
        return self.db.get(cid)

    def poke_ref(self, area, index, player):
        return card_at(self.obs, area, index, player)


# ------------------------------------------------------------------ values

def card_value(ctx: Ctx, cid: int) -> float:
    """How much we'd like to have this card in hand right now."""
    info = ctx.info(cid)
    if info is None:
        return 10.0
    t = info.type
    if t == POKEMON:
        if info.stage > 0:
            return 70.0 if info.evolves_from in ctx.my_names else 35.0
        bench_room = ctx.P["benchMax"] - len(ctx.P["bench"])
        base = 60.0 if len(ctx.my_pokes) < 3 else 35.0
        if info.name in ctx.hand_evo_from:
            base += 15
        return base if bench_room > 0 else 15.0
    if t in (BASIC_ENERGY, SPECIAL_ENERGY):
        needs = sum(1 for p in ctx.my_pokes if energy_need(p) > 0)
        return 45.0 if needs else 20.0
    if t == SUPPORTER:
        return 45.0
    if t == ITEM:
        return 38.0
    if t == TOOL:
        return 28.0
    if t == STADIUM:
        return 18.0
    return 20.0


def poke_readiness(ctx: Ctx, poke) -> float:
    """How good this Pokémon would be in the Active Spot."""
    if poke is None:
        return 0.0
    dmg = best_ready_damage(poke)
    need = energy_need(poke)
    hp = poke.get("hp", 0)
    score = dmg + 0.3 * hp - 25 * need
    if ctx.op_active is not None and dmg:
        eff = effective_damage(poke["id"], dmg, ctx.op_active)
        if eff >= ctx.op_active["hp"]:
            score += 150
    return score


def target_value_for_damage(ctx: Ctx, poke, dmg=30) -> float:
    """Value of hitting opponent Pokémon `poke` with ~dmg damage."""
    if poke is None:
        return 0.0
    hp = poke["hp"]
    v = 100.0 - hp / 5.0
    if hp <= dmg:
        v += 120 * prize_value(poke)
    return v


# --------------------------------------------------------------- MAIN phase

def score_main(ctx: Ctx, opt) -> float:
    t = opt["type"]
    if t == O_END:
        return 0.0
    if t == O_ATTACK:
        a = ctx.db.attacks.get(opt.get("attackId"))
        if a is None or ctx.my_active is None:
            return 300.0
        dmg = effective_damage(ctx.my_active["id"], attack_value(a), ctx.op_active)
        s = 300.0 + dmg / 5.0
        if ctx.op_active is not None and dmg >= ctx.op_active["hp"]:
            s += 100 + 20 * prize_value(ctx.op_active)
        return s
    if t == O_EVOLVE:
        info = ctx.info(_hand_id(ctx, opt))
        return 860.0 + (10 * info.stage if info else 0)
    if t == O_PLAY:
        cid = _hand_id(ctx, opt)
        info = ctx.info(cid)
        if info is None:
            return 500.0
        if info.type == POKEMON:
            room = ctx.P["benchMax"] - len(ctx.P["bench"])
            if room <= 0:
                return -10.0
            s = 820.0 - 25 * len(ctx.P["bench"])
            if info.name in ctx.hand_evo_from:
                s += 20
            return s
        if info.type == ITEM:
            return 770.0
        if info.type == SUPPORTER:
            return 740.0
        if info.type == TOOL:
            return 650.0
        if info.type == STADIUM:
            cur_st = ctx.cur["stadium"]
            return 600.0 if not cur_st or cur_st[0]["playerIndex"] != ctx.i else -5.0
        return 500.0
    if t == O_ATTACH:
        cid = _hand_id(ctx, opt)
        info = ctx.info(cid)
        target = card_at(ctx.obs, opt.get("inPlayArea"), opt.get("inPlayIndex"), ctx.i)
        if target is None:
            return 400.0
        need = energy_need(target)
        if info is not None and info.type == TOOL:
            return 640.0 if not target["tools"] else -5.0
        is_active = opt.get("inPlayArea") == ACTIVE
        s = 780.0
        if need <= 0:
            s -= 250  # already powered
        elif need == 1:
            s += 80 if is_active else 50
        else:
            s += 40 if is_active else 25
        tinfo = ctx.info(target["id"])
        if info is not None and tinfo is not None and info.type == BASIC_ENERGY:
            # Prefer matching type for typed costs.
            if any(e == info.energy_type for a in tinfo.attacks for e in a.cost):
                s += 15
        return s
    if t == O_ABILITY:
        return 720.0
    if t == O_RETREAT:
        act = ctx.my_active
        if act is None or ctx.cur.get("retreated"):
            return -50.0
        bench_best = max((poke_readiness(ctx, b) for b in ctx.P["bench"]), default=-1e9)
        if bench_best > poke_readiness(ctx, act) + 60 and best_ready_damage(act) == 0:
            return 690.0
        if act["hp"] <= 0.25 * act["maxHp"] and bench_best > 0:
            return 680.0
        return -50.0
    if t == O_DISCARD:
        return -20.0
    return 100.0


def _hand_id(ctx, opt):
    c = card_at(ctx.obs, opt.get("area", HAND), opt.get("index"), ctx.i)
    return c["id"] if c else None


# ------------------------------------------------------- other selections

def score_card_option(ctx: Ctx, opt, context) -> float:
    t = opt["type"]
    player = opt.get("playerIndex", ctx.i)
    mine = player == ctx.i
    if t in (O_TOOL_CARD, O_ENERGY_CARD):
        poke = card_at(ctx.obs, opt.get("area"), opt.get("index"), player)
        surplus = -energy_need(poke) if poke else 0
        base = 5.0 * surplus
        if poke is not None and opt.get("area") == ACTIVE:
            base -= 5
        return base if mine else -base
    c = card_at(ctx.obs, opt.get("area"), opt.get("index"), player)
    cid = c["id"] if c else None

    if context in (C_SETUP_ACTIVE,):
        return poke_readiness(ctx, _as_poke(ctx, c))
    if context in (C_SWITCH, C_TO_ACTIVE):
        if mine:
            return 100 + poke_readiness(ctx, _as_poke(ctx, c))
        return target_value_for_damage(ctx, c, best_ready_damage(ctx.my_active) if ctx.my_active else 0)
    if context in (C_DMG_COUNTER, C_DMG_COUNTER_ANY, C_DAMAGE, C_EFFECT_TARGET, C_DETACH_FROM, C_DISABLE_ATTACK):
        if mine:
            return -50.0
        return target_value_for_damage(ctx, c)
    if context in (C_REMOVE_DMG, C_HEAL):
        if not mine or c is None:
            return -10.0
        return 1.0 + (c.get("maxHp", 0) - c.get("hp", 0))
    if context in (C_ATTACH_FROM, C_ATTACH_TO):
        # Which of our Pokémon gets the card: the one that most needs energy, active first.
        if c is None:
            return 1.0
        if "energies" in c:
            s = 10.0 * energy_need(c) + (5 if opt.get("area") == ACTIVE else 0)
            return s + 1 if mine else -s
        return card_value(ctx, cid)
    if context in LOSS_CONTEXTS:
        v = card_value(ctx, cid) if cid is not None else 20.0
        return -v if mine else v
    if context in GAIN_CONTEXTS or context in (C_EVOLVES_FROM,):
        if cid is None:
            return 5.0
        if context in (C_TO_BENCH, C_TO_FIELD, C_SETUP_BENCH):
            return 50.0 + 0.1 * (c.get("hp", 0) if isinstance(c, dict) else 0) if mine else -10.0
        return card_value(ctx, cid) if mine else 5.0
    # Unknown context: mild preference for our own cards, opponent's otherwise.
    return 1.0


def _as_poke(ctx, c):
    if c is None:
        return None
    if "energies" in c:
        return c
    info = ctx.info(c["id"])
    return {"id": c["id"], "hp": info.hp if info else 0, "maxHp": info.hp if info else 0, "energies": [], "tools": []}


def score_option(ctx: Ctx, idx: int) -> float:
    sel = ctx.select
    opt = sel["option"][idx]
    t = opt["type"]
    context = sel["context"]
    if context == C_MAIN:
        return score_main(ctx, opt)
    if t == O_NUMBER:
        n = opt.get("number", 0)
        return float(n) if context in (C_DRAW_COUNT, C_DMG_COUNTER_COUNT, C_REMOVE_DMG_COUNT) else -float(n)
    if t in (O_YES, O_NO):
        yes = t == O_YES
        if context == C_IS_FIRST:
            return 1.0 if yes else 0.0
        return 1.0 if yes else 0.0
    if t == O_ATTACK:
        a = ctx.db.attacks.get(opt.get("attackId"))
        if a is None:
            return 1.0
        return float(attack_value(a)) if context != C_DISABLE_ATTACK else float(attack_value(a))
    if t == O_SPECIAL:
        pr = CONDITION_PRIORITY.get(opt.get("specialConditionType"), 0)
        return float(pr)
    if t == O_ENERGY:
        # Paying a cost: keep energies that our Pokémon still needs; pay with surplus.
        poke = card_at(ctx.obs, opt.get("area"), opt.get("index"), opt.get("playerIndex", ctx.i))
        return -1.0 - (energy_need(poke) if poke else 0)
    if t == O_SKILL:
        return -float(idx)  # keep engine order
    if t in (O_CARD, O_TOOL_CARD, O_ENERGY_CARD):
        return score_card_option(ctx, opt, context)
    return 0.0


def score_all(obs) -> list[float]:
    ctx = Ctx(obs)
    return [score_option(ctx, i) for i in range(len(obs["select"]["option"]))]


def pick(select, scores, rng=None, temperature=0.0) -> list[int]:
    """Choose indices from scores respecting min/max counts.

    temperature > 0 samples a softmax-perturbed ranking (used for rollouts).
    """
    n = len(scores)
    lo, hi = select["minCount"], min(select["maxCount"], n)
    if temperature > 0 and rng is not None:
        keyed = [(s / temperature) - math.log(-math.log(rng.random() or 1e-12)) for s in scores]  # Gumbel
    else:
        keyed = scores
    order = sorted(range(n), key=lambda i: keyed[i], reverse=True)
    if select["context"] == C_MAIN or select["type"] in (6, 9, 8):  # single choice selects
        k = max(lo, min(hi, 1))
    else:
        k = sum(1 for s in scores if s > 0)
        k = max(lo, min(hi, k))
    return order[:k]


class HeuristicAgent(Agent):
    name = "heuristic"

    def act(self, obs):
        sel = obs["select"]
        if sel["context"] == C_MAIN and obs["current"]["turnActionCount"] > 60:
            # Safety valve against loops of repeatable actions.
            ends = [i for i, o in enumerate(sel["option"]) if o["type"] in (O_END, O_ATTACK)]
            if ends:
                return [ends[0]]
        return pick(sel, score_all(obs))
