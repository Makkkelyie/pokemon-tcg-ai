"""Print a human-readable trace of one game (for debugging agent decisions)."""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ptcg.agents import make_agent  # noqa: E402
from ptcg.cards import card_db  # noqa: E402
from ptcg.engine import default_deck_path, ensure_cg, load_deck  # noqa: E402
from ptcg.state import card_at  # noqa: E402

OPT = ["NUMBER", "YES", "NO", "CARD", "TOOL_CARD", "ENERGY_CARD", "ENERGY", "PLAY", "ATTACH", "EVOLVE",
       "ABILITY", "DISCARD", "RETREAT", "ATTACK", "END", "SKILL", "SPECIAL"]
AREA = {1: "deck", 2: "hand", 3: "discard", 4: "active", 5: "bench", 6: "prize", 7: "stadium", 12: "looking"}


def describe(obs, opt):
    db = card_db()
    t = OPT[opt["type"]]
    me = obs["current"]["yourIndex"]
    parts = [t]
    if "area" in opt and opt.get("index") is not None:
        c = card_at(obs, opt["area"], opt["index"], opt.get("playerIndex", me))
        who = "" if opt.get("playerIndex", me) == me else "opp "
        parts.append(f"{who}{AREA.get(opt['area'], opt['area'])}:{db[c['id']].name if c else '?'}")
    elif t == "PLAY":
        c = card_at(obs, 2, opt["index"], me)
        parts.append(db[c["id"]].name if c else "?")
    if "inPlayArea" in opt:
        c = card_at(obs, opt["inPlayArea"], opt["inPlayIndex"], me)
        parts.append(f"-> {AREA[opt['inPlayArea']]}:{db[c['id']].name if c else '?'}")
    if opt.get("attackId") is not None:
        a = db.attacks.get(opt["attackId"])
        parts.append(f"{a.name}({a.damage})" if a else str(opt["attackId"]))
    if opt.get("number") is not None:
        parts.append(str(opt["number"]))
    return " ".join(parts)


def board(obs):
    db = card_db()
    out = []
    for i, ps in enumerate(obs["current"]["players"]):
        act = ps["active"][0] if ps["active"] and ps["active"][0] else None
        a = f"{db[act['id']].name} {act['hp']}/{act['maxHp']} E{len(act['energies'])}" if act else "-"
        b = ", ".join(f"{db[p['id']].name} {p['hp']} E{len(p['energies'])}" for p in ps["bench"])
        out.append(f"  P{i}: prizes {len(ps['prize'])} hand {ps['handCount']} deck {ps['deckCount']} | {a} | {b}")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a", nargs="?", default="heuristic")
    ap.add_argument("b", nargs="?", default="heuristic")
    ap.add_argument("--deck-a", default=default_deck_path())
    ap.add_argument("--deck-b", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    game = ensure_cg().game
    da, db_ = load_deck(args.deck_a), load_deck(args.deck_b or args.deck_a)
    agents = [make_agent(args.a, da, seed=args.seed), make_agent(args.b, db_, seed=args.seed + 1)]
    for ag, d in zip(agents, (da, db_)):
        ag({"select": None})
    obs, _ = game.battle_start(da, db_)
    last_turn = -1
    while obs["current"]["result"] < 0:
        cur = obs["current"]
        if cur["turn"] != last_turn:
            last_turn = cur["turn"]
            print(f"--- turn {cur['turn']}")
            print(board(obs))
        p = cur["yourIndex"]
        act = agents[p](obs)
        sel = obs["select"]
        chosen = [describe(obs, sel["option"][i]) for i in act]
        if len(sel["option"]) > 1:
            print(f"P{p} ctx{sel['context']} ({len(sel['option'])} opts) -> {chosen}")
        obs = game.battle_select(act)
    print("RESULT", obs["current"]["result"])
    print(board(obs))
    game.battle_finish()


if __name__ == "__main__":
    main()
