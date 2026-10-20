"""Hidden-information tracking and determinization for the search API.

search_begin needs concrete card ids for every hidden zone (our deck order and
prizes, the opponent's deck/prizes/hand and a face-down Active). We know our
own 60-card list exactly; for the opponent we keep, per card id, the largest
number of copies ever visible at once (board, attachments, discard, stadium)
over the whole BO3 episode -- the same deck is used in all games -- and fill
the rest with basic Energy of their most common type.
"""
from collections import Counter

from .cards import BASIC_ENERGY, card_db
from .state import in_play


def _visible_ids(ps, include_hand=True) -> Counter:
    c = Counter()
    for p in in_play(ps):
        c[p["id"]] += 1
        for e in p.get("energyCards", []):
            c[e["id"]] += 1
        for t in p.get("tools", []):
            c[t["id"]] += 1
        for pe in p.get("preEvolution", []):
            c[pe["id"]] += 1
    for d in ps["discard"]:
        c[d["id"]] += 1
    for pr in ps["prize"]:
        if pr:
            c[pr["id"]] += 1
    if include_hand and ps.get("hand"):
        for h in ps["hand"]:
            c[h["id"]] += 1
    return c


class Belief:
    def __init__(self, my_deck: list[int]):
        self.my_deck = Counter(my_deck)
        self.my_filler = self.my_deck.most_common(1)[0][0]
        self.op_seen = Counter()
        db = card_db()
        self.basic_energy_of_type = {}
        for c in db.cards.values():
            if c.type == BASIC_ENERGY and c.energy_type not in self.basic_energy_of_type:
                self.basic_energy_of_type[c.energy_type] = c.id
        self.my_basic = next((cid for cid in my_deck if db[cid].is_pokemon and db[cid].basic), None)

    def observe(self, obs):
        cur = obs.get("current")
        if not cur:
            return
        i = cur["yourIndex"]
        op = cur["players"][1 - i]
        vis = _visible_ids(op, include_hand=False)
        for st in cur.get("stadium") or []:
            if st["playerIndex"] == 1 - i:
                vis[st["id"]] += 1
        for cid, n in vis.items():
            if n > self.op_seen[cid]:
                self.op_seen[cid] = n
        for log in obs.get("logs") or []:
            # Cards the opponent played/attached/evolved are revealed even if they leave play.
            if log.get("playerIndex") == 1 - i and log.get("type") in (10, 11, 12) and log.get("cardId"):
                cid = log["cardId"]
                if self.op_seen[cid] == 0:
                    self.op_seen[cid] = 1

    # ------------------------------------------------------------------
    def _op_filler(self) -> int:
        db = card_db()
        counts = Counter()
        for cid, n in self.op_seen.items():
            info = db.get(cid)
            if info and info.type == BASIC_ENERGY:
                counts[cid] += n
        if counts:
            return counts.most_common(1)[0][0]
        types = Counter()
        for cid, n in self.op_seen.items():
            info = db.get(cid)
            if info and info.is_pokemon:
                for a in info.attacks:
                    for e in a.cost:
                        if e:
                            types[e] += n
        if types:
            t = types.most_common(1)[0][0]
            if t in self.basic_energy_of_type:
                return self.basic_energy_of_type[t]
        return self.my_filler

    def determinize(self, obs, rng):
        """Return kwargs for search_api.begin with a random consistent assignment."""
        cur = obs["current"]
        i = cur["yourIndex"]
        me, op = cur["players"][i], cur["players"][1 - i]

        # --- our hidden cards: deck list minus everything we can see
        vis = _visible_ids(me)
        for st in cur.get("stadium") or []:
            if st["playerIndex"] == i:
                vis[st["id"]] += 1
        for lk in cur.get("looking") or []:
            if lk and lk.get("playerIndex", i) == i:
                vis[lk["id"]] += 1
        sel_deck = obs["select"].get("deck")
        if sel_deck:
            for c in sel_deck:
                vis[c["id"]] += 1
        pool = []
        for cid, n in self.my_deck.items():
            pool += [cid] * max(0, n - vis[cid])
        rng.shuffle(pool)
        n_fd_prize = sum(1 for p in me["prize"] if p is None)
        need = (0 if sel_deck else me["deckCount"]) + n_fd_prize
        pool = (pool + [self.my_filler] * need)[:need]
        my_prize_fd = pool[:n_fd_prize]
        my_deck = pool[n_fd_prize:]
        if sel_deck:
            my_deck = []
        my_prize = []
        it = iter(my_prize_fd)
        for p in me["prize"]:
            my_prize.append(p["id"] if p else next(it))

        # --- opponent's hidden cards
        ovis = _visible_ids(op, include_hand=False)
        hidden = []
        for cid, n in self.op_seen.items():
            hidden += [cid] * max(0, n - ovis[cid])
        rng.shuffle(hidden)
        n_op_fd_prize = sum(1 for p in op["prize"] if p is None)
        face_down_active = len(op["active"]) > 0 and op["active"][0] is None
        n_hidden = op["deckCount"] + op["handCount"] + n_op_fd_prize
        filler = self._op_filler()
        hidden = (hidden[:n_hidden] + [filler] * n_hidden)[:n_hidden]
        rng.shuffle(hidden)
        op_hand = hidden[: op["handCount"]]
        rest = hidden[op["handCount"]:]
        op_prize_fd = rest[:n_op_fd_prize]
        op_deck = rest[n_op_fd_prize:]
        op_prize = []
        it = iter(op_prize_fd)
        for p in op["prize"]:
            op_prize.append(p["id"] if p else next(it))
        op_active = []
        if face_down_active:
            db = card_db()
            basics = [c for c in self.op_seen if db[c].is_pokemon and db[c].basic]
            op_active = [rng.choice(basics) if basics else self.my_basic]
        if cur["turn"] == 0 and self.my_basic is not None:
            # Setup needs at least one Basic Pokémon in the opponent's deck.
            db = card_db()
            if not any(db[c].is_pokemon and db[c].basic for c in op_deck) and op_deck:
                op_deck[0] = self.my_basic
        return dict(my_deck=my_deck, my_prize=my_prize, op_deck=op_deck, op_prize=op_prize,
                    op_hand=op_hand, op_active=op_active)
