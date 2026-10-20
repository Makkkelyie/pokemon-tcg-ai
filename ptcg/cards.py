"""Card / attack metadata from the engine, as plain dicts keyed by id."""
from dataclasses import dataclass, field
from functools import lru_cache

from .engine import ensure_cg

# Enum values from cg/api.py (kept as ints for speed).
POKEMON, ITEM, TOOL, SUPPORTER, STADIUM, BASIC_ENERGY, SPECIAL_ENERGY = range(7)
COLORLESS = 0


@dataclass
class AttackInfo:
    id: int
    name: str
    damage: int
    cost: list[int]
    text: str


@dataclass
class CardInfo:
    id: int
    name: str
    type: int
    hp: int
    retreat: int
    weakness: int | None
    resistance: int | None
    energy_type: int
    basic: bool
    stage: int  # 0 basic, 1, 2 (for Pokémon)
    prize_value: int  # prizes the opponent takes on KO
    tera: bool
    evolves_from: str | None
    attacks: list[AttackInfo] = field(default_factory=list)
    skills: list[str] = field(default_factory=list)

    @property
    def is_pokemon(self) -> bool:
        return self.type == POKEMON

    @property
    def is_energy(self) -> bool:
        return self.type in (BASIC_ENERGY, SPECIAL_ENERGY)

    @property
    def max_damage(self) -> int:
        return max((a.damage for a in self.attacks), default=0)


class CardDB:
    def __init__(self):
        cg = ensure_cg()
        api = cg.api
        self.attacks = {
            a.attackId: AttackInfo(a.attackId, a.name, a.damage, list(a.energies), a.text)
            for a in api.all_attack()
        }
        self.cards: dict[int, CardInfo] = {}
        for c in api.all_card_data():
            stage = 2 if c.stage2 else 1 if c.stage1 else 0
            prize = 3 if c.megaEx else 2 if c.ex else 1
            self.cards[c.cardId] = CardInfo(
                id=c.cardId,
                name=c.name,
                type=int(c.cardType),
                hp=c.hp,
                retreat=c.retreatCost,
                weakness=c.weakness,
                resistance=c.resistance,
                energy_type=int(c.energyType),
                basic=c.basic,
                stage=stage,
                prize_value=prize,
                tera=c.tera,
                evolves_from=c.evolvesFrom,
                attacks=[self.attacks[i] for i in c.attacks if i in self.attacks],
                skills=[s.text for s in c.skills],
            )
        self.by_name: dict[str, list[int]] = {}
        for c in self.cards.values():
            self.by_name.setdefault(c.name, []).append(c.id)

    def __getitem__(self, cid: int) -> CardInfo:
        return self.cards[cid]

    def get(self, cid):
        return self.cards.get(cid)

    def evolves_into(self, base_id: int, evo_id: int) -> bool:
        evo = self.cards.get(evo_id)
        base = self.cards.get(base_id)
        return bool(evo and base and evo.evolves_from == base.name)


@lru_cache(maxsize=1)
def card_db() -> CardDB:
    return CardDB()


def energy_satisfies(attached: list[int], cost: list[int]) -> bool:
    """True if attached energy units pay `cost` (colorless = any; RAINBOW pays any type)."""
    pool = list(attached)
    need_colorless = 0
    for e in cost:
        if e == COLORLESS:
            need_colorless += 1
            continue
        if e in pool:
            pool.remove(e)
        elif 10 in pool:  # RAINBOW
            pool.remove(10)
        elif 11 in pool and e in (5, 7):  # TEAM_ROCKET: psychic or darkness
            pool.remove(11)
        else:
            return False
    return len(pool) >= need_colorless


def missing_energy(attached: list[int], cost: list[int]) -> int:
    """Rough number of extra energy units needed to pay `cost`."""
    if energy_satisfies(attached, cost):
        return 0
    return max(1, len(cost) - len(attached))
