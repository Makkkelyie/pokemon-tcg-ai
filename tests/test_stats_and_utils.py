from ptcg.cards import energy_satisfies
from ptcg.match import _bo_decided, wilson


def test_wilson_bounds():
    lo, hi = wilson(50, 100)
    assert 0.40 < lo < 0.5 < hi < 0.60
    assert wilson(0, 10)[0] == 0.0
    assert wilson(10, 10)[1] == 1.0
    assert wilson(0, 0) == (0.0, 1.0)


def test_bo3_decision():
    assert _bo_decided([2, 0, 0], 3) == 0
    assert _bo_decided([1, 1, 0], 3) == -1
    assert _bo_decided([1, 2, 0], 3) == 1
    assert _bo_decided([1, 1, 1], 3) == 2


def test_energy_cost():
    W, G, C, RAINBOW = 3, 1, 0, 10
    assert energy_satisfies([W, W], [W, C])
    assert not energy_satisfies([G], [W])
    assert energy_satisfies([RAINBOW, G], [W, C])
    assert not energy_satisfies([W], [W, C])
