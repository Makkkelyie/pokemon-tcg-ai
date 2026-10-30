# Pokémon TCG AI Battle Challenge (Kaggle Playground) — agent, harness and meta analysis

An entry for Kaggle's [The Pokémon Company — PTCG AI Battle Challenge (Playground)](https://www.kaggle.com/competitions/the-pokemon-company-ptcg-ai-battle-challenge-playground),
a simulation competition: agents play best-of-three matches of the Pokémon Trading Card Game
against each other on a ladder (Gaussian skill rating, final Bradley-Terry tournament).

**Links:** [GitHub repository](https://github.com/Makkkelyie/pokemon-tcg-ai) · [Kaggle notebook with the same code](https://www.kaggle.com/code/kellydannimao/ptcg-local-harness-meta-analysis-and-search) (published on Kaggle, as competition rule 3.6b requires for publicly shared code)

The project follows one rule throughout: **measure every change locally, with confidence
intervals, before trusting it.** That rule is what turned up the two main findings below.
First, deck choice moved local win rate far more than any search I added. Second, determinized
search on top of a strong policy network did not measurably beat the policy at the compute
budgets I tried.

> Card names, deck lists and engine files are deliberately absent from this repository. The
> competition rules (2.4, 3.6b) restrict the competition data to competition use and forbid
> publicly sharing "Pokémon Elements" (card data, deck recipes, simulator program). Decks are
> referred to by an 8-character hash of their sorted card-id list.

## What is in here

| Path | What |
|---|---|
| `ptcg/match.py` | Local best-of-N runner that mirrors the official `cabt` interpreter (seat rules, reverse start in game 2, error/timeout = match loss, per-game 600 s pool), multiprocessing, Wilson 95% intervals |
| `ptcg/agents/base.py` | Random and "first option" baselines |
| `ptcg/agents/heuristic.py` | Rule-based agent: scores every legal option (KO checks, energy needs, evolution, bench, retreat), then picks between `minCount` and `maxCount` of them |
| `ptcg/belief.py`, `ptcg/agents/search.py`, `ptcg/evaluate.py` | My first search agent: determinized flat Monte-Carlo with UCB1 over candidates, heuristic rollouts and a hand-written evaluation function, using the engine's forward model (`search_begin` / `search_step`) |
| `ptcg/agents/policy_search.py` | Search driver around a pretrained policy/value network (see Credits): gating on policy uncertainty, top-k candidates, PUCT or uniform allocation with common random numbers, rollout variants, an override margin rule, a time manager, and per-game tracker resets |
| `ptcg/meta.py` | Opponent-deck prior for determinization: ladder decklists weighted by usage, filtered by how many of the opponent's revealed cards each list contains |
| `scripts/meta_analysis.py` | Streams a day of Kaggle's ladder replays (1,665 matches, 21 GB unzipped) straight out of the zip and fits Bradley-Terry models |
| `ptcg/external_worker.py`, `ptcg/agents/external.py` | Runs any public Kaggle agent folder in its own process, loaded exactly the way `kaggle_environments` loads it, so it can be used as a local opponent |
| `scripts/build_submission.py`, `submission/main.py` | Builds `submission.tar.gz` (main.py + deck.csv at top level) |
| `tests/` | 15 pytest tests: action legality for every agent, determinizations accepted by the engine, the tarball loading under Kaggle's `exec` loader, and a parity run in the official `kaggle_environments` `cabt` env |

## Setup

```bash
/opt/homebrew/bin/python3.13 -m venv .venv          # the env needs Python >= 3.11
.venv/bin/pip install numpy pytest torch requests jsonschema
.venv/bin/pip install --no-deps kaggle-environments==1.33.0   # only for the parity test
kaggle competitions download -c the-pokemon-company-ptcg-ai-battle-challenge-playground -p data/raw
(cd data/raw && unzip -q *.zip)
# optional: public baseline network, used by the `ps` agent
kaggle datasets download cristobaltudela/ptcg-playground-public-baseline -p data/external/bots/dipam_port --unzip
python scripts/setup_dipam.py
.venv/bin/python -m pytest
```

```bash
python scripts/run_match.py heuristic random -n 200                        # N BO3 matches, win rate + 95% CI
python scripts/run_match.py ps ext --a-kw mode=none --b-kw path=<agent dir> --deck-a decks/<deck>.csv
python scripts/meta_analysis.py data/external/episodes/*.zip                 # ladder meta tables
python scripts/build_submission.py --deck decks/<deck>.csv --config '{"agent": "ps", "mode": "none"}' --name sub
```

## What the environment looks like

* An agent is a function `agent(obs) -> list[int]` of option indices. It must return its
  60-card deck when `obs["select"] is None`. The engine only ever offers legal options, so a
  "move" is choosing among them, sometimes several at once.
* The SDK exposes a **forward model**: from any observation you can start a simulation with
  your own guess of every hidden card, then step it. That makes determinized search possible.
* On the ladder each player has a 600 s thinking pool. **In the replays the pool resets at the
  start of every game** (2,406 of 2,406 games 2/3 started full). `kaggle-environments` 1.33.0
  on PyPI keeps one pool per match, and the runner supports both.
* `main.py` is `exec()`ed without `__file__`, and the *last callable* defined is taken as the
  agent. An illegal action or an exception loses the whole match. The tests check both.

## Results

All numbers come from `results/*.log`, regenerated by the commands above. Each line is a set of
independent best-of-three matches with seats alternated. **Match score** counts a won match as
1, a draw as ½ and a loss as 0; the brackets are Wilson 95% intervals. The engine's shuffles
cannot be seeded, so runs cannot be paired.

### 1. Baselines (both sides use the competition's sample deck)

| Agent A | Agent B | Matches | Match score [95% CI] | Game win rate |
|---|---|---:|---|---|
| heuristic | random | 200 | 0.985 [0.957, 0.995] | 0.925 |
| MC search (0.1 s/decision) | heuristic | 120 | 0.583 [0.494, 0.668] | 0.555 |

### 2. Against the strongest public bot

The best public agent on the ladder is a transformer policy trained with RL (Credits). With the
seat bug fix, it is referred to below as **public policy**. Same deck on both sides (the bot's own
"baseline deck"):

| Agent A | Agent B | Matches | Match score [95% CI] | Game win rate |
|---|---|---:|---|---|
| heuristic (sample deck) | public policy | 12 | 0.000 [0.000, 0.243] | 0.040 |
| MC search (0.1 s/decision) | public policy | 40 | 0.025 [0.004, 0.129] | 0.160 |

My hand-written heuristic and evaluation function are far behind a learned policy. Because of
this I switched to search *on top of* that policy.

### 3. Does search help a strong policy? (A = `ps` agent with the same network; baseline deck, 100 matches each)

| Search configuration | Match score [95% CI] | Game win rate | ms / decision |
|---|---|---|---:|
| none: policy argmax (sanity check, should be ≈ 0.5) | 0.550 [0.452, 0.644] | 0.533 | 14 |
| PUCT, 4 candidates × 4 worlds, greedy policy rollouts to end of turn, 0.3 s | 0.490 [0.394, 0.587] | 0.492 | 508 |
| uniform + margin 0.05 / ≥3 visits, same rollouts, 0.3 s | 0.480 [0.385, 0.577] | 0.490 | 497 |
| uniform, no rollout (value head after the action), 8 worlds, 0.15 s | 0.570 [0.472, 0.663] | 0.532 | 136 |

None of the four configurations is distinguishable from the policy it searches over. Rollouts
through the network cost ~5 ms per forward pass, so a 0.3 s budget allows only ~15 short
rollouts spread over 4 candidates and 4 hidden-card worlds. Their noise swamps the small
differences between the moves a well-trained policy is torn between. The PUCT version also
almost never overrides the prior (2 overrides in 159 searched decisions in one traced match). The
honest conclusion at this budget is that the search is not worth its cost, so the submission
plays the policy.

### 4. Deck choice (A = public policy network piloting each deck; B = public policy with the baseline deck; 100 matches each)

| Deck (hash) | Ladder share | Pilot-adjusted deck effect (Elo ± se) | Match score [95% CI] |
|---|---:|---:|---|
| Deck A (`fcc26c27` family) | 22.4% | −58 ± 41 | **0.880** [0.802, 0.930] |
| Deck B (`38a4f4b5`) | 19.7% | +18 ± 49 | **0.790** [0.700, 0.858] |
| baseline deck (the bot's own) | 7.7% | −134 ± 54 | 0.550 [0.452, 0.644] |
| Deck C (`0cfe5f2b`) | 11.3% | +61 ± 52 | 0.470 [0.375, 0.567] |
| Deck C' (`ba2c3f50`) | (same archetype) | | 0.440 [0.347, 0.538] |
| Deck D (`b246db99`) | 3.1% | +132 ± 57 | 0.090 [0.048, 0.162] |

A single opponent can hide rock-paper-scissors effects, so the two best decks also played a
five-opponent gauntlet: the same network piloting three ladder decks, plus two public
rule-based bots. 60 matches per cell:

| A \ opponent | Deck A | Deck B | Deck C' | rules bot 1 | rules bot 2 |
|---|---|---|---|---|---|
| Deck A | 0.53 [0.41, 0.65] | 0.37 [0.26, 0.49] | 0.60 [0.47, 0.71] | 1.00 [0.94, 1.00] | 1.00 [0.94, 1.00] |
| Deck B | 0.63 [0.51, 0.74] | 0.50 [0.38, 0.62] | 0.22 [0.13, 0.34] | 0.98 [0.91, 1.00] | 0.93 [0.84, 0.97] |

The matchups form a cycle: A beats C and the baseline deck, B beats A, C beats B. Weighted by
ladder share over the testable archetypes (A 22%, B 20%, C 11%, baseline 8%), both decks come
out at ≈ 0.535. The local test cannot separate them. The replay model prefers B by about 76 Elo,
with wide intervals. Kaggle keeps two active submissions per team, so I built one per deck.

### 5. Ladder meta (one day of replays, 2026-10-19, 1,665 BO3 matches between top-rated agents)

* **Agent strength:** Bradley-Terry (MM algorithm) over (team, deck) agents.
* **Deck strength separated from pilot skill:** a ridge-penalized logistic Bradley-Terry
  model, `logit P(i beats j) = (team_i + deck_i) − (team_j + deck_j)`, fitted by Newton's
  method. Standard errors come from the inverse penalized Hessian. Teams that play several
  decks identify the deck effects. Without this split, a deck played by one strong team simply
  inherits that team's rating.
* **First player** wins 56.9% of games [55.3%, 58.4%], and agents choose to go first in 98.7% of
  decisions.
* **Thinking time:** median pool use is 17 s per match (p90 69 s). Almost nobody searches deeply.

### 6. The meta moved (ladder replays 2026-10-28/29, 3,317 BO3 matches; re-run of sections 4–5)

Ten days after submission both decks had become the two weakest archetypes on the ladder
(pilot-adjusted deck effect −134 and −157 Elo), and the score drifted down with them. The
decks now on top mostly contain cards outside the network's vocabulary (6–21 cards map to the
UNK embedding), so the network cannot pilot them. I re-ran the deck test with
`scripts/deck_gauntlet.py`: each fully known candidate, piloted by the network with the KO guard,
against the four testable meta decks (Deck E 34% share, F 10%, G 8%, H 5%), 24 matches per pairing.

| Candidate | vs E | vs F | vs G | vs H | Share-weighted |
|---|---|---|---|---|---|
| Deck I (`5b3c6b0f`) | 0.58 | 0.71 | 0.96 | 0.96 | **0.69** |
| Deck E (`1652a67e`) | 0.50 | 0.67 | 0.67 | 0.79 | 0.58 |
| Deck F (`dc058243`) | 0.63 | 0.58 | 0.38 | 0.54 | 0.58 |
| Deck A (submitted) | 0.17 | 0.46 | 0.00 | 0.04 | 0.19 |
| Deck B (submitted) | 0.25 | 0.29 | 0.69 | 0.08 | 0.30 |

Every pairing has an interval of about ±0.2, but the gap between the new and old decks is far
larger than that. The two untestable top decks are left out, because a field piloted by a
network that doesn't know their cards loses to everything (1.00 for every candidate). The new
submissions are Deck I and Deck F. Deck F ties Deck E locally, and I chose it because its
pilot-adjusted ladder effect is higher (+75 vs +13 Elo).

### Leaderboard

| Submission | Local evidence | Leaderboard score |
|---|---|---|
| public-policy pilot, Deck A | sections 4 and 5 | 970.7 (provisional) |
| public-policy pilot, Deck B | sections 4 and 5 | 911.2 (provisional) |
| + KO guard, Deck A / Deck B | KO-guard replay study | 1022.0 / 1058.5 (2026-10-30) |
| + KO guard, Deck I / Deck F | section 6 | submitted 2026-10-30 |

Scores were read on 2026-10-20, 44 minutes after submission. Ratings start at μ0 = 600, and σ is still
large this early, so the gap between the two decks is not yet meaningful. For scale, the top of the
leaderboard was ≈ 1359 on the same day.

## What I learned

* **Build the evaluation harness first, and make it faithful.** Reading the official
  interpreter and the replays turned up rules that matter: seat 1 answers "go first?" in
  game 2, an exception loses the whole match, the loader has no `__file__`, the last callable
  wins, and the pool resets per game on the ladder (not in the PyPI package). Each one is a test.
* **Report intervals, not point estimates.** With 100 BO3 matches the 95% interval on match
  score is about ±0.1. Most "improvements" I tried were inside it.
* **A strong prior is hard to beat with a small search budget.** Determinized MC search clearly
  helped my weak heuristic (0.58 vs 0.50). On top of a learned policy, the same idea is noise
  unless rollouts are cheap enough to run hundreds of times per decision.
* **The meta matters more than micro-play here.** Changing the deck the same network pilots
  moved the local match score from 0.09 to 0.88 against one opponent. Separating deck effects
  from pilot skill needs a model, not raw win rates.
* **Negative results are results.** The search code is kept, tested and documented, and it is
  off in the submission because the measurements say so.

## Credits

* **Policy/value network, observation encoder and card tracker** (`dipam_ptcg`, vendored by
  `scripts/setup_dipam.py`, not committed): Dipam Chakraborty, [kaggle-ptcg-ai-battle](https://github.com/dipamc/kaggle-ptcg-ai-battle),
  MIT license. It was ported to the Playground card pool by Cristobal Tudela (Kaggle dataset
  `cristobaltudela/ptcg-playground-public-baseline`, MIT). The one-line seat fix was published by
  Akmal Xodarev (busyaprime). The license ships with the submission as `LICENSE-dipam`.
* **Local opponents:** public notebooks by Camilo Gonzalez and Briano (rule-based bots).
  Dariush Afshar's public "Bo3 Gauntlet" notebook documented the loader and seat rules. Darren
  Sheehan's "Daily Meta" notebook described the replay format.
* **Engine/SDK and card data:** The Pokémon Company / Kaggle (competition data, competition use only).
* Everything else (harness, heuristic and MC agents, search driver, meta model, tests): this repository.
