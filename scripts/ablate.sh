#!/bin/zsh
# Run a list of PolicySearchAgent configs against a fixed opponent; results append to results/matches.jsonl.
#   scripts/ablate.sh <n_matches> <opponent_dir> "<kw...>" "<kw...>" ...
N=$1; OPP=$2; shift 2
for KW in "$@"; do
  echo "=== ps $KW"
  .venv/bin/python scripts/run_match.py ps ext --deck-a decks/ogerpon_dipam.csv \
    --a-kw ${=KW} --b-kw path=$OPP -n $N -w 6 --log results/matches.jsonl 2>&1 | tail -5
done
