"""Subprocess host for a third-party Kaggle agent folder (main.py [+ deck.csv, cg/]).

Loads main.py the way kaggle_environments does (exec without __file__, the
agent dir on sys.path only while loading, last callable wins) and answers one
JSON observation per stdin line with one JSON line on the protocol fd. The
agent's own prints go to stderr so they cannot corrupt the protocol.
"""
import json
import os
import sys


class Struct(dict):
    def __init__(self, **entries):
        dict.__init__(self, entries)
        self.__dict__.update(entries)

    def __setattr__(self, attr, value):
        self.__dict__[attr] = value
        self[attr] = value


def structify(o):
    if isinstance(o, list):
        return [structify(x) for x in o]
    if isinstance(o, dict):
        return Struct(**{k: structify(v) for k, v in o.items()})
    return o


def main():
    agent_dir = os.path.abspath(sys.argv[1])
    proto = os.fdopen(os.dup(1), "w")
    os.dup2(2, 1)
    os.chdir(agent_dir)
    with open(os.path.join(agent_dir, "main.py")) as f:
        src = f.read()
    sys.path.append(agent_dir)
    env = {}
    exec(compile(src, os.path.join(agent_dir, "main.py"), "exec"), env)
    sys.path.pop()
    fn = [v for v in env.values() if callable(v)][-1]
    nargs = getattr(getattr(fn, "__code__", None), "co_argcount", 1)
    proto.write(json.dumps({"ready": True}) + "\n")
    proto.flush()
    for line in sys.stdin:
        obs = json.loads(line)
        try:
            args = [structify(obs), Struct()][: max(1, nargs)]
            act = fn(*args)
            out = {"action": [int(a) for a in act]}
        except Exception as e:  # reported to the runner as an agent error
            out = {"error": repr(e)}
        proto.write(json.dumps(out) + "\n")
        proto.flush()


if __name__ == "__main__":
    main()
