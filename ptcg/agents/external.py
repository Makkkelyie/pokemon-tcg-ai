"""Proxy for a third-party agent folder running in its own Python process.

Used only for local benchmarking against public bots (never shipped). The
process is kept alive across matches within one runner worker; Kaggle agents
reset their per-episode state when asked for the deck.
"""
import atexit
import json
import os
import subprocess
import sys

_PROCS = {}


def _proc(agent_dir):
    p = _PROCS.get(agent_dir)
    if p is None or p.poll() is not None:
        worker = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "external_worker.py")
        p = subprocess.Popen([sys.executable, "-I", worker, agent_dir], stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
        msg = json.loads(p.stdout.readline() or '{"error": "worker died while loading"}')
        if not msg.get("ready"):
            raise RuntimeError(f"external agent {agent_dir} failed to load: {msg}")
        _PROCS[agent_dir] = p
    return p


@atexit.register
def _cleanup():
    for p in _PROCS.values():
        try:
            p.kill()
        except Exception:
            pass


class ExternalAgent:
    name = "ext"

    def __init__(self, deck=None, seed=None, path=None):
        if not path or not os.path.exists(os.path.join(path, "main.py")):
            raise ValueError(f"external agent folder without main.py: {path}")
        self.path = os.path.abspath(path)
        self.proc = _proc(self.path)

    def __call__(self, obs):
        self.proc.stdin.write(json.dumps(obs) + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        if not line:
            _PROCS.pop(self.path, None)
            raise RuntimeError("external agent process died")
        msg = json.loads(line)
        if "error" in msg:
            raise RuntimeError(f"external agent error: {msg['error']}")
        return msg["action"]
