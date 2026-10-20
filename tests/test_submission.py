"""The built tarball must load the way kaggle_environments loads agents and play legally."""
import os
import subprocess
import sys
import tarfile

import pytest

from .conftest import ROOT, requires_engine

# The env's builtin "random" agent ships a hard-coded deck that newer engine builds reject
# ("deck error"), so only our own agents play here.
PARITY_SCRIPT = r"""
import sys
from kaggle_environments import make
p, q = sys.argv[1], sys.argv[2]
for agents in ([p, q], [q, p], [p, p]):
    env = make("cabt")
    last = env.run(agents)[-1]
    statuses = [s.status for s in last]
    assert statuses == ["DONE", "DONE"], (statuses, env.steps[0][0].get("error"))
print("OK")
"""


@requires_engine
def test_build_and_load(tmp_path):
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    from build_submission import build
    from ptcg.engine import default_deck_path
    _, tar_path = build(default_deck_path(), {"budget": 0.01, "max_sims": 10}, out_dir=str(tmp_path))
    with tarfile.open(tar_path) as t:
        names = t.getnames()
    assert "main.py" in names and "deck.csv" in names
    assert any(n.startswith("cg/") for n in names) and any(n.startswith("ptcg/") for n in names)
    # Extract and load exactly like kaggle_environments.agent.get_last_callable.
    agent_dir = tmp_path / "agent"
    with tarfile.open(tar_path) as t:
        t.extractall(agent_dir, filter="data")
    code = f"""
import sys
src = open(r'{agent_dir / "main.py"}').read()
sys.path.append(r'{agent_dir}')
env = {{}}
exec(compile(src, 'main.py', 'exec'), env)
sys.path.pop()
agent = [v for v in env.values() if callable(v)][-1]
assert agent.__name__ == 'agent', agent
deck = agent({{'select': None}})
assert len(deck) == 60
print('LOADED')
"""
    out = subprocess.run([sys.executable, "-I", "-c", code], cwd=str(tmp_path), capture_output=True, text=True)
    assert "LOADED" in out.stdout, out.stderr


@requires_engine
def test_official_env_parity(tmp_path):
    pytest.importorskip("kaggle_environments")
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    from build_submission import build
    from ptcg.engine import default_deck_path
    sub, _ = build(default_deck_path(), {"budget": 0.01, "max_sims": 10}, out_dir=str(tmp_path))
    sub2, _ = build(default_deck_path(), {"budget": 0.0, "max_sims": 1}, out_dir=str(tmp_path), name="other")
    out = subprocess.run([sys.executable, "-c", PARITY_SCRIPT, os.path.join(sub, "main.py"),
                          os.path.join(sub2, "main.py")],
                         cwd=str(tmp_path), capture_output=True, text=True, timeout=600)
    assert "OK" in out.stdout, out.stderr[-2000:]
