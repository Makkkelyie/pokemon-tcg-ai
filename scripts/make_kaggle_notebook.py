"""Package this repository's code as a Kaggle notebook attached to the competition.

Rule 3.6b: code shared publicly during the competition must also be shared on
Kaggle (competition forum or notebooks). The notebook writes every tracked
source file with %%writefile, points the code at the competition data input,
runs the test suite and a short demo match. It contains no competition data,
card names or deck lists; those stay in the Kaggle input.

  python scripts/make_kaggle_notebook.py      # -> build/kaggle_notebook/
  kaggle kernels push -p build/kaggle_notebook  # pushes as PRIVATE; make it public in the UI after review
"""
import json
import os
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SLUG = "the-pokemon-company-ptcg-ai-battle-challenge-playground"
KERNEL_ID = "kellydannimao/ptcg-local-harness-meta-analysis-and-search"
TITLE = "PTCG: Local Harness, Meta Analysis and Search"


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.splitlines(keepends=True)}


def main():
    files = subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).split()
    src = [f for f in files if f.endswith((".py", ".ini", ".txt", ".sh")) or f == "LICENSE"]
    readme = open(os.path.join(ROOT, "README.md"), encoding="utf-8").read()
    cells = [
        md(readme),
        md("## Code\n\nEvery source file of the repository, written to `/kaggle/working/ptcg-ai`. "
           "The engine and card data come from the competition input; nothing from it is printed."),
        code("import os\nos.makedirs('/kaggle/working/ptcg-ai', exist_ok=True)\n"
             f"for d in {sorted({os.path.dirname(f) for f in src if os.path.dirname(f)})!r}:\n"
             "    os.makedirs(os.path.join('/kaggle/working/ptcg-ai', d), exist_ok=True)\n"
             "%cd /kaggle/working/ptcg-ai"),
    ]
    for f in src:
        body = open(os.path.join(ROOT, f), encoding="utf-8").read()
        if not body.strip():  # %%writefile refuses an empty cell body
            body = "# (intentionally empty)\n"
        cells.append(code(f"%%writefile {f}\n{body}"))
    cells += [
        md("## Tests\n\nTests that need the vendored public network or `kaggle_environments` skip "
           "themselves when those are not attached."),
        code("import glob, os\n"
             "hits = sorted(glob.glob('/kaggle/input/**/sample_submission/cg/api.py', recursive=True), key=len)\n"
             "assert hits, 'attach the competition data as an input'\n"
             "CG_DIR = os.path.dirname(os.path.dirname(hits[0]))\n"
             "os.environ['PTCG_CG_DIR'] = CG_DIR\n"
             "print('engine dir:', CG_DIR)"),
        code("!python -m pytest -q -p no:cacheprovider"),
        md("## Demo: heuristic agent vs random, 20 best-of-three matches (sample deck)"),
        code("!python scripts/run_match.py heuristic random -n 20 -w 2 --log /tmp/matches.jsonl"),
        code("# Keep competition files out of this notebook's outputs.\n"
             "!rm -rf /kaggle/working/ptcg-ai/build /kaggle/working/ptcg-ai/results"),
    ]
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                                        "name": "python3"},
                                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    out = os.path.join(ROOT, "build", "kaggle_notebook")
    os.makedirs(out, exist_ok=True)
    nb_name = "ptcg-local-harness.ipynb"
    with open(os.path.join(out, nb_name), "w", encoding="utf-8") as fh:
        json.dump(nb, fh, indent=1, ensure_ascii=False)
    meta = {"id": KERNEL_ID, "title": TITLE, "code_file": nb_name, "language": "python",
            "kernel_type": "notebook", "is_private": True, "enable_gpu": False, "enable_internet": False,
            "dataset_sources": [], "competition_sources": [SLUG], "kernel_sources": []}
    with open(os.path.join(out, "kernel-metadata.json"), "w") as fh:
        json.dump(meta, fh, indent=1)
    print(f"wrote {out}/{nb_name} ({len(src)} files, {len(cells)} cells)")


if __name__ == "__main__":
    main()
