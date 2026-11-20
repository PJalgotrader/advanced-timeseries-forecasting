"""Turns a notebook's source file into the notebook.

  python tools/build_notebook.py 07-deep-sequence-models/src/01_rnn_lstm_one_series.py

The source is one plain Python file: a line `# %%` starts a code cell, `# %% [markdown]` starts a text cell (its lines
begin with `# `). Plain files are easy to read, compare and edit, for people and for coding assistants; the notebook
is built from them and then run (tools/run_notebook.sh), so its saved outputs always belong to its code.
Writes <folder>/<name>.ipynb next to the `src` folder. Needs no packages.
"""
import json, os, sys

src = os.path.abspath(sys.argv[1])
out = os.path.join(os.path.dirname(os.path.dirname(src)), os.path.basename(src)[:-3] + ".ipynb")
cells, kind, buf = [], None, []
def flush():
    if kind is None: return
    while buf and not buf[-1].strip(): buf.pop()
    while buf and not buf[0].strip(): buf.pop(0)
    if not buf: return
    lines = [(l[2:] if l.startswith("# ") else l[1:] if l.startswith("#") else l) for l in buf] if kind == "markdown" else buf
    cell = {"cell_type": kind, "metadata": {}, "source": [l + "\n" for l in lines[:-1]] + [lines[-1]]}
    if kind == "code": cell.update(execution_count=None, outputs=[])
    cells.append(cell)
for line in open(src, encoding="utf-8").read().splitlines():
    if line.startswith("# %%"):
        flush(); kind, buf = ("markdown" if "[markdown]" in line else "code"), []
    else: buf.append(line)
flush()
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open(out, "w", encoding="utf-8", newline="\n"), indent=1, ensure_ascii=False)
print(len(cells), "cells ->", out)
