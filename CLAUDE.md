# Advanced Time Series Forecasting — notes for coding agents

The public-facing code for my forecasting videos (YouTube: @pedramjahangiry). Started 2026-11-20 with the two Python
videos of Deep Forecasting, Module 7. It stands alone, and it will cover many more models. Status and to-dos:
[TRACKER.md](TRACKER.md).

## Rules (mine, they bind)

- **No university name, course number or "Prof." anywhere**: not in notebooks, README, comments or commit
  messages. This repository is not my university course repository, which I change myself, separately.
- **The repository is public** (since 2026-11-24, on my word). Everything committed here is visible to anyone: no
  keys, no local paths, no private notes.
- **Notebooks must age well.** They go with videos that stay up for years:
  - no installation steps and no version-specific workarounds in a notebook; setup is "the setup video for this year"
  - `import keras` and `from keras import layers` only (no `tensorflow.keras`), so the code does not depend on the backend
  - windows are made with a few lines of NumPy, not with a library helper
  - data is read from `data/` (with a download fallback), never parsed line by line
  - every notebook follows the six steps: data, prepare, windows, baseline, model, score
  - scores in real units (passengers, degrees), with a baseline, over several runs where a run is cheap
  - the first cell prints the Keras version and backend; the README's "Tested with" table is updated after every run
- **A notebook says what its video says.** The numbers in a video come from the same code and seeds as the
  notebook's saved outputs. If a notebook changes after its video is out, keep the old results reachable (Git) and
  say in the README what changed.
- Plain language in the text cells. A newcomer who has watched the video should be able to read every line.

## How notebooks are made

The source of a notebook is `<folder>/src/<name>.py` (cells marked `# %%` and `# %% [markdown]`). Build it with
`python tools/build_notebook.py <src file>`, then run it with `tools/run_notebook.sh` and commit both files.
Never edit the `.ipynb` by hand.

On my machine the notebooks run in WSL (Ubuntu), conda environment `tf_gpu` (Keras 3, TensorFlow, the GPU):

```
wsl -e bash -lc 'cd /mnt/e/Github/advanced-timeseries-forecasting && bash tools/run_notebook.sh <notebook.ipynb> ~/miniconda3/envs/tf_gpu/bin/python tf_gpu'
```

Pass the kernel name (`tf_gpu`): the default `python3` kernel on that machine belongs to an older Python with Keras 2.

## Where the rest lives

- The films, their scripts and the experiments behind their numbers: `E:\Github\lecture-remakes`
  (projects `m7p41-python-univariate`, `m7p42-python-multivariate`).
- My layout rule for every project: code on E: under Git with a private GitHub repository; commit and push at the
  end of every session.
