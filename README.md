<a href="https://pedramjahangiry.com"><img src="assets/header/banner.png" alt="Dr. Pedram Jahangiry: ML, deep learning and time series, explained visually" width="100%"></a>

# Advanced Time Series Forecasting

The code behind my video series on forecasting with machine learning and deep learning:
**ML, deep learning & time series, explained visually.**

Each notebook goes with one video. The videos explain the ideas with pictures; the notebooks are the same thing in
code, short enough to read from top to bottom. More models are added as the series grows.

- Videos: https://www.youtube.com/@pedramjahangiry
- Interactive tools and 3D books: https://pedramjahangiry.com

## What is here

| Folder | Notebook | Video |
|---|---|---|
| `07-deep-sequence-models/` | `01_rnn_lstm_one_series.ipynb`: RNN and LSTM on one monthly series (airline passengers), at three horizons: one step ahead, multi-step, multi-output | [RNN and LSTM in Python: Forecasting a Time Series with Keras, Step by Step](https://youtu.be/X6PwdT5-V_Q) (Deep Forecasting, Module 7, part 4 of 5) |
| `07-deep-sequence-models/` | `02_rnn_lstm_weather.ipynb`: dense, convolution, RNN, LSTM, GRU and bidirectional models on weather data with 14 features | [RNN vs LSTM vs GRU in Python: 7 Models on Real Weather Data in Keras](https://youtu.be/0fZhslU5JZQ) (Module 7, part 5 of 5) |

The theory for both is in three videos: dense network vs RNN, the RNN, the LSTM.

## Every notebook follows the same six steps

1. **The data**, split by time: train, validation, test.
2. **Prepare** it, using the training part only.
3. **Windows**: cut the series into samples with a few lines of NumPy.
4. **A baseline** that needs no training.
5. **Build and train** a model, with early stopping.
6. **Score** it in real units, over several runs, and at three horizons where the data allows: **one step ahead**,
   **multi-step** (a model fed its own forecasts) and **multi-output** (the whole horizon at once). Windows are cut with
   **Tx** values in and **Ty** values out.

The models change from notebook to notebook. The steps do not.

## Running the notebooks

Setup changes every year, so it is not written into the notebooks. Watch the setup video for the current year on the
channel. In short, a notebook needs Python with `keras` and one of its backends (TensorFlow, PyTorch or JAX),
`numpy`, `pandas` and `matplotlib`. The second notebook is much faster with a GPU.

Every notebook prints its Keras version and backend in the first cell. Compare it with this table:

| Last run | Python | Keras | Backend | Notes |
|---|---|---|---|---|
| 2026-11-20 | 3.12 | 3.15.1 | TensorFlow 2.21 | notebooks 01 and 02; Linux (WSL), NVIDIA RTX 4070. Notebook 01 takes about 20 minutes on a CPU; notebook 02 about 4 hours on the GPU (10 runs per fast model, and its two slow models) |

An earlier version of notebook 01 (before the three horizons) also ran unchanged on Keras 2.14 with TensorFlow 2.14
(Python 3.10).

## Working with a coding assistant

These notebooks are meant to be changed. Open one next to an AI coding assistant and ask for what you need:
"forecast 12 months ahead", "use my own data", "rewrite this for PyTorch", "update this for the current version of
Keras". Then check what comes back, every time:

- **the shapes**: samples, time steps, features
- **the split**: by time, never shuffled
- **the baseline**: still there, and beaten
- **the test data**: touched once, at the very end

## For maintainers

The notebooks are built from plain Python files in each folder's `src/`:

```
python tools/build_notebook.py 07-deep-sequence-models/src/01_rnn_lstm_one_series.py
bash tools/run_notebook.sh 07-deep-sequence-models/01_rnn_lstm_one_series.ipynb <python> <kernel>
```

Edit the `src` file, build, run, and commit both, so that the saved outputs always belong to the code.

Every notebook opens with the same header cell (banner, Colab button, the video, links):
`python tools/notebook_header.py <notebook.ipynb> <video link>` adds or replaces it in the notebook and its `src` file
without touching the saved outputs. The banner is `assets/header/banner.html`, rendered to `banner.png` with
`node tools/render_banner.js`.

## Data

- `data/airline_passengers.csv`: monthly international airline passengers, 1949 to 1960 (Box and Jenkins).
- Jena climate, 2009 to 2016 (Max Planck Institute for Biogeochemistry): downloaded by the weather notebook.

## Licence

The code is under the MIT licence ([LICENSE](LICENSE)): use it, change it, share it, with credit. The data sets keep
their own terms (see Data).
