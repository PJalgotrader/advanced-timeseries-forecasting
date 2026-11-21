# Advanced Time Series Forecasting — tracker

Rules and how notebooks are built: [CLAUDE.md](CLAUDE.md). Update at the end of every session.

## Notebooks

| Notebook | Video | Status |
|---|---|---|
| `07-deep-sequence-models/01_rnn_lstm_one_series.ipynb` | Module 7 part 4.1 (second cut, 2026-11-20, not published) | Run 2026-11-20 with Tx/Ty and the three horizons (Keras 3.15.1, TensorFlow 2.21, CPU); its numbers are the film's exactly |
| `07-deep-sequence-models/02_rnn_lstm_weather.ipynb` | Module 7 part 4.2 (third cut, 2026-11-20, not published) | Run 2026-11-20 with 10 runs per fast model (about 4 hours, GPU); within 0.02 degrees of the film |

## Next up

- [ ] **Pedram: make the repository public** before the first video that links to it is published.
- [ ] **Pedram: choose a licence** for the code (none is set).
- [ ] The setup video for the year (the videos say "watch the installation video for this year"); when it exists,
      link it in the README.
- [ ] A yearly check: run every notebook top to bottom in the current environment and update "Tested with".

## Log

- 2026-11-20 — Repository created (private) on Pedram's word: standalone, no university name anywhere, more models to come.
  Notebook 01 written from scratch for the Module 7 part 4.1 video: six steps, NumPy windows, `import keras`,
  scores in passengers over 5 runs.
- 2026-11-20 — Notebook 02 (seven models on the Jena weather data) written for the Module 7 part 4.2 video. The
  video's numbers come from the same code run model by model (three runs for the fast models): baseline 2.62,
  dense 2.83, convolution 3.25, simple RNN 2.48, LSTM 2.55, bidirectional LSTM 2.59, LSTM with recurrent dropout
  2.47, two GRU layers with dropout 2.42 (degrees, test data).
- 2026-11-20 — Notebook 02 rerun with 10 runs per fast model (the film's second cut showed 3 runs were too few for the
  dense network): dense 2.68, convolution 3.20 (one run diverged, 6.10), simple RNN 2.48, LSTM 2.57, bidirectional
  2.57, slow models 2.47 and 2.42. Layers are copied inside the run loop, so every seed really starts its own weights.
  A note says the notebook is one step ahead only, and why.
- 2026-11-20 — Notebook 01: Pedram's Tx and Ty in `make_windows(series, Tx, Ty, start, end)`, and section 7 with his
  three horizons (one step ahead, multi-step, multi-output; 10 runs each): dense 3.8 / 3.9 / 4.3, simple RNN 3.2 / 4.1 /
  4.7, LSTM 3.6 / 4.1 / 4.9 percent. A foundation video on the three horizons is promised in both Python videos.
