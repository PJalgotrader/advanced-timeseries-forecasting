# Advanced Time Series Forecasting — tracker

Rules and how notebooks are built: [CLAUDE.md](CLAUDE.md). Update at the end of every session.

## Notebooks

| Notebook | Video | Status |
|---|---|---|
| `07-deep-sequence-models/01_rnn_lstm_one_series.ipynb` | Module 7 part 4.1 (first cut made 2026-11-20, not published) | Built and run 2026-11-20 (Keras 3.15.1, TensorFlow 2.21) |
| `07-deep-sequence-models/02_rnn_lstm_weather.ipynb` | Module 7 part 4.2 (first cut made 2026-11-20, not published) | Written and built 2026-11-20. Its full run (about 4 hours on a shared GPU) was still going when the video was delivered: the file has no saved outputs until that run is committed. |

## Next up

- [ ] **Pedram: make the repository public** before the first video that links to it is published.
- [ ] **Pedram: choose a licence** for the code (none is set).
- [ ] Commit the weather notebook with its saved outputs once its run has finished, and add its row to the
      README's "Tested with" table.
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
