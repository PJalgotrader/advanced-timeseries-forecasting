# %% [markdown]
# # RNN and LSTM for one time series, in Keras
#
# The code for the video **"RNN and LSTM in Python: forecasting one time series"** (Deep Forecasting, Module 7, part 4.1).
# The theory is in the three videos before it: dense network vs RNN, the RNN, the LSTM.
#
# Every forecasting project in this repository follows the same six steps:
#
# 1. the data, split by time
# 2. prepare it, using the training part only
# 3. cut it into windows
# 4. a baseline to beat
# 5. build and train a model
# 6. score it in real units, more than once
#
# Setup is not covered here: it changes every year. Watch the setup video for the current year, and see the
# "Tested with" table in the README.

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
from keras import layers

print("keras", keras.__version__, "on", keras.backend.backend())

# %% [markdown]
# ## 1. The data, split by time
#
# Airline passengers per month, in thousands, 1949 to 1960: 144 months. The last 12 months are the **test** year. The
# 24 months before it are for **validation** (deciding when to stop training). Everything earlier is for **training**.
# Never shuffle a time series before splitting it: the model would be trained on the future.

# %%
try:
    df = pd.read_csv("../data/airline_passengers.csv")
except FileNotFoundError:   # running somewhere without the repository, for example in a hosted notebook
    df = pd.read_csv("https://raw.githubusercontent.com/PJalgotrader/advanced-timeseries-forecasting/main/data/airline_passengers.csv")

passengers = df["Passengers"].to_numpy("float32")
months = pd.to_datetime(df["Month"])
N_TEST, N_VAL = 12, 24

plt.figure(figsize=(10, 3.5))
plt.plot(months, passengers, color="#1f3a68")
plt.axvspan(months.iloc[-N_TEST - N_VAL], months.iloc[-N_TEST], color="#f2c14e", alpha=0.25, label="validation")
plt.axvspan(months.iloc[-N_TEST], months.iloc[-1], color="#2e9e6b", alpha=0.25, label="test")
plt.title("Airline passengers per month (thousands)"); plt.legend(); plt.show()

# %% [markdown]
# ## 2. Prepare it
#
# A recurrent layer squeezes its state through tanh, which is flat for large numbers. Fed raw values like 112 or 400,
# every unit sits at 1 and the forecast is one flat line (measured in the "dense network vs RNN" video). So:
#
# - take the **logarithm**: the seasonal swings stop growing with the level
# - take the **difference**: the trend is gone, and what is left is small, around zero
#
# `change[i]` is the change of the log from month `i` to month `i + 1`. That is the series the networks see.

# %%
change = np.diff(np.log(passengers))            # 143 values, roughly between -0.2 and +0.2
print(len(change), "changes, from", change.min().round(2), "to", change.max().round(2))

plt.figure(figsize=(10, 2.5)); plt.plot(months[1:], change, color="#1f3a68")
plt.title("Change of the log, month to month"); plt.show()

# %% [markdown]
# ## 3. Cut it into windows
#
# One sample is a **window**: 12 months in, the next month out. Ten lines of NumPy, nothing hidden.
# A recurrent layer wants the shape **(samples, time steps, features)**: here (samples, 12, 1).
#
# The test windows reach back into the validation months for their inputs. That is fine: at forecast time those
# months are in the past. What matters is that no test **target** was used for training.

# %%
LENGTH = 12

def make_windows(series, length, start, end):
    """Every window of `length` steps whose target, the next step, lies in series[start:end]."""
    X = np.stack([series[t - length:t] for t in range(start, end)])
    y = series[start:end]
    return X[..., None], y                      # the None adds the last axis: 1 feature

n = len(change)
test_start, val_start = n - N_TEST, n - N_TEST - N_VAL
X_train, y_train = make_windows(change, LENGTH, LENGTH, val_start)
X_val, y_val = make_windows(change, LENGTH, val_start, test_start)
X_test, y_test = make_windows(change, LENGTH, test_start, n)

for name, X in [("train", X_train), ("validation", X_val), ("test", X_test)]:
    print(f"{name:10s} {X.shape}")
print("\nfirst window:", X_train[0, :, 0].round(3), "-> target", y_train[0].round(3))

# %% [markdown]
# ## 4. A baseline to beat
#
# Scores are in **passengers**, not in changes of logs: undo the difference and the logarithm first. A forecast of
# next month is this month times e to the forecast change.
#
# Two baselines that need no training: "next month = this month", and "next month = the same month last year".

# %%
actual = passengers[test_start + 1:]            # the 12 months of the test year

def to_passengers(change_forecast):
    """Undo the difference and the log: next month = this month x exp(forecast change)."""
    return passengers[test_start:-1] * np.exp(change_forecast)

def score(forecast):
    """Mean absolute percentage error over the test year."""
    return float(np.mean(np.abs(forecast - actual) / actual) * 100)

results = {"this month again": score(passengers[test_start:-1]),
           "same month last year": score(passengers[test_start + 1 - 12:-12])}
for name, value in results.items():
    print(f"{name:22s} {value:5.1f}%")

# %% [markdown]
# ## 5. Build and train a model
#
# ### The simple RNN
#
# `SimpleRNN(16)` is the recurrent cell of the theory videos with 16 slots of memory. Its parameters:
# 1 x 16 weights on the input, 16 x 16 on the previous state, 16 biases: **288**. The `Dense(1)` on top turns the last
# state into one number: 16 weights and 1 bias.

# %%
def build(*hidden_layers):
    """A model that reads a window of 12 months with 1 feature and returns one number."""
    inputs = keras.Input(shape=(LENGTH, 1))
    x = inputs
    for layer in hidden_layers:
        x = layer(x)
    return keras.Model(inputs, layers.Dense(1)(x))

model = build(layers.SimpleRNN(16))
model.summary()

# %% [markdown]
# Training stops when the validation loss has not improved for 40 epochs, and the weights of the best epoch are kept.
# That one callback replaces guessing the number of epochs.

# %%
def train(model, verbose=0):
    model.compile(optimizer=keras.optimizers.Adam(0.003), loss="mse")
    stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=40, restore_best_weights=True)
    return model.fit(X_train, y_train, validation_data=(X_val, y_val),
                     epochs=400, batch_size=16, callbacks=[stop], verbose=verbose)

keras.utils.set_random_seed(0)
model = build(layers.SimpleRNN(16))
history = train(model, verbose=2)

# %%
def plot_history(history, title):
    best = int(np.argmin(history.history["val_loss"]))
    plt.figure(figsize=(8, 3))
    plt.plot(history.history["loss"], label="training", color="#1f3a68")
    plt.plot(history.history["val_loss"], label="validation", color="#e07b39")
    plt.axvline(best, color="grey", ls=":", label=f"best epoch: {best + 1}")
    plt.yscale("log"); plt.xlabel("epoch"); plt.title(title); plt.legend(); plt.show()

plot_history(history, "Simple RNN: loss")

# %%
def forecast(model):
    return to_passengers(model.predict(X_test, verbose=0)[:, 0])

rnn_forecast = forecast(model)
print(f"simple RNN, this run: {score(rnn_forecast):.1f}%")

plt.figure(figsize=(9, 3.5))
plt.plot(months.iloc[-N_TEST:], actual, "o-", color="#2e9e6b", label="actual")
plt.plot(months.iloc[-N_TEST:], rnn_forecast, "o--", color="#1f3a68", label="simple RNN, one month ahead")
plt.title("The test year"); plt.legend(); plt.show()

# %% [markdown]
# ### The LSTM
#
# One word changes. The LSTM has three gates and a candidate, each with its own copy of the RNN's weights:
# 4 x 288 = **1,152** parameters.

# %%
build(layers.LSTM(16)).summary()

# %% [markdown]
# ### Stacking layers
#
# The first recurrent layer has to hand over its state at **every** step, not only the last one, so that the second
# layer has a sequence to read: `return_sequences=True`. Its output shape becomes (12, 32) instead of (32).

# %%
build(layers.SimpleRNN(32, return_sequences=True),
      layers.SimpleRNN(32),
      layers.Dropout(0.1)).summary()

# %% [markdown]
# ### Reading the window in both directions
#
# `Bidirectional` runs two copies of the layer, one from the first month to the last and one backwards, and joins
# their states: 2 x 16 = 32 numbers. It helps with text, where a later word explains an earlier one. In a
# forecasting window the latest months matter most, and the backward copy reads them first and has 11 more steps to
# forget them.

# %%
build(layers.Bidirectional(layers.SimpleRNN(16))).summary()

# %% [markdown]
# ## 6. Score it in real units, more than once
#
# One training run is one draw: the starting weights are random. So each model is trained **five times** and the
# table shows the middle run and the range. Judge a model by that, never by its best run.

# %%
candidates = {
    "dense network (32, 16)": lambda: build(layers.Flatten(), layers.Dense(32, activation="relu"), layers.Dense(16, activation="relu")),
    "simple RNN (16)":        lambda: build(layers.SimpleRNN(16)),
    "LSTM (16)":              lambda: build(layers.LSTM(16)),
    "two RNN layers (32, 32)": lambda: build(layers.SimpleRNN(32, return_sequences=True), layers.SimpleRNN(32), layers.Dropout(0.1)),
    "bidirectional RNN (16)": lambda: build(layers.Bidirectional(layers.SimpleRNN(16))),
}
RUNS = 5
table = []
for name, make in candidates.items():
    errors = []
    for seed in range(RUNS):
        keras.utils.set_random_seed(seed)
        model = make()
        train(model)
        errors.append(score(forecast(model)))
    table.append({"model": name, "parameters": model.count_params(), "median %": np.median(errors), "best %": min(errors), "worst %": max(errors)})
    print(f"{name:26s} {np.median(errors):4.1f}%   ({min(errors):.1f} to {max(errors):.1f})")

table = pd.DataFrame(table).round(1)
table

# %%
plt.figure(figsize=(9, 3.8))
names = list(results) + list(table["model"])
values = list(results.values()) + list(table["median %"])
bars = plt.barh(names[::-1], values[::-1], color=(["#1f3a68"] * len(table) + ["#9aa7b8"] * len(results)))
for name, lo, hi in zip(table["model"], table["best %"], table["worst %"]):
    plt.plot([lo, hi], [name, name], color="black", lw=2)
plt.xlabel("error on the test year, percent (lower is better)"); plt.title("One month ahead, 1960"); plt.tight_layout(); plt.show()

# %% [markdown]
# ## What this shows
#
# - Every network beats the two baselines by a wide margin, and they land within about half a percentage point of
#   each other. On 144 well-behaved months, more layers and gates buy nothing.
# - The classical "airline model" (a seasonal ARIMA with two parameters) scores about 3.0% on the same year, and ETS
#   about 4.1%. For a small, regular series, start there.
# - What carries over to bigger problems is the recipe: split by time, prepare with the training part, windows,
#   a baseline, early stopping, scores in real units, several runs. The next notebook uses it on weather data with
#   14 features and 420,000 rows.
#
# ## Try it yourself, with a coding assistant
#
# Good things to ask an AI coding assistant, with this notebook open:
#
# - "Forecast 12 months ahead instead of 1." (Change the target in `make_windows`, or feed forecasts back in.)
# - "Swap the airline data for my own monthly series."
# - "Port this to PyTorch." or "Update this for the current version of Keras."
#
# And always check what you get back: the shapes of the windows, that the split is by time, that the baseline is
# still there, and that the score is measured on months the model never trained on.
