# %% [markdown]
# # RNN, LSTM and GRU on weather data, in Keras
#
# The code for the video **"RNN, LSTM and GRU in Python: forecasting temperature from 14 weather measurements"**
# (Deep Forecasting, Module 7, part 4.2). The theory is in the three videos before it: dense network vs RNN, the RNN,
# the LSTM. The first notebook of this folder shows the same six steps on a small series; read that one first.
#
# The task follows François Chollet, *Deep Learning with Python*: a weather station records 14 measurements every
# 10 minutes. Given the last 5 days, forecast the temperature 24 hours from now.
#
# 1. the data, split by time
# 2. prepare it, using the training part only
# 3. cut it into windows
# 4. a baseline to beat
# 5. build and train the models
# 6. score them in real units
#
# Setup is not covered here: watch the setup video for the current year. A GPU is strongly recommended. Two of the
# seven models are slow by design (the reason is part of the lesson); set `RUN_SLOW = False` to skip them. The five
# fast models are trained 10 times each, which takes about an hour and a half; set `RUNS = 1` for a quick look.

# %%
import os, time, urllib.request, zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
from keras import layers

print("keras", keras.__version__, "on", keras.backend.backend())
RUN_SLOW = True      # the two models with recurrent dropout: about 50 and 105 minutes on a recent GPU
EPOCHS = 10
RUNS = 10            # how many times each fast model is trained (one run is one draw: see step 5)

# %% [markdown]
# ## 1. The data, split by time
#
# The Jena climate data: 14 measurements (pressure, temperature, humidity, wind ...) every 10 minutes from 2009 to
# 2016, recorded at the Max Planck Institute for Biogeochemistry. 420,451 rows.

# %%
path = "../data/jena_climate_2009_2016.csv"
if not os.path.exists(path):     # 13 MB, downloaded once
    os.makedirs("../data", exist_ok=True)
    urllib.request.urlretrieve("https://s3.amazonaws.com/keras-datasets/jena_climate_2009_2016.csv.zip", path + ".zip")
    zipfile.ZipFile(path + ".zip").extract("jena_climate_2009_2016.csv", "../data")

df = pd.read_csv(path)
print(df.shape)
df.head(3)

# %%
temperature = df["T (degC)"].to_numpy("float32")            # the target, in degrees Celsius
data = df.drop(columns="Date Time").to_numpy("float32")     # the 14 features (the temperature is one of them)

fig, ax = plt.subplots(1, 2, figsize=(13, 3))
ax[0].plot(temperature, lw=0.3, color="#1f3a68"); ax[0].set_title("Temperature, 8 years (°C)")
ax[1].plot(temperature[:1440], color="#1f3a68"); ax[1].set_title("The first 10 days: 1,440 readings")
plt.show()

# %% [markdown]
# Split by time: the first half for training, the next quarter for validation, the last quarter for the test.

# %%
n_train = int(0.5 * len(data))
n_val = int(0.25 * len(data))
print("train", n_train, "| validation", n_val, "| test", len(data) - n_train - n_val)

# %% [markdown]
# ## 2. Prepare it
#
# The 14 columns live on very different scales: pressure is around 990, humidity around 80, wind speed around 2.
# Each column is rescaled to mean 0 and standard deviation 1, with the mean and standard deviation of the
# **training** half only. The target stays in degrees, so every error below is in degrees.

# %%
mean = data[:n_train].mean(axis=0)
std = data[:n_train].std(axis=0)
data = (data - mean) / std
pd.DataFrame({"mean": mean, "std": std}, index=df.columns[1:]).round(2).T

# %% [markdown]
# ## 3. Cut it into windows
#
# One sample: one reading per hour (every 6th row) for 5 days, so **120 time steps with 14 features**. Its target: the
# temperature 24 hours after the window's last reading.
#
# For a window that starts at row `s`, the readings are rows `s, s+6, ..., s+714`, and the target is row `s + 858`
# (714 + 144: the last reading plus 24 hours of 10-minute rows).
#
# **How far ahead.** In the language of the first notebook, Tx = 120 hours in and Ty = 1 value out: every forecast in
# this notebook is **one step ahead**, made from real measurements. Multi-step (feeding forecasts back in) would need
# forecasts of all 14 measurements, not only the temperature. Multi-output works with many inputs too: try Ty = 24
# hours of temperature. One step, multi-step and multi-output have a video of their own, in theory and in Python.

# %%
RATE, LENGTH, AHEAD = 6, 120, 24                 # rows per hour, hours in, hours ahead
offsets = np.arange(LENGTH) * RATE               # 0, 6, ..., 714
delay = RATE * (LENGTH + AHEAD - 1)              # 858

def make_windows(start, end):
    """Windows that start in rows start ... end, each with the temperature 24 hours after its last reading."""
    starts = np.arange(start, min(end, len(data) - delay) - offsets[-1])
    X = data[starts[:, None] + offsets]          # (windows, 120, 14)
    y = temperature[starts + delay]
    return X, y

X_train, y_train = make_windows(0, n_train)
X_val, y_val = make_windows(n_train, n_train + n_val)
X_test, y_test = make_windows(n_train + n_val, len(data))
for name, X in [("train", X_train), ("validation", X_val), ("test", X_test)]:
    print(f"{name:10s} {X.shape}   {X.nbytes / 1e9:.1f} GB")

# %% [markdown]
# ## 4. A baseline to beat
#
# "In 24 hours it will be as warm as it is now." The temperature is column 1; undo the scaling to get degrees.

# %%
def naive(X, y):
    now = X[:, -1, 1] * std[1] + mean[1]         # the last reading's temperature, back in degrees
    return float(np.mean(np.abs(now - y)))

baseline = naive(X_test, y_test)
results = {"temperature now (baseline)": {"test": baseline, "best": baseline, "worst": baseline, "runs": 1, "validation": naive(X_val, y_val), "parameters": 0, "seconds per epoch": 0}}
print(f"validation {naive(X_val, y_val):.2f} degrees   test {naive(X_test, y_test):.2f} degrees")

# %% [markdown]
# ## 5. Build and train the models
#
# One function trains any model the same way: mean squared error as the loss, the mean absolute error (in degrees) as
# the number we read, 10 epochs, and the weights of the best validation epoch are the ones we test.
#
# One training run is one draw: the starting weights are random, and the result moves with them. A few runs are not
# enough to see that: for the dense network, three runs can land 0.2 degrees away from the middle of ten. So the five
# fast models are trained `RUNS` = 10 times each, and the table shows the middle run with the best and the worst.
# The two slow models are trained once.

# %%
class Clock(keras.callbacks.Callback):
    """Remembers how long each epoch took."""
    def on_train_begin(self, logs=None): self.seconds = []
    def on_epoch_begin(self, epoch, logs=None): self.start = time.time()
    def on_epoch_end(self, epoch, logs=None): self.seconds.append(time.time() - self.start)

def run(name, *hidden_layers, epochs=EPOCHS, runs=1):
    """Trains the model `runs` times from different starting weights and records the middle test error."""
    tests, histories, seconds = [], [], []
    for seed in range(runs):
        keras.utils.set_random_seed(seed)
        inputs = keras.Input(shape=(LENGTH, data.shape[-1]))
        x = inputs
        for layer in hidden_layers:
            x = type(layer).from_config(layer.get_config())(x)      # a fresh copy of the layer, with fresh starting weights
        model = keras.Model(inputs, layers.Dense(1)(x))
        model.compile(optimizer="rmsprop", loss="mse", metrics=["mae"])
        if seed == 0:
            model.summary()
        clock = Clock()
        best = keras.callbacks.EarlyStopping(monitor="val_mae", patience=epochs, restore_best_weights=True)
        history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=256,
                            callbacks=[clock, best], verbose=2 if seed == 0 else 0)
        tests.append(model.evaluate(X_test, y_test, batch_size=1024, verbose=0)[1])
        histories.append(history.history); seconds.append(np.mean(clock.seconds[1:]))
        print(f"run {seed + 1} of {runs}: test error {tests[-1]:.2f} degrees")
    results[name] = {"test": float(np.median(tests)), "best": min(tests), "worst": max(tests), "runs": runs,
                     "validation": float(np.median([min(h["val_mae"]) for h in histories])),
                     "parameters": model.count_params(), "seconds per epoch": float(np.mean(seconds))}
    h = histories[0]
    plt.figure(figsize=(7, 2.6))
    plt.plot(h["mae"], "o-", label="training", color="#1f3a68")
    plt.plot(h["val_mae"], "o-", label="validation", color="#e07b39")
    plt.axhline(results["temperature now (baseline)"]["validation"], color="grey", ls=":", label="baseline (validation)")
    plt.title(f"{name}: test error {results[name]['test']:.2f} degrees (the first run's curves)"); plt.xlabel("epoch"); plt.ylabel("degrees"); plt.legend(); plt.show()

# %% [markdown]
# ### A dense network
#
# `Flatten` unrolls a window into one long row: 120 x 14 = 1,680 numbers. Every one of them gets its own weight in
# every unit: 1,680 x 16 + 16 = **26,896** parameters in the first layer.

# %%
run("dense network", layers.Flatten(), layers.Dense(16, activation="relu"), runs=RUNS)

# %% [markdown]
# ### A 1D convolution
#
# A convolution slides a small window along the time axis and looks for the same pattern everywhere: 8 filters, each
# 24 hours wide over the 14 features (24 x 14 x 8 + 8 = 2,696 parameters). Pooling keeps the strongest response and
# halves the length. It treats every position alike, which suits images better than a forecast, where the most recent
# hours matter most.

# %%
run("1D convolution",
    layers.Conv1D(8, 24, activation="relu"), layers.MaxPooling1D(2),
    layers.Conv1D(8, 12, activation="relu"), layers.MaxPooling1D(2),
    layers.Conv1D(8, 6, activation="relu"), layers.GlobalAveragePooling1D(), runs=RUNS)

# %% [markdown]
# ### The simple RNN
#
# 14 x 16 weights on the input, 16 x 16 on the previous state, 16 biases: **496** parameters, shared by all 120 steps.

# %%
run("simple RNN (16)", layers.SimpleRNN(16), runs=RUNS)

# %% [markdown]
# ### The LSTM
#
# Three gates and a candidate, each with its own copy of those weights: 4 x 496 = **1,984**.

# %%
run("LSTM (16)", layers.LSTM(16), runs=RUNS)

# %% [markdown]
# ### A bidirectional LSTM
#
# Two LSTMs, one reading the window forward and one backward, their final states joined: 2 x 1,984 parameters. The
# backward copy sees the most recent hours first and the oldest last.

# %%
run("bidirectional LSTM (16)", layers.Bidirectional(layers.LSTM(16)), runs=RUNS)

# %% [markdown]
# ### LSTM with recurrent dropout (slow)
#
# A bigger LSTM (32 units) overfits, so it gets dropout. Inside the recurrent loop Keras uses `recurrent_dropout`:
# one random mask per sequence, the same at every time step. The price: the layer can no longer use the fast GPU
# routine for recurrent layers, and an epoch takes tens of times longer.

# %%
if RUN_SLOW:
    run("LSTM (32) + recurrent dropout", layers.LSTM(32, recurrent_dropout=0.25), layers.Dropout(0.5))

# %% [markdown]
# ### Two stacked GRU layers with dropout (slower)
#
# The GRU is a lighter cousin of the LSTM: two gates instead of three, and no separate cell state. The first layer
# returns its state at every step (`return_sequences=True`) so that the second layer has a sequence to read.

# %%
if RUN_SLOW:
    run("two GRU layers (32) + dropout",
        layers.GRU(32, recurrent_dropout=0.5, return_sequences=True),
        layers.GRU(32, recurrent_dropout=0.5), layers.Dropout(0.5))

# %% [markdown]
# ## 6. Score them in real units
#
# One table, one chart, everything in degrees on the test quarter. `test` is the middle run; `best` and `worst` show
# how far apart the runs landed. Your numbers will differ a little from these and from the video: the starting weights
# are random, and a GPU does not add numbers in the same order twice. The dense network and the convolution move the
# most from run to run, and now and then a run of the convolution falls apart completely (it lands off the chart;
# the `worst` column shows it). The simple RNN is the steadiest.

# %%
table = pd.DataFrame(results).T
table[["parameters", "runs"]] = table[["parameters", "runs"]].astype(int)
table.round(2)

# %%
order = table.sort_values("test", ascending=False)
plt.figure(figsize=(9, 4))
plt.barh(order.index, order["test"], color=["#9aa7b8" if "baseline" in n else "#1f3a68" for n in order.index])
for name, lo, hi in zip(order.index, order["best"], order["worst"]):
    plt.plot([lo, hi], [name, name], color="black", lw=2)
plt.axvline(table.loc["temperature now (baseline)", "test"], color="grey", ls=":")
plt.xlim(2, 3.6); plt.xlabel("error on the test data, degrees Celsius (lower is better)"); plt.tight_layout(); plt.show()

# %% [markdown]
# ## What this shows
#
# - The baseline is hard to beat: a dense network with 27,000 weights ends up level with it at best, and the
#   convolution well behind it.
# - The recurrent models beat it, with a few hundred to a few thousand weights. They read the window in order and
#   reuse the same weights at every hour. Only the simple RNN beats it in every single run.
# - The LSTM does not beat the simple RNN here. This task rewards the last day or two, not a long memory.
# - Reading the window backward as well does not help a forecast.
# - Regularized, larger recurrent models gain a little more, and cost far more time.
# - The gains are tenths of a degree. Weather 24 hours ahead from one station is a hard problem.
#
# ## Try it yourself, with a coding assistant
#
# - "Forecast 12 hours ahead instead of 24." (Change `AHEAD`.)
# - "Use only the temperature column as input, and compare."
# - "Add a GRU with 16 units and no dropout to the table."
# - "Rewrite this for PyTorch." or "Update this for the current version of Keras."
#
# Then check what you get back: the shapes of the windows, that the split is by time, that the baseline is still in
# the table, and that the test quarter is only used for the final score.
