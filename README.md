AI Hidden Layer Demo with NIFTY Bhavcopy
=======================================

Purpose
-------
This project demonstrates how a simple AI model (MLP) learns from
NIFTY bhavcopy data and how its hidden-layer math works. It builds features from
candlestick patterns, trains a model to predict the next day, and prints the
hidden-layer calculations for one sample.

What this project does
----------------------
1. Downloads Yahoo Finance candles (cached offline).
2. Optionally merges a previous-day bhavcopy snapshot.
3. Builds pattern recognition features such as doji, hammer, and engulfing.
4. Creates numerical features like returns, volatility, and moving average gaps.
5. Shifts the target so the model predicts the next trading day.
6. Trains an MLP (hidden layer).
7. Prints evaluation metrics and a next-day OHLC estimate.
8. Shows manual hidden-layer calculations for one scaled input.

Data requirements
-----------------
Yahoo Finance provides the daily time series. The optional snapshot CSV should
include at least these columns (case-insensitive):
- Symbol: SYMBOL
- OHLC: OPEN, HIGH, LOW
- Close: LTP (preferred) or PREV_CLOSE
- Volume is optional (VOLUME)

If your column names differ, rename them before running.

Install
-------
Create the environment and install dependencies:

```
uv sync
```

Usage
-----
Run the demo using Yahoo Finance candles and an optional previous-day bhavcopy snapshot:

```
python main.py --symbol RELIANCE --snapshot data/NIFTY_OIL_&_GAS.csv
```

Optional flags:
- --yahoo-ticker RELIANCE.NS (override the Yahoo ticker)
- --history-period 1mo|6mo|1y|10y (default: 10y)
- --history-interval 1d|1h (default: 1d)
- --target close|return|direction (default: direction)
- --max-iter 2000 (training iterations)
- --hidden-layers 8 or 16,8 (hidden layer sizes)
- --chart predicted_next_day.png (output chart path)

Example:

```
python main.py --symbol TCS --snapshot data/NIFTY_OIL_&_GAS.csv --history-period 1mo --history-interval 1d --target direction --hidden-layers 16,8
```

Hidden layer tuning
-------------------
Use the separate tuner script to compare multiple layer configurations:

```
python tune_hidden_layers.py --symbol RELIANCE --snapshot data/NIFTY_OIL_&_GAS.csv --history-period 1mo --history-interval 1d --hidden-grid "8;16;16,8;32,16"
```

Snapshot expectations
---------------------
The snapshot CSV is a single-day bhavcopy file without DATE. The script replaces
the most recent Yahoo candle with this snapshot, so the model uses your latest
official OHLC values.

If the snapshot includes LTP, it is used as the close; otherwise PREV_CLOSE is
used. Volume is optional.

Yahoo Finance uses .NS tickers for NSE stocks, so RELIANCE becomes RELIANCE.NS
unless you pass --yahoo-ticker explicitly.

Offline cache
-------------
Yahoo Finance data is cached under data/yahoo_cache. The script reuses the cached
file when the ticker, period, and interval match; otherwise it fetches fresh data.

How the hidden layer is explained
---------------------------------
When the model is an MLP, the code prints a manual forward pass:

1. Scale the input features with the same scaler used in training.
2. Compute the hidden pre-activation: $z_1 = xW_1 + b_1$
3. Apply tanh activation: $a_1 = tanh(z_1)$
4. Compute output pre-activation: $z_2 = a_1W_2 + b_2$
5. Apply output activation (identity for regression, sigmoid for direction).

This makes the hidden-layer calculations explicit and easy to inspect.

Notes and limitations
---------------------
- This is a learning demo, not trading advice.
- The next-day OHLC is a simple estimate based on recent range.
- Real trading systems require robust validation and risk controls.

