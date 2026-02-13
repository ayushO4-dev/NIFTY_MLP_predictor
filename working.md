# Code Walkthrough: main.py

This document explains the code in `main.py` in precise detail, step by step, aligned with the current implementation.

## 1) Imports and data model

The script imports standard libraries, data science packages, and scikit-learn models. The core result object is a small dataclass:

- `ModelResult`: stores the trained `model`, the `StandardScaler`, `feature_names`, `target_name`, and `metrics` (accuracy or MAE/RMSE).

This keeps everything needed for prediction and reporting in one place.

## 2) Output helpers

- `print_section(title)`: prints a clean section header for terminal output.
- `print_kv(label, value)`: prints a single key/value line for consistent formatting.

The earlier step-by-step print function was removed to keep output focused.

## 3) Yahoo Finance data loading with caching

### `sanitize_cache_key(text)`
Replaces non-alphanumeric characters with `_`. This makes safe filenames for the cache.

### `load_yahoo_history(ticker, period, interval)`
This function fetches OHLCV data and returns a tuple: `(history_df, cache_hit)`.

Flow:
1. Build a cache directory at `data/yahoo_cache`.
2. Create a cache key from `ticker`, `period`, `interval`.
3. If a matching cache CSV + JSON metadata exists, load and return it (with `cache_hit=True`).
4. Otherwise download from Yahoo Finance:
   - `yf.download(ticker, period=..., interval=..., auto_adjust=False, progress=False)`
5. Normalize the dataframe:
   - Fix MultiIndex columns if present.
   - Rename date column to `date` and OHLCV columns to lower-case (`open`, `high`, `low`, `close`, `volume`).
   - Parse `date` into datetime and drop invalid rows.
6. Save the cleaned dataframe to CSV, save metadata to JSON, return with `cache_hit=False`.

This is the mechanism that avoids re-downloading when the period/interval do not change.

## 4) Snapshot parsing and merging

### `parse_numeric(value)`
Converts a string like `"11,356.40"` or `"-"` into a float or `NaN`. This is needed because bhavcopy files often contain commas and placeholders.

### `load_snapshot(csv_path, symbol)`
Reads a single-day bhavcopy snapshot (no date column), extracts the row for the selected symbol, and returns a `pd.Series` with:

- `open`, `high`, `low` from `OPEN/HIGH/LOW`
- `close` from `LTP` if present, else `PREV_CLOSE`
- `volume` from `VOLUME` (optional)
- `change`, `change_percent`, `change_30d_percent` if present

### `apply_snapshot(history, snapshot)`
Overwrites the last row in Yahoo data with snapshot values (only if they are not `NaN`). This ensures the most recent candle uses official bhavcopy values while retaining historical Yahoo data.

## 5) Feature engineering

### `add_pattern_features(frame)`
Builds candlestick pattern features using OHLC data:

- `pattern_doji`: small candle body relative to range
- `pattern_hammer`: long lower wick, small upper wick
- `pattern_shooting_star`: long upper wick, small lower wick
- `pattern_bullish_engulf`: bullish candle that engulfs prior bearish body
- `pattern_bearish_engulf`: bearish candle that engulfs prior bullish body

Each pattern is converted to a 0/1 integer feature.

### `add_numeric_features(frame)`
Adds numeric features derived from prices and volume:

- `return_1d`: daily return
- `change`: absolute close change
- `change_percent`: close percent change
- `change_30d_percent`: 30-day percent change
- `range_pct`: (high-low)/close
- `body_pct`: (close-open)/close
- `volume_change`: volume percent change
- `return_5d`: 5-day return
- `volatility_5d`: 5-day standard deviation of daily returns
- `ma_5`, `ma_10`: moving averages
- `ma_5_gap`, `ma_10_gap`: distance of close from moving averages

### `sanitize_features(frame, feature_names)`
Replaces `inf` and `-inf` with `NaN`, which can happen with `pct_change` when a prior close is zero.

### `filter_feature_names(frame, feature_names)`
Drops features with fewer than 5 valid data points. This prevents extremely short history windows (e.g., 30-day features on only 1 month of data) from wiping out all rows.

## 6) Target construction

### `prepare_dataset(frame, target_mode)`
Creates the prediction target by shifting the close forward one row:

- `next_close` = `close.shift(-1)`

Then sets `target` based on `target_mode`:

- `close`: predict the next close price
- `return`: predict next-day return
- `direction`: predict up (1) or down (0) relative to today

Rows with `NaN` target are dropped.

## 7) Model training

### `train_model(frame, feature_names, target_name, max_iter, hidden_layers)`
1. Splits the data chronologically: first 80% train, last 20% test.
2. Fits `StandardScaler` on training features and transforms train/test.
3. Trains one of:
   - `MLPClassifier` for direction
   - `MLPRegressor` for close/return
4. Computes metrics:
   - Accuracy for classification
   - MAE and RMSE for regression
5. Returns `ModelResult` with the model, scaler, and metrics.

## 8) Hidden layer math (manual forward pass)

### `explain_hidden_layer(model, sample, is_classifier)`
This makes the hidden-layer calculations explicit by manually computing one forward pass:

1. Extract weights and biases: `W1`, `b1`, `W2`, `b2`.
2. Compute hidden pre-activation:
   - $z_1 = xW_1 + b_1$
3. Apply `tanh` activation:
   - $a_1 = tanh(z_1)$
4. Compute output pre-activation:
   - $z_2 = a_1W_2 + b_2$
5. Apply output activation:
   - Sigmoid for classifier
   - Identity for regression

The function prints all intermediate vectors with consistent precision for readability.

## 9) Next-day OHLC prediction

### `predict_next_day(frame, model_result, target_mode, output_path)`
Uses the last row of features to predict the next bar and then constructs an estimated OHLC:

- `pred_close`: predicted close
- `pred_open`: last close
- `pred_high`, `pred_low`: offset by half the recent average range

It prints the OHLC estimate and saves a simple chart (`predicted_next_day.png` by default).

## 10) Main execution flow

### `main()`
1. Parse CLI arguments.
2. Build Yahoo ticker (default `SYMBOL.NS`).
3. Load Yahoo history (cached or fetched).
4. Optionally merge snapshot.
5. Compute features, target, sanitize, and filter.
6. Train the model.
7. Print model stats and data summary.
8. Run hidden-layer explanation.
9. Predict next-day OHLC and save the chart.

## 11) Key CLI flags

- `--symbol`: NIFTY symbol (e.g., RELIANCE)
- `--snapshot`: previous-day bhavcopy CSV
- `--history-period`: Yahoo period (`1mo`, `6mo`, `1y`, `10y`)
- `--history-interval`: Yahoo interval (`1d`, `1h`)
- `--target`: `close`, `return`, `direction`
- `--max-iter`: training iterations
- `--hidden-layers`: comma-separated layer sizes (e.g., `8` or `16,8`)
- `--chart`: output chart path

## 12) Hidden layer tuning script

The `tune_hidden_layers.py` script runs a small grid search over hidden-layer
sizes using the same feature pipeline as `main.py`.

### How it works
1. Loads Yahoo history and optional snapshot.
2. Builds features, targets, and filters invalid rows.
3. Trains an MLP for each layer configuration.
4. Prints accuracy (direction) or RMSE/MAE (regression) and reports the best.

### Key CLI flags
- `--hidden-grid`: semicolon-separated list of layer sizes (e.g., `8;16;16,8`)
- `--history-period` / `--history-interval`: same as in `main.py`
- `--max-iter`: training iterations

---

