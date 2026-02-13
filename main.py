from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler


@dataclass
class ModelResult:
    model: object
    scaler: StandardScaler
    feature_names: list[str]
    target_name: str
    metrics: dict[str, float]


def print_section(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("-" * 72)


def print_kv(label: str, value: str) -> None:
    print(f"{label}: {value}")


def sanitize_cache_key(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)


def load_yahoo_history(ticker: str, period: str, interval: str) -> tuple[pd.DataFrame, bool]:
    cache_dir = Path("data") / "yahoo_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_key = f"{sanitize_cache_key(ticker)}_{sanitize_cache_key(period)}_{sanitize_cache_key(interval)}"
    cache_file = cache_dir / f"{cache_key}.csv"
    meta_file = cache_dir / f"{cache_key}.json"

    if cache_file.exists() and meta_file.exists():
        try:
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            meta = {}
        if meta.get("ticker") == ticker and meta.get("period") == period and meta.get("interval") == interval:
            cached = pd.read_csv(cache_file)
            cached["date"] = pd.to_datetime(cached["date"], errors="coerce")
            cached = cached.dropna(subset=["date"])
            return cached, True

    history = yf.download(
        ticker,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
    )
    if history.empty:
        raise ValueError(f"No Yahoo Finance data returned for '{ticker}'.")
    if isinstance(history.columns, pd.MultiIndex):
        history.columns = [col[0] for col in history.columns]
    history = history.reset_index()
    if "Date" in history.columns:
        history = history.rename(columns={"Date": "date"})
    elif "Datetime" in history.columns:
        history = history.rename(columns={"Datetime": "date"})
    elif "index" in history.columns:
        history = history.rename(columns={"index": "date"})
    history = history.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )
    if "date" not in history.columns:
        raise ValueError("Yahoo Finance data did not include a date column.")
    history["date"] = pd.to_datetime(history["date"], errors="coerce")
    history = history.dropna(subset=["date"])
    history.to_csv(cache_file, index=False)
    meta_file.write_text(
        json.dumps({"ticker": ticker, "period": period, "interval": interval}, indent=2),
        encoding="utf-8",
    )
    return history, False


def parse_numeric(value: object) -> float:
    if pd.isna(value):
        return float("nan")
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).replace(",", "").strip()
    if text in {"-", ""}:
        return float("nan")
    return float(text)


def load_snapshot(csv_path: Path, symbol: str) -> pd.Series:
    frame = pd.read_csv(csv_path)
    frame.columns = [col.strip().upper() for col in frame.columns]
    symbol = symbol.upper()
    if "SYMBOL" not in frame.columns:
        raise ValueError("Snapshot CSV must include SYMBOL column.")
    row = frame.loc[frame["SYMBOL"].astype(str).str.upper() == symbol]
    if row.empty:
        raise ValueError(f"No snapshot row found for symbol '{symbol}'.")
    row = row.iloc[0]
    close_source = "LTP" if "LTP" in frame.columns else "PREV_CLOSE"
    snapshot = pd.Series(
        {
            "open": parse_numeric(row.get("OPEN")),
            "high": parse_numeric(row.get("HIGH")),
            "low": parse_numeric(row.get("LOW")),
            "close": parse_numeric(row.get(close_source)),
            "volume": parse_numeric(row.get("VOLUME")),
            "change": parse_numeric(row.get("CHANGE")),
            "change_percent": parse_numeric(row.get("CHANGE_PERCENT")),
            "change_30d_percent": parse_numeric(row.get("30D_CHANGE_PERCENT")),
        }
    )
    if snapshot[["open", "high", "low", "close"]].isna().any():
        raise ValueError("Snapshot row is missing required OHLC values.")
    return snapshot


def apply_snapshot(history: pd.DataFrame, snapshot: pd.Series) -> pd.DataFrame:
    history = history.copy()
    last_idx = history.index[-1]
    for col in [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "change",
        "change_percent",
        "change_30d_percent",
    ]:
        if not pd.isna(snapshot[col]):
            history.loc[last_idx, col] = snapshot[col]
    return history


def add_pattern_features(frame: pd.DataFrame) -> pd.DataFrame:
    eps = 1e-9
    body = frame["close"] - frame["open"]
    candle_range = (frame["high"] - frame["low"]).replace(0, eps)
    upper_wick = frame["high"] - frame[["close", "open"]].max(axis=1)
    lower_wick = frame[["close", "open"]].min(axis=1) - frame["low"]

    doji = (body.abs() / candle_range) < 0.1
    hammer = (lower_wick > 2 * body.abs()) & (upper_wick <= body.abs())
    shooting_star = (upper_wick > 2 * body.abs()) & (lower_wick <= body.abs())

    prev_open = frame["open"].shift(1)
    prev_close = frame["close"].shift(1)
    bullish_engulf = (body > 0) & (prev_close < prev_open) & (frame["close"] > prev_open)
    bearish_engulf = (body < 0) & (prev_close > prev_open) & (frame["close"] < prev_open)

    frame["pattern_doji"] = doji.astype(int)
    frame["pattern_hammer"] = hammer.astype(int)
    frame["pattern_shooting_star"] = shooting_star.astype(int)
    frame["pattern_bullish_engulf"] = bullish_engulf.astype(int)
    frame["pattern_bearish_engulf"] = bearish_engulf.astype(int)
    return frame


def add_numeric_features(frame: pd.DataFrame) -> pd.DataFrame:
    frame["return_1d"] = frame["close"].pct_change()
    frame["change"] = frame["close"].diff()
    frame["change_percent"] = frame["close"].pct_change() * 100
    frame["change_30d_percent"] = frame["close"].pct_change(30) * 100
    frame["range_pct"] = (frame["high"] - frame["low"]) / frame["close"].replace(0, np.nan)
    frame["body_pct"] = (frame["close"] - frame["open"]) / frame["close"].replace(0, np.nan)
    frame["volume_change"] = frame["volume"].pct_change()
    frame["return_5d"] = frame["close"].pct_change(5)
    frame["volatility_5d"] = frame["return_1d"].rolling(5).std()
    frame["ma_5"] = frame["close"].rolling(5).mean()
    frame["ma_10"] = frame["close"].rolling(10).mean()
    frame["ma_5_gap"] = (frame["close"] - frame["ma_5"]) / frame["ma_5"]
    frame["ma_10_gap"] = (frame["close"] - frame["ma_10"]) / frame["ma_10"]
    return frame


def filter_feature_names(
    frame: pd.DataFrame,
    feature_names: list[str],
) -> tuple[list[str], list[str]]:
    min_non_nan = 5
    kept = []
    dropped = []
    for name in feature_names:
        if frame[name].notna().sum() >= min_non_nan:
            kept.append(name)
        else:
            dropped.append(name)
    return kept, dropped


def sanitize_features(frame: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    frame = frame.copy()
    frame[feature_names] = frame[feature_names].replace([np.inf, -np.inf], np.nan)
    return frame


def prepare_dataset(frame: pd.DataFrame, target_mode: str) -> Tuple[pd.DataFrame, str]:
    frame = frame.copy()
    frame["next_close"] = frame["close"].shift(-1)
    if target_mode == "close":
        frame["target"] = frame["next_close"]
        target_name = "next_close"
    elif target_mode == "return":
        frame["target"] = frame["next_close"] / frame["close"] - 1
        target_name = "next_return"
    elif target_mode == "direction":
        frame["target"] = (frame["next_close"] > frame["close"]).astype(int)
        target_name = "next_direction"
    else:
        raise ValueError("target_mode must be one of: close, return, direction")

    frame = frame.dropna(subset=["target"])
    return frame, target_name


def train_model(
    frame: pd.DataFrame,
    feature_names: list[str],
    target_name: str,
    max_iter: int,
    hidden_layers: tuple[int, ...],
) -> ModelResult:
    split_idx = int(len(frame) * 0.8)
    train = frame.iloc[:split_idx]
    test = frame.iloc[split_idx:]

    scaler = StandardScaler()
    x_train = scaler.fit_transform(train[feature_names].to_numpy())
    x_test = scaler.transform(test[feature_names].to_numpy())
    y_train = train["target"].to_numpy()
    y_test = test["target"].to_numpy()

    metrics: dict[str, float] = {}
    if target_name == "next_direction":
        model = MLPClassifier(
            hidden_layer_sizes=hidden_layers,
            activation="tanh",
            max_iter=max_iter,
            random_state=42,
        )
        model.fit(x_train, y_train)
        preds = model.predict(x_test)
        score = accuracy_score(y_test, preds)
        metrics["accuracy"] = float(score)
    else:
        model = MLPRegressor(
            hidden_layer_sizes=hidden_layers,
            activation="tanh",
            max_iter=max_iter,
            random_state=42,
        )
        model.fit(x_train, y_train)
        preds = model.predict(x_test)
        mae = mean_absolute_error(y_test, preds)
        rmse = mean_squared_error(y_test, preds, squared=False)
        metrics["mae"] = float(mae)
        metrics["rmse"] = float(rmse)

    return ModelResult(
        model=model,
        scaler=scaler,
        feature_names=feature_names,
        target_name=target_name,
        metrics=metrics,
    )


def activation_tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def parse_hidden_layers(value: str) -> tuple[int, ...]:
    text = value.strip()
    if not text:
        raise ValueError("Hidden layers must be a comma-separated list of integers.")
    parts = [part.strip() for part in text.split(",")]
    layers = tuple(int(part) for part in parts)
    if any(layer <= 0 for layer in layers):
        raise ValueError("Hidden layer sizes must be positive integers.")
    return layers


def explain_hidden_layer(model: object, sample: np.ndarray, is_classifier: bool) -> None:
    print_section("Hidden Layer calculations")
    if not hasattr(model, "coefs_"):
        print("Model does not expose hidden-layer weights.")
        return
    np.set_printoptions(precision=4, suppress=True)
    weights_0 = model.coefs_[0]
    bias_0 = model.intercepts_[0]
    hidden_raw = np.dot(sample, weights_0) + bias_0
    hidden_act = activation_tanh(hidden_raw)

    weights_1 = model.coefs_[1]
    bias_1 = model.intercepts_[1]
    output_raw = np.dot(hidden_act, weights_1) + bias_1
    if is_classifier:
        output = sigmoid(output_raw)
    else:
        output = output_raw

    print("Input sample (scaled):")
    print(np.array2string(sample, separator=", "))
    print("Hidden pre-activation:")
    print(np.array2string(hidden_raw, separator=", "))
    print("Hidden activation (tanh):")
    print(np.array2string(hidden_act, separator=", "))
    print("Output raw:")
    print(np.array2string(output_raw, separator=", "))
    print("Output activation:")
    print(np.array2string(output, separator=", "))


def print_weights_biases(model: object) -> None:
    print_section("Weights and biases")
    if not hasattr(model, "coefs_"):
        print("Model does not expose weights and biases.")
        return
    np.set_printoptions(precision=4, suppress=True)
    for idx, (weights, biases) in enumerate(zip(model.coefs_, model.intercepts_), start=1):
        print(f"Layer {idx} weights shape: {weights.shape}")
        print(np.array2string(weights, separator=", "))
        print(f"Layer {idx} biases shape: {biases.shape}")
        print(np.array2string(biases, separator=", "))


def predict_next_day(
    frame: pd.DataFrame,
    model_result: ModelResult,
    target_mode: str,
    output_path: Path,
) -> None:
    last_row = frame.iloc[-1]
    features = model_result.scaler.transform(
        last_row[model_result.feature_names].to_numpy().reshape(1, -1)
    )
    prediction = model_result.model.predict(features)

    last_close = float(last_row["close"])
    avg_range = float((frame["high"] - frame["low"]).tail(5).mean())
    if target_mode == "direction":
        direction = int(prediction[0])
        avg_return = float(frame["close"].pct_change().abs().tail(10).mean())
        pred_close = last_close * (1 + (1 if direction == 1 else -1) * avg_return)
    elif target_mode == "return":
        pred_close = last_close * (1 + float(prediction[0]))
    else:
        pred_close = float(prediction[0])

    pred_open = last_close
    pred_high = max(pred_open, pred_close) + 0.5 * avg_range
    pred_low = min(pred_open, pred_close) - 0.5 * avg_range
    pred_low = max(0.0, pred_low)

    print("Predicted next-day OHLC:")
    print_kv("Open", f"{pred_open:.2f}")
    print_kv("High", f"{pred_high:.2f}")
    print_kv("Low", f"{pred_low:.2f}")
    print_kv("Close", f"{pred_close:.2f}")

    history = frame.tail(40)
    plt.figure(figsize=(10, 5))
    plt.plot(history["date"], history["close"], label="Actual close")
    next_date = history["date"].iloc[-1] + pd.Timedelta(days=1)
    plt.scatter([next_date], [pred_close], color="orange", label="Predicted next close")
    plt.title("Next-day prediction")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print_kv("Chart", str(output_path))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NIFTY bhavcopy AI demo")
    parser.add_argument("--symbol", required=True, help="NIFTY symbol to analyze")
    parser.add_argument(
        "--yahoo-ticker",
        default=None,
        help="Yahoo Finance ticker (default: SYMBOL.NS)",
    )
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=None,
        help="Path to previous-day bhavcopy snapshot CSV",
    )
    parser.add_argument(
        "--history-period",
        default="10y",
        help="Yahoo Finance period (e.g., 1mo, 6mo, 1y, 10y)",
    )
    parser.add_argument(
        "--history-interval",
        default="1d",
        help="Yahoo Finance interval (e.g., 1d, 1h)",
    )
    parser.add_argument(
        "--target",
        choices=["close", "return", "direction"],
        default="direction",
        help="Target to predict",
    )
    parser.add_argument(
        "--hidden-layers",
        default="8",
        help="Comma-separated hidden layer sizes (e.g., 8 or 16,8)",
    )
    parser.add_argument(
        "--chart",
        type=Path,
        default=Path("predicted_next_day.png"),
        help="Output chart file",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=2000,
        help="Maximum training iterations for perceptron/MLP",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    hidden_layers = parse_hidden_layers(args.hidden_layers)
    symbol = args.symbol.upper()
    symbol_base = symbol.split(".")[0]
    ticker = args.yahoo_ticker
    if not ticker:
        ticker = f"{symbol}.NS" if "." not in symbol else symbol
    frame, cache_hit = load_yahoo_history(ticker, args.history_period, args.history_interval)
    frame["symbol"] = symbol_base
    if args.snapshot:
        snapshot = load_snapshot(args.snapshot, symbol_base)
        frame = apply_snapshot(frame, snapshot)

    frame = add_pattern_features(frame)
    frame = add_numeric_features(frame)
    frame, target_name = prepare_dataset(frame, args.target)

    feature_names = [
        "return_1d",
        "change",
        "change_percent",
        "change_30d_percent",
        "range_pct",
        "body_pct",
        "volume_change",
        "return_5d",
        "volatility_5d",
        "ma_5_gap",
        "ma_10_gap",
        "pattern_doji",
        "pattern_hammer",
        "pattern_shooting_star",
        "pattern_bullish_engulf",
        "pattern_bearish_engulf",
    ]

    frame = sanitize_features(frame, feature_names)
    feature_names, dropped_features = filter_feature_names(frame, feature_names)
    if not feature_names:
        raise ValueError("No usable features after filtering; need more history.")

    frame = frame.dropna(subset=feature_names + ["target"])
    if frame.empty:
        raise ValueError("Not enough rows after feature engineering; need more history.")
    model_result = train_model(
        frame,
        feature_names,
        target_name,
        args.max_iter,
        hidden_layers,
    )

    print_section("Model related statistics and prediction")
    data_start = frame["date"].min().date()
    data_end = frame["date"].max().date()
    source_label = "cache" if cache_hit else "yahoo"
    print_kv("Data", f"{ticker} {args.history_period}/{args.history_interval}")
    print_kv("Range", f"{data_start} to {data_end} ({len(frame)} rows)")
    print_kv("Source", source_label)
    if dropped_features:
        print("Filtered features: " + ", ".join(dropped_features))
    if "accuracy" in model_result.metrics:
        print_kv("Accuracy", f"{model_result.metrics['accuracy']:.3f}")
    if "mae" in model_result.metrics:
        print_kv("MAE", f"{model_result.metrics['mae']:.6f}")
    if "rmse" in model_result.metrics:
        print_kv("RMSE", f"{model_result.metrics['rmse']:.6f}")
    print_kv("Hidden layers", ", ".join(str(layer) for layer in hidden_layers))

    sample = model_result.scaler.transform(
        frame[feature_names].iloc[-1].to_numpy().reshape(1, -1)
    ).squeeze(0)
    # print_weights_biases(model_result.model)
    explain_hidden_layer(
        model_result.model,
        sample,
        is_classifier=target_name == "next_direction",
    )

    predict_next_day(frame, model_result, args.target, args.chart)


if __name__ == "__main__":
    main()
