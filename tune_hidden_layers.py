from __future__ import annotations

import argparse
from pathlib import Path

from main import (
    add_numeric_features,
    add_pattern_features,
    apply_snapshot,
    filter_feature_names,
    load_snapshot,
    load_yahoo_history,
    prepare_dataset,
    sanitize_features,
    train_model,
)


def parse_hidden_grid(value: str) -> list[tuple[int, ...]]:
    configs: list[tuple[int, ...]] = []
    for chunk in value.split(";"):
        text = chunk.strip()
        if not text:
            continue
        parts = [part.strip() for part in text.split(",")]
        layers = tuple(int(part) for part in parts)
        if any(layer <= 0 for layer in layers):
            raise ValueError("Hidden layer sizes must be positive integers.")
        configs.append(layers)
    if not configs:
        raise ValueError("Hidden grid is empty. Example: 8;16;16,8")
    return configs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hidden layer grid search")
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
        "--max-iter",
        type=int,
        default=2000,
        help="Maximum training iterations for MLP",
    )
    parser.add_argument(
        "--hidden-grid",
        default="8;16;16,8;32,16",
        help="Hidden layer grid, semicolon-separated (e.g., 8;16;16,8)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    symbol = args.symbol.upper()
    symbol_base = symbol.split(".")[0]
    ticker = args.yahoo_ticker
    if not ticker:
        ticker = f"{symbol}.NS" if "." not in symbol else symbol

    frame, _cache_hit = load_yahoo_history(
        ticker,
        args.history_period,
        args.history_interval,
    )
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
    feature_names, _dropped = filter_feature_names(frame, feature_names)
    if not feature_names:
        raise ValueError("No usable features after filtering; need more history.")

    frame = frame.dropna(subset=feature_names + ["target"])
    if frame.empty:
        raise ValueError("Not enough rows after feature engineering; need more history.")

    grid = parse_hidden_grid(args.hidden_grid)
    results: list[tuple[tuple[int, ...], dict[str, float]]] = []
    for layers in grid:
        result = train_model(
            frame,
            feature_names,
            target_name,
            args.max_iter,
            layers,
        )
        results.append((layers, result.metrics))

    print("\nGrid search results")
    print("=" * 72)
    best_layers: tuple[int, ...] | None = None
    best_score: float | None = None
    for layers, metrics in results:
        if "accuracy" in metrics:
            score = metrics["accuracy"]
            print(f"layers={layers} accuracy={score:.3f}")
            if best_score is None or score > best_score:
                best_score = score
                best_layers = layers
        else:
            score = metrics.get("rmse", float("inf"))
            mae = metrics.get("mae", float("nan"))
            print(f"layers={layers} rmse={score:.6f} mae={mae:.6f}")
            if best_score is None or score < best_score:
                best_score = score
                best_layers = layers

    if best_layers is not None:
        label = "accuracy" if args.target == "direction" else "rmse"
        print("-" * 72)
        print(f"Best layers: {best_layers} ({label}={best_score:.6f})")


if __name__ == "__main__":
    main()
