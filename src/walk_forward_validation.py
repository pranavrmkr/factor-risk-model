
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score




ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed"

RETURNS_FILE = DATA_DIR / "daily_returns.csv"
FACTORS_FILE = DATA_DIR / "factor_returns.csv"

TARGET = "HDFCBANK.NS"
FACTORS = ["Market", "Momentum", "Value"]

MIN_TRAIN_SIZE = 252       # Approximately one trading year
TEST_BLOCK_SIZE = 21       




def load_data():
    stock_returns = pd.read_csv(
        RETURNS_FILE, index_col="Date", parse_dates=True
    )
    factor_returns = pd.read_csv(
        FACTORS_FILE, index_col="Date", parse_dates=True
    )

    if TARGET not in stock_returns.columns:
        raise ValueError(f"Missing target column: {TARGET}")

    missing = [f for f in FACTORS if f not in factor_returns.columns]
    if missing:
        raise ValueError(f"Missing factor columns: {missing}")

    target = stock_returns[TARGET].rename("Actual")
    factors = factor_returns[FACTORS]

    
    data = pd.concat([target, factors], axis=1, join="inner")
    data = data.sort_index()
    data = data.replace([np.inf, -np.inf], np.nan).dropna()

    
    data[FACTORS] = data[FACTORS].shift(1)
    data = data.dropna()

    if not data.index.is_unique:
        raise ValueError("Duplicate dates found.")

    if len(data) <= MIN_TRAIN_SIZE:
        raise ValueError(
            "Not enough observations for walk-forward validation."
        )

    return data




def walk_forward_predict(data):
    predictions = []
    fold_number = 0

    # Initial training window, followed by sequential test blocks.
    for test_start in range(
        MIN_TRAIN_SIZE, len(data), TEST_BLOCK_SIZE
    ):
        test_end = min(
            test_start + TEST_BLOCK_SIZE, len(data)
        )

        train = data.iloc[:test_start]
        test = data.iloc[test_start:test_end]

        if test.empty:
            continue

        y_train = train["Actual"]

        # Market-only benchmark.
        X_market_train = sm.add_constant(
            train[["Market"]], has_constant="add"
        )
        market_model = sm.OLS(
            y_train, X_market_train
        ).fit()

        # Three-factor model.
        X_factor_train = sm.add_constant(
            train[FACTORS], has_constant="add"
        )
        factor_model = sm.OLS(
            y_train, X_factor_train
        ).fit()

        X_market_test = sm.add_constant(
            test[["Market"]], has_constant="add"
        )
        X_factor_test = sm.add_constant(
            test[FACTORS], has_constant="add"
        )

        fold = pd.DataFrame(index=test.index)
        fold["Actual"] = test["Actual"]
        fold["Market_Only"] = market_model.predict(X_market_test)
        fold["Three_Factor"] = factor_model.predict(X_factor_test)
        fold["Zero_Baseline"] = 0.0
        fold["Fold"] = fold_number
        fold["Training_Observations"] = len(train)

        predictions.append(fold)
        fold_number += 1

    if not predictions:
        raise RuntimeError("No walk-forward predictions generated.")

    return pd.concat(predictions).sort_index()




def evaluate(predictions):
    actual = predictions["Actual"]
    results = []

    for model_name in [
        "Market_Only",
        "Three_Factor",
        "Zero_Baseline",
    ]:
        predicted = predictions[model_name]

        results.append({
            "Model": model_name,
            "Observations": len(predictions),
            "Test_R2": r2_score(actual, predicted),
            "MAE": mean_absolute_error(actual, predicted),
            "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
            "Direction_Accuracy_Pct": (
                np.mean(np.sign(actual) == np.sign(predicted)) * 100
                if model_name != "Zero_Baseline"
                else np.nan
            ),
        })

    return pd.DataFrame(results).set_index("Model")


def save_results(predictions, metrics):
    metrics.to_csv(DATA_DIR / "walk_forward_metrics.csv")

    predictions.index.name = "Date"
    predictions.to_csv(DATA_DIR / "walk_forward_predictions.csv")

    plt.figure(figsize=(13, 6))
    plt.plot(
        predictions.index,
        predictions["Actual"] * 100,
        label="Actual return",
        alpha=0.65,
        linewidth=0.8,
    )
    plt.plot(
        predictions.index,
        predictions["Three_Factor"] * 100,
        label="Three-factor forecast",
        alpha=0.85,
        linewidth=0.8,
    )
    plt.axhline(0, linestyle="--", linewidth=0.8)
    plt.title("HDFC Bank: Walk-Forward Return Forecasts")
    plt.xlabel("Date")
    plt.ylabel("Daily return (%)")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(
        DATA_DIR / "walk_forward_predictions.png", dpi=150
    )
    plt.close()




def main():
    data = load_data()
    predictions = walk_forward_predict(data)
    metrics = evaluate(predictions)

    print("=" * 65)
    print("WALK-FORWARD VALIDATION")
    print("=" * 65)
    print(f"Training begins with {MIN_TRAIN_SIZE} observations.")
    print(f"Test block size: {TEST_BLOCK_SIZE} observations.")
    print(f"Test observations: {len(predictions)}")
    print(
        f"Test period: {predictions.index.min().date()} to "
        f"{predictions.index.max().date()}"
    )
    print(f"Refit folds: {predictions['Fold'].nunique()}")

    print("\nOut-of-sample results:")
    print(metrics.round(5).to_string())

    save_results(predictions, metrics)

    print("\nSaved:")
    print("- data/processed/walk_forward_metrics.csv")
    print("- data/processed/walk_forward_predictions.csv")
    print("- data/processed/walk_forward_predictions.png")


if __name__ == "__main__":
    main()
