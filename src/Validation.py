
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
TRAIN_RATIO = 0.80




def load_data():
    if not RETURNS_FILE.exists():
        raise FileNotFoundError(f"Missing file: {RETURNS_FILE}")

    if not FACTORS_FILE.exists():
        raise FileNotFoundError(f"Missing file: {FACTORS_FILE}")

    stock_returns = pd.read_csv(
        RETURNS_FILE, index_col="Date", parse_dates=True
    )
    factor_returns = pd.read_csv(
        FACTORS_FILE, index_col="Date", parse_dates=True
    )

    if TARGET not in stock_returns.columns:
        raise ValueError(f"{TARGET} not found in daily_returns.csv")

    missing = [f for f in FACTORS if f not in factor_returns.columns]
    if missing:
        raise ValueError(f"Missing factor columns: {missing}")

    stock = stock_returns[TARGET].rename("Actual")
    factors = factor_returns[FACTORS]

    data = pd.concat([stock, factors], axis=1, join="inner")
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    data = data.sort_index()

    if len(data) < 100:
        raise ValueError(
            f"Only {len(data)} aligned observations. "
            "Check the input files."
        )

    if not data.index.is_unique:
        raise ValueError("Duplicate dates found in the aligned data.")

    return data



def split_data(data):
    split_index = int(len(data) * TRAIN_RATIO)

    train = data.iloc[:split_index].copy()
    test = data.iloc[split_index:].copy()

    if len(train) < 30 or len(test) < 10:
        raise ValueError("Not enough observations for the split.")

    return train, test




def fit_models(train):
    y_train = train["Actual"]

    # Model A: market-only regression
    X_market_train = sm.add_constant(train[["Market"]])
    market_model = sm.OLS(y_train, X_market_train).fit()

    # Model B: three-factor regression
    X_factor_train = sm.add_constant(train[FACTORS])
    factor_model = sm.OLS(y_train, X_factor_train).fit()

    return market_model, factor_model




def calculate_metrics(actual, predicted):
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)

    return {
        "Test_R2": r2_score(actual, predicted),
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
        "Direction_Accuracy_Pct": (
            np.mean(np.sign(actual) == np.sign(predicted)) * 100
        ),
    }


def evaluate_models(train, test, market_model, factor_model):
    # These predictions use coefficients fitted on training data.
    X_market_test = sm.add_constant(test[["Market"]], has_constant="add")
    X_factor_test = sm.add_constant(test[FACTORS], has_constant="add")

    predictions = pd.DataFrame(index=test.index)
    predictions["Actual"] = test["Actual"]
    predictions["Market_Only"] = market_model.predict(X_market_test)
    predictions["Three_Factor"] = factor_model.predict(X_factor_test)

    # Simple baseline: predict zero return every day.
    predictions["Zero_Baseline"] = 0.0

    results = []

    for model_name in [
        "Market_Only",
        "Three_Factor",
        "Zero_Baseline",
    ]:
        metrics = calculate_metrics(
            predictions["Actual"],
            predictions[model_name],
        )
        metrics["Model"] = model_name
        results.append(metrics)

    metrics_df = pd.DataFrame(results).set_index("Model")

    return predictions, metrics_df



def save_outputs(predictions, metrics):
    metrics.to_csv(DATA_DIR / "validation_metrics.csv")

    predictions.index.name = "Date"
    predictions.to_csv(DATA_DIR / "validation_predictions.csv")

    plt.figure(figsize=(12, 6))
    plt.plot(
        predictions.index,
        predictions["Actual"] * 100,
        label="Actual HDFC return",
        alpha=0.65,
        linewidth=0.8,
    )
    plt.plot(
        predictions.index,
        predictions["Three_Factor"] * 100,
        label="Three-factor prediction",
        alpha=0.85,
        linewidth=0.9,
    )
    plt.axhline(0, linestyle="--", linewidth=0.8)
    plt.title("Out-of-Sample HDFC Bank Return Predictions")
    plt.xlabel("Date")
    plt.ylabel("Daily return (%)")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(DATA_DIR / "validation_predictions.png", dpi=150)
    plt.close()


def main():
    data = load_data()
    train, test = split_data(data)

    print("OUT-OF-SAMPLE VALIDATION")
    print("-" * 50)
    print(f"Total observations: {len(data)}")
    print(f"Training observations: {len(train)}")
    print(f"Testing observations: {len(test)}")
    print(f"Training period: {train.index.min().date()} to "
          f"{train.index.max().date()}")
    print(f"Testing period: {test.index.min().date()} to "
          f"{test.index.max().date()}")

    market_model, factor_model = fit_models(train)

    print("\nTraining coefficients — three-factor model:")
    print(factor_model.params.round(6))

    predictions, metrics = evaluate_models(
        train, test, market_model, factor_model
    )

    print("\nTest-set metrics:")
    print(metrics.round(5).to_string())

    save_outputs(predictions, metrics)

    print("\nSaved:")
    print("- data/processed/validation_metrics.csv")
    print("- data/processed/validation_predictions.csv")
    print("- data/processed/validation_predictions.png")


if __name__ == "__main__":
    main()
