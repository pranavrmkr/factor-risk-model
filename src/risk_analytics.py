
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed"

PREDICTIONS_FILE = DATA_DIR / "validation_predictions.csv"

TRADING_DAYS = 252
ROLLING_WINDOW = 21
CONFIDENCE_LEVELS = [0.95, 0.99]


# --------------------------------------------------
# 2. Load out-of-sample predictions
# --------------------------------------------------

def load_data():
    if not PREDICTIONS_FILE.exists():
        raise FileNotFoundError(
            f"Missing {PREDICTIONS_FILE}. "
            "Run src/validation.py first."
        )

    data = pd.read_csv(
        PREDICTIONS_FILE,
        index_col="Date",
        parse_dates=True,
    )

    required = ["Actual", "Three_Factor"]
    missing = [col for col in required if col not in data.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    data = data.sort_index()
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=required)

    if not data.index.is_unique:
        raise ValueError("Duplicate dates found in predictions.")

    if len(data) < ROLLING_WINDOW:
        raise ValueError(
            f"Need at least {ROLLING_WINDOW} observations."
        )

    # Out-of-sample model residuals.
    data["Residual"] = data["Actual"] - data["Three_Factor"]

    return data


# --------------------------------------------------
# 3. Calculate risk statistics
# --------------------------------------------------

def calculate_risk_metrics(returns, series_name):
    returns = pd.Series(returns).dropna().astype(float)

    if len(returns) < 2:
        raise ValueError(
            f"Not enough observations for {series_name}."
        )

    if not np.isfinite(returns).all():
        raise ValueError(
            f"Non-finite returns found in {series_name}."
        )

    # Returns are decimal values, e.g. 0.01 = 1%.
    daily_volatility = returns.std(ddof=1)
    annualized_volatility = daily_volatility * np.sqrt(TRADING_DAYS)

    # Loss is the negative of return.
    losses = -returns

    metrics = {
        "Series": series_name,
        "Observations": len(returns),
        "Mean_Daily_Return_Pct": returns.mean() * 100,
        "Daily_Volatility_Pct": daily_volatility * 100,
        "Annualized_Volatility_Pct": annualized_volatility * 100,
    }

    # Historical VaR and Expected Shortfall.
    for confidence in CONFIDENCE_LEVELS:
        tail_probability = 1 - confidence

        return_quantile = returns.quantile(tail_probability)
        var_loss = -return_quantile

        # Average losses at or beyond the VaR threshold.
        tail_returns = returns[returns <= return_quantile]
        expected_shortfall = -tail_returns.mean()

        label = int(confidence * 100)

        metrics[f"VaR_{label}_Pct"] = var_loss * 100
        metrics[f"ES_{label}_Pct"] = expected_shortfall * 100

    # Maximum drawdown based on compounded returns.
    wealth = np.concatenate([
        [1.0],
        (1 + returns).cumprod().to_numpy()
    ])

    running_peak = np.maximum.accumulate(wealth)
    drawdowns = wealth / running_peak - 1

    metrics["Max_Drawdown_Pct"] = drawdowns.min() * 100

    return metrics


# --------------------------------------------------
# 4. Calculate rolling volatility
# --------------------------------------------------

def calculate_rolling_volatility(data):
    rolling_vol = (
        data["Actual"]
        .rolling(ROLLING_WINDOW)
        .std()
        * np.sqrt(TRADING_DAYS)
        * 100
    )

    rolling_vol.name = "Rolling_Annualized_Volatility_Pct"

    return rolling_vol


# --------------------------------------------------
# 5. Save results and charts
# --------------------------------------------------

def save_outputs(data, rolling_vol, metrics):
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(
        DATA_DIR / "risk_metrics.csv",
        index=False,
    )

    rolling_vol.to_csv(
        DATA_DIR / "rolling_volatility.csv",
        index_label="Date",
    )

    # Plot rolling annualized volatility.
    plt.figure(figsize=(12, 6))
    plt.plot(
        rolling_vol.index,
        rolling_vol,
        linewidth=1.2,
        label="21-day rolling annualized volatility",
    )
    plt.title("HDFC Bank: Rolling Volatility on Test Data")
    plt.xlabel("Date")
    plt.ylabel("Annualized volatility (%)")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(
        DATA_DIR / "rolling_volatility.png",
        dpi=150,
    )
    plt.close()

    # Plot the out-of-sample residuals.
    plt.figure(figsize=(12, 5))
    plt.plot(
        data.index,
        data["Residual"] * 100,
        linewidth=0.8,
        label="Three-factor residual",
    )
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.title("HDFC Bank: Out-of-Sample Model Residuals")
    plt.xlabel("Date")
    plt.ylabel("Residual return (%)")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(
        DATA_DIR / "risk_residuals.png",
        dpi=150,
    )
    plt.close()


# --------------------------------------------------
# 6. Run risk analytics
# --------------------------------------------------

def main():
    data = load_data()

    actual_returns = data["Actual"]
    residual_returns = data["Residual"]

    actual_metrics = calculate_risk_metrics(
        actual_returns,
        "HDFC Bank actual returns",
    )

    residual_metrics = calculate_risk_metrics(
        residual_returns,
        "Three-factor residuals",
    )

    rolling_vol = calculate_rolling_volatility(data)

    metrics = [actual_metrics, residual_metrics]

    save_outputs(data, rolling_vol, metrics)

    results = pd.DataFrame(metrics).set_index("Series")

    print("=" * 65)
    print("OUT-OF-SAMPLE RISK ANALYTICS")
    print("=" * 65)

    print(f"Test observations: {len(data)}")
    print(
        f"Test period: {data.index.min().date()} to "
        f"{data.index.max().date()}"
    )

    print("\nRisk metrics:")
    print(results.round(4).to_string())

    print("\nFiles saved:")
    print("- data/processed/risk_metrics.csv")
    print("- data/processed/rolling_volatility.csv")
    print("- data/processed/rolling_volatility.png")
    print("- data/processed/risk_residuals.png")


if __name__ == "__main__":
    main()
