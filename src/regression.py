
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm


PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "daily_returns.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

STOCK = "HDFCBANK.NS"
MARKET = "^NSEI"


def load_returns() -> pd.DataFrame:
    """Load daily returns produced in Stage 3."""

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Return data not found: {INPUT_FILE}\n"
            "Run 'python src/returns.py' first."
        )

    returns = pd.read_csv(
        INPUT_FILE,
        index_col="Date",
        parse_dates=True,
    )

    returns = returns.apply(pd.to_numeric, errors="coerce")
    returns = returns.sort_index()
    returns = returns.replace([np.inf, -np.inf], np.nan)

    required = [STOCK, MARKET]
    missing = [column for column in required if column not in returns.columns]

    if missing:
        raise ValueError(
            f"Missing required return columns: {missing}. "
            "Check data/raw/stock_prices.csv and rerun Stage 3."
        )

    # no need to use data which have missing values
    aligned = returns[required].dropna()

    if len(aligned) < 30:
        raise ValueError(
            f"Only {len(aligned)} aligned observations available. "
            "More data is needed for a meaningful initial analysis."
        )

    print(f"Using {len(aligned)} aligned daily observations.")
    print(
        f"Date range: {aligned.index.min().date()} "
        f"to {aligned.index.max().date()}"
    )

    return aligned


def fit_market_regression(
    returns: pd.DataFrame,
) -> tuple[object, pd.DataFrame]:
    #relate stock returns with funcitons

    y = returns[STOCK]
    X = sm.add_constant(returns[MARKET])

    model = sm.OLS(y, X).fit()

    results = pd.DataFrame(
        {
            "Metric": [
                "Observations",
                "Alpha (daily, decimal)",
                "Alpha (daily, percentage points)",
                "Market beta",
                "R-squared",
                "Adjusted R-squared",
                "Beta p-value",
                "Residual standard deviation (daily, decimal)",
            ],
            "Value": [
                int(model.nobs),
                model.params["const"],
                model.params["const"] * 100,
                model.params[MARKET],
                model.rsquared,
                model.rsquared_adj,
                model.pvalues[MARKET],
                np.sqrt(model.mse_resid),
            ],
        }
    )

    fitted = pd.DataFrame(
        {
            "Actual Stock Return": y,
            "Market Return": returns[MARKET],
            "Predicted Stock Return": model.predict(X),
            "Residual": model.resid,
        },
        index=returns.index,
    )

    return model, results, fitted


def save_regression_chart(
    returns: pd.DataFrame,
    model: object,
) -> None:
    

    x = returns[MARKET].to_numpy()
    y = returns[STOCK].to_numpy()

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.scatter(
        x * 100,
        y * 100,
        alpha=0.45,
        s=16,
        label="Observed daily returns",
    )

    line_x = np.linspace(x.min(), x.max(), 200)
    line_y = model.params["const"] + model.params[MARKET] * line_x

    ax.plot(
        line_x * 100,
        line_y * 100,
        linewidth=2,
        label="Fitted regression line",
    )

    ax.set_title("HDFC Bank vs Nifty 50: Market Regression")
    ax.set_xlabel("Nifty 50 daily return (%)")
    ax.set_ylabel("HDFC Bank daily return (%)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()

    output_file = OUTPUT_DIR / "market_regression.png"
    fig.savefig(output_file, dpi=150)
    plt.close(fig)

    print(f"Saved regression chart: {output_file}")


def main() -> None:
    returns = load_returns()

    model, results, fitted = fit_market_regression(returns)

    results.to_csv(
        OUTPUT_DIR / "regression_summary.csv",
        index=False,
        float_format="%.10f",
    )

    fitted.to_csv(
        OUTPUT_DIR / "regression_predictions.csv",
        float_format="%.10f",
    )

    save_regression_chart(returns, model)

    print("\nRegression Summary")
    print(model.summary())

    print("\nKey Results")
    print(results.to_string(index=False))

    print(f"\nOutputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
