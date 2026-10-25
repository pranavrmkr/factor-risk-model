
from pathlib import Path

import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt




PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = DATA_DIR

RETURNS_FILE = DATA_DIR / "daily_returns.csv"
FACTORS_FILE = DATA_DIR / "factor_returns.csv"

TARGET_STOCK = "HDFCBANK.NS"



def load_data():
    if not RETURNS_FILE.exists():
        raise FileNotFoundError(
            f"Missing {RETURNS_FILE}. Run src/returns.py first."
        )

    if not FACTORS_FILE.exists():
        raise FileNotFoundError(
            f"Missing {FACTORS_FILE}. Run src/factor_builder.py first."
        )

    stock_returns = pd.read_csv(
        RETURNS_FILE,
        index_col="Date",
        parse_dates=True
    )

    factor_returns = pd.read_csv(
        FACTORS_FILE,
        index_col="Date",
        parse_dates=True
    )

    if TARGET_STOCK not in stock_returns.columns:
        raise ValueError(
            f"{TARGET_STOCK} was not found in daily_returns.csv. "
            f"Available columns: {list(stock_returns.columns)}"
        )

    required_factors = ["Market", "Momentum", "Value"]

    missing_factors = [
        factor for factor in required_factors
        if factor not in factor_returns.columns
    ]

    if missing_factors:
        raise ValueError(
            f"Missing factors: {missing_factors}. "
            f"Available columns: {list(factor_returns.columns)}"
        )

    return stock_returns, factor_returns


def prepare_data(stock_returns, factor_returns):
    stock = stock_returns[TARGET_STOCK].rename("Stock")

    factors = factor_returns[
        ["Market", "Momentum", "Value"]
    ]

    # Match observations by date.
    data = pd.concat([stock, factors], axis=1, join="inner")

    # Remove rows containing missing or infinite values.
    data = data.replace([float("inf"), float("-inf")], float("nan"))
    data = data.dropna()

    if len(data) < 100:
        raise ValueError(
            f"Only {len(data)} aligned observations found. "
            "Check the dates and missing values in both CSV files."
        )

    if data["Stock"].std() == 0:
        raise ValueError("The target stock returns have zero variation.")

    if data[["Market", "Momentum", "Value"]].std().eq(0).any():
        raise ValueError("At least one factor has zero variation.")

    return data



def fit_models(data):
    y = data["Stock"]

    # Model A: market factor only.
    X_market = sm.add_constant(data[["Market"]])
    market_model = sm.OLS(y, X_market).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": 5}
    )

    # Model B: all four factors.
    factor_columns = ["Market", "Momentum", "Value"]
    X_factors = sm.add_constant(data[factor_columns])

    multi_factor_model = sm.OLS(y, X_factors).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": 5}
    )

    return market_model, multi_factor_model



def save_results(data, market_model, multi_factor_model):
    factor_columns = ["Market", "Momentum", "Value", "Size"]

    # Save factor coefficients and statistical significance.
    coefficients = pd.DataFrame({
        "Coefficient": multi_factor_model.params,
        "Robust_Std_Error": multi_factor_model.bse,
        "P_Value": multi_factor_model.pvalues,
        "CI_Lower": multi_factor_model.conf_int()[0],
        "CI_Upper": multi_factor_model.conf_int()[1],
    })

    coefficients.index.name = "Term"
    coefficients.to_csv(OUTPUT_DIR / "multi_factor_coefficients.csv")

    # Save overall model metrics.
    metrics = pd.DataFrame([
        {
            "Model": "Market Only",
            "Observations": int(market_model.nobs),
            "R_Squared": market_model.rsquared,
            "Adjusted_R_Squared": market_model.rsquared_adj,
            "AIC": market_model.aic,
            "BIC": market_model.bic,
        },
        {
            "Model": "Four Factor",
            "Observations": int(multi_factor_model.nobs),
            "R_Squared": multi_factor_model.rsquared,
            "Adjusted_R_Squared": multi_factor_model.rsquared_adj,
            "AIC": multi_factor_model.aic,
            "BIC": multi_factor_model.bic,
        },
    ])

    metrics.to_csv(
        OUTPUT_DIR / "model_comparison.csv",
        index=False
    )

    # Save actual returns, fitted returns and residuals.
    predictions = pd.DataFrame({
        "Actual_Return": data["Stock"],
        "Predicted_Return": multi_factor_model.fittedvalues,
        "Residual": multi_factor_model.resid,
    })

    predictions.index.name = "Date"
    predictions.to_csv(
        OUTPUT_DIR / "multi_factor_predictions.csv"
    )

    #plot comparing actual and fitted results

    plt.figure(figsize=(12, 6))
    plt.plot(
        predictions.index,
        predictions["Actual_Return"] * 100,
        label="Actual HDFC return",
        alpha=0.65,
        linewidth=0.8
    )
    plt.plot(
        predictions.index,
        predictions["Predicted_Return"] * 100,
        label="Model fitted return",
        alpha=0.85,
        linewidth=0.9
    )

    plt.title("HDFC Bank: Actual vs Multi-Factor Fitted Returns")
    plt.xlabel("Date")
    plt.ylabel("Daily return (%)")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "multi_factor_predictions.png",
        dpi=150
    )
    plt.close()

    #plot residuals 
    plt.figure(figsize=(12, 5))
    plt.scatter(
        predictions.index,
        predictions["Residual"] * 100,
        s=8,
        alpha=0.6
    )
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.title("HDFC Bank: Multi-Factor Regression Residuals")
    plt.xlabel("Date")
    plt.ylabel("Residual return (%)")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "multi_factor_residuals.png",
        dpi=150
    )
    plt.close()



def main():
    print("Loading stock returns and factor returns...")
    stock_returns, factor_returns = load_data()

    data = prepare_data(stock_returns, factor_returns)

    print(f"Target stock: {TARGET_STOCK}")
    print(f"Aligned observations: {len(data)}")
    print(f"Date range: {data.index.min().date()} to "
          f"{data.index.max().date()}")

    market_model, multi_factor_model = fit_models(data)

    print("\n" + "=" * 55)
    print("MARKET-ONLY MODEL")
    print("=" * 55)
    print(f"R-squared: {market_model.rsquared:.4f}")
    print(f"Adjusted R-squared: {market_model.rsquared_adj:.4f}")

    print("\n" + "=" * 55)
    print("FOUR-FACTOR MODEL")
    print("=" * 55)
    print(f"R-squared: {multi_factor_model.rsquared:.4f}")
    print(
        f"Adjusted R-squared: "
        f"{multi_factor_model.rsquared_adj:.4f}"
    )

    print("\nFactor coefficients:")
    print(multi_factor_model.params.round(6))

    print("\nRobust p-values (HAC standard errors):")
    print(multi_factor_model.pvalues.round(6))

    print("\nSaving results...")
    save_results(data, market_model, multi_factor_model)

    print("\nFiles saved in data/processed/:")
    print("- multi_factor_coefficients.csv")
    print("- model_comparison.csv")
    print("- multi_factor_predictions.csv")
    print("- multi_factor_predictions.png")
    print("- multi_factor_residuals.png")

    print("\nStage 7 complete.")


if __name__ == "__main__":
    main()
