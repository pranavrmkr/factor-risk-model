
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_FILE = PROJECT_ROOT / "data" / "raw" / "stock_prices.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_prices() -> pd.DataFrame:
    #loads the prices 

    if not RAW_DATA_FILE.exists():
        raise FileNotFoundError(
            f"Price data not found: {RAW_DATA_FILE}\n"
            "Run 'python src/data_loader.py' first."
        )

    prices = pd.read_csv(
        RAW_DATA_FILE,
        index_col="Date",
        parse_dates=True,
    )

    prices = prices.apply(pd.to_numeric, errors="coerce")
    prices = prices.sort_index()
    prices = prices[~prices.index.duplicated(keep="last")]

    # Remove unncessary columns .
    prices = prices.dropna(axis=1, how="all")

    if prices.empty:
        raise ValueError("The price dataset contains no usable data.")

    if (prices.dropna() <= 0).any().any():
        raise ValueError("Prices must be positive.")

    print(f"Loaded {len(prices)} dates and {len(prices.columns)} symbols.")

    return prices


def calculate_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Calculate simple daily percentage returns."""

    
    returns = prices.pct_change(fill_method=None)

    # An unavailable price means that return cannot be calculated.
    returns = returns.replace([np.inf, -np.inf], np.nan)

    return returns


def calculate_statistics(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate descriptive statistics for each symbol."""

    statistics = pd.DataFrame(index=prices.columns)

    statistics["First Price"] = prices.ffill().iloc[0]
    statistics["Last Price"] = prices.ffill().iloc[-1]

    # Total change across the available price history.
    statistics["Total Return (%)"] = (
        (prices.ffill().iloc[-1] / prices.ffill().iloc[0] - 1)
        * 100
    )

    statistics["Mean Daily Return (%)"] = returns.mean() * 100
    statistics["Daily Volatility (%)"] = returns.std() * 100

    # Annualization assumes daily observations on approximately
    # 252 trading days per year.
    statistics["Annualized Volatility (%)"] = (
        returns.std() * np.sqrt(252) * 100
    )

    statistics["Positive Days (%)"] = (
        (returns > 0).sum()
        / returns.notna().sum().replace(0, np.nan)
        * 100
    )

    statistics["Observations"] = returns.count()

    return statistics.round(3)


def save_return_chart(returns: pd.DataFrame) -> None:
    """Save a cumulative growth chart for the available symbols."""

    # This chart shows the compounded growth of one unit invested
    # at the beginning of the available return history.
    growth = (1 + returns.fillna(0)).cumprod()

    ax = growth.plot(figsize=(12, 6), linewidth=1.2)

    ax.set_title("Cumulative Growth from Daily Returns")
    ax.set_xlabel("Date")
    ax.set_ylabel("Growth of ₹1 (before costs)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)

    plt.tight_layout()

    output_file = OUTPUT_DIR / "cumulative_returns.png"
    plt.savefig(output_file, dpi=150)
    plt.close()

    print(f"Saved chart: {output_file}")


def save_correlation_chart(returns: pd.DataFrame) -> None:
    """Save a correlation heatmap."""

    # Drop symbols without enough return observations.
    usable = returns.dropna(axis=1, thresh=30)

    if usable.shape[1] < 2:
        print("Not enough usable symbols for a correlation heatmap.")
        return

    correlation = usable.corr()

    fig, ax = plt.subplots(figsize=(9, 7))

    image = ax.imshow(
        correlation,
        vmin=-1,
        vmax=1,
        aspect="auto",
        cmap="coolwarm",
    )

    ax.set_xticks(range(len(correlation.columns)))
    ax.set_yticks(range(len(correlation.index)))

    ax.set_xticklabels(
        correlation.columns,
        rotation=45,
        ha="right",
    )
    ax.set_yticklabels(correlation.index)

    ax.set_title("Daily Return Correlation Matrix")

    fig.colorbar(image, ax=ax, label="Correlation")
    fig.tight_layout()

    output_file = OUTPUT_DIR / "correlation_heatmap.png"
    fig.savefig(output_file, dpi=150)
    plt.close(fig)

    print(f"Saved chart: {output_file}")


def main() -> None:
    prices = load_prices()

    returns = calculate_returns(prices)

    statistics = calculate_statistics(prices, returns)

    # Save analysis outputs.
    returns.to_csv(OUTPUT_DIR / "daily_returns.csv")
    statistics.to_csv(OUTPUT_DIR / "risk_summary.csv")

    save_return_chart(returns)
    save_correlation_chart(returns)

    print("\n--- Risk Summary ---")
    print(statistics.to_string())

    print("\n--- Latest Daily Returns (%) ---")
    print((returns.tail() * 100).round(3).to_string())

    print(f"\nAnalysis complete. Outputs saved to: {OUTPUT_DIR}")


    #hdfc bank shows 0% return so to check 
    print("\nHDFC Bank diagnostic:")
    print("Raw daily returns:")
    print(returns["HDFCBANK.NS"].dropna().head(10).to_string())

    print("\nUnrounded mean daily return:")
    print(f"{returns['HDFCBANK.NS'].mean():.10f}")

    print("\nMean daily return as percentage:")
    print(f"{returns['HDFCBANK.NS'].mean() * 100:.8f}%")

    print("\nNumber of valid observations:")
    print(returns["HDFCBANK.NS"].count())


if __name__ == "__main__":
    main()
