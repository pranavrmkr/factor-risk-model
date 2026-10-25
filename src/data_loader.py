
from pathlib import Path

import pandas as pd
import yfinance as yf



PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

TICKERS = [
    "RELIANCE.NS",
    "TCS.NS",
    "INFY.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "^NSEI",
]


def download_prices(
    tickers: list[str] = TICKERS,
    period: str = "5y",
) -> pd.DataFrame:
    """Download adjusted historical closing prices."""

    print(f"Downloading {len(tickers)} symbols...")

    data = yf.download(
        tickers=tickers,
        period=period,
        interval="1d",
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=True,
    )

    if data.empty:
        raise RuntimeError(
            "No data was downloaded. Check your internet connection "
            "and the Yahoo Finance symbols."
        )

    # Recent yfinance versions return MultiIndex columns for
    # multiple tickers. Extract the adjusted closing prices.
    if isinstance(data.columns, pd.MultiIndex):
        if "Close" not in data.columns.get_level_values(0):
            raise RuntimeError("Closing prices were not returned.")

        prices = data["Close"].copy()
    else:
        # Handle a single-ticker response.
        if "Close" not in data.columns:
            raise RuntimeError("Closing prices were not returned.")

        prices = data[["Close"]].copy()
        prices.columns = tickers[:1]

    prices.index = pd.to_datetime(prices.index)
    prices.index.name = "Date"
    prices = prices.sort_index()

    # Convert any infinities to missing values.
    prices = prices.replace([float("inf"), float("-inf")], pd.NA)

    # Report data quality before removing incomplete rows.
    print("\nMissing values per symbol:")
    print(prices.isna().sum())

    print("\nDownloaded date range:")
    print(f"{prices.index.min().date()} to {prices.index.max().date()}")

    # Keep rows with at least one valid price.
    prices = prices.dropna(how="all")

    # Save the unfilled prices. Do not invent prices for missing dates.
    output_file = RAW_DATA_DIR / "stock_prices.csv"
    prices.to_csv(output_file)

    print(f"\nSaved prices to: {output_file}")
    print(f"Dataset shape: {prices.shape[0]} rows x {prices.shape[1]} columns")

    return prices


def main() -> None:
    prices = download_prices()

    print("\nFirst five rows:")
    print(prices.head())

    print("\nLast five rows:")
    print(prices.tail())

    print("\nSummary statistics:")
    print(prices.describe())


if __name__ == "__main__":
    main()
