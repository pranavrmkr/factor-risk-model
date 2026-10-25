
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PRICE_FILE = PROJECT_ROOT / "data" / "raw" / "stock_prices.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

STOCKS = [
    "RELIANCE.NS",
    "TCS.NS",
    "INFY.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS",
]

LOOKBACK_DAYS = 252
SKIP_RECENT_DAYS = 21
GROUP_SIZE = 2


def load_prices() -> pd.DataFrame:
    """Load the existing historical stock prices."""

    if not PRICE_FILE.exists():
        raise FileNotFoundError(
            "Price file not found. Run src/data_loader.py first."
        )

    prices = pd.read_csv(
        PRICE_FILE,
        index_col="Date",
        parse_dates=True,
    )

    missing = [stock for stock in STOCKS if stock not in prices.columns]
    if missing:
        raise ValueError(f"Missing stock columns: {missing}")

    prices = prices[STOCKS].apply(pd.to_numeric, errors="coerce")
    prices = prices.sort_index()
    prices = prices.replace([np.inf, -np.inf], np.nan)

    return prices


def get_latest_fundamentals() -> pd.DataFrame:
    """
    Download latest market capitalization and price-to-book ratios.

    IMPORTANT:
    These are current snapshots, not historical point-in-time values.
    Using them for past returns introduces look-ahead bias.
    """

    records = []

    for ticker in STOCKS:
        print(f"Fetching current fundamentals for {ticker}...")

        try:
            info = yf.Ticker(ticker).info

            market_cap = info.get("marketCap")
            price_to_book = info.get("priceToBook")

            if market_cap is None or price_to_book is None:
                print(
                    f"Warning: missing market cap or price-to-book "
                    f"for {ticker}."
                )

            records.append(
                {
                    "Stock": ticker,
                    "Market Cap": market_cap,
                    "Price to Book": price_to_book,
                }
            )

        except Exception as exc:
            print(f"Warning: could not fetch {ticker}: {exc}")
            records.append(
                {
                    "Stock": ticker,
                    "Market Cap": np.nan,
                    "Price to Book": np.nan,
                }
            )

    fundamentals = pd.DataFrame(records).set_index("Stock")

    fundamentals.to_csv(
        OUTPUT_DIR / "latest_fundamentals.csv"
    )

    return fundamentals


def long_short_return(
    returns: pd.DataFrame,
    long_stocks: list[str],
    short_stocks: list[str],
) -> pd.Series:
    """Calculate equal-weighted long-minus-short returns."""

    long_return = returns[long_stocks].mean(axis=1)
    short_return = returns[short_stocks].mean(axis=1)

    return long_return - short_return


def build_momentum_factor(
    prices: pd.DataFrame,
) -> pd.Series:
    """
    Rank stocks by trailing 12-month momentum, skipping the
    most recent 21 trading days.

    Rebalance monthly using only prices available at that date.
    """

    # Monthly rebalancing dates.
    rebalance_dates = prices.resample("ME").last().index

    monthly_weights = {}

    for date in rebalance_dates:
        history = prices.loc[:date]

        if len(history) <= LOOKBACK_DAYS + SKIP_RECENT_DAYS:
            continue

        # End the signal 21 trading observations before the date.
        end_index = len(history) - SKIP_RECENT_DAYS - 1
        start_index = end_index - LOOKBACK_DAYS

        if start_index < 0:
            continue

        start_prices = history.iloc[start_index]
        end_prices = history.iloc[end_index]

        momentum = end_prices / start_prices - 1
        momentum = momentum.replace([np.inf, -np.inf], np.nan).dropna()

        if len(momentum) < 2 * GROUP_SIZE:
            continue

        ranked = momentum.sort_values()

        weak = ranked.index[:GROUP_SIZE].tolist()
        strong = ranked.index[-GROUP_SIZE:].tolist()

        weights = pd.Series(0.0, index=prices.columns)
        weights[strong] = 1 / GROUP_SIZE
        weights[weak] = -1 / GROUP_SIZE

        monthly_weights[date] = weights

    if not monthly_weights:
        raise ValueError(
            "Not enough historical prices to construct momentum. "
            "Check the data range and lookback settings."
        )

    # Apply each month's weights to the following daily returns.
    daily_returns = prices.pct_change(fill_method=None)
    factor = pd.Series(np.nan, index=prices.index, name="Momentum")

    dates = list(monthly_weights.keys())

    for i, rebalance_date in enumerate(dates):
        next_date = (
            dates[i + 1]
            if i + 1 < len(dates)
            else prices.index[-1]
        )

        mask = (
            (prices.index > rebalance_date)
            & (prices.index <= next_date)
        )

        weights = monthly_weights[rebalance_date]
        factor.loc[mask] = daily_returns.loc[mask].mul(
            weights,
            axis=1,
        ).sum(axis=1, min_count=len(STOCKS))

    return factor


def build_static_fundamental_factor(
    returns: pd.DataFrame,
    fundamentals: pd.DataFrame,
    characteristic: str,
    long_is_low: bool,
    factor_name: str,
) -> pd.Series:
    """
    Build a long-short factor from the latest available characteristic.

    This is an educational historical illustration, NOT a valid
    point-in-time backtest, because the characteristic snapshot is
    current rather than historical.
    """

    values = fundamentals[characteristic].dropna()

    if len(values) < 2 * GROUP_SIZE:
        raise ValueError(
            f"Not enough valid observations for {characteristic}. "
            "At least four stocks are needed."
        )

    ranked = values.sort_values()

    if long_is_low:
        long_stocks = ranked.index[:GROUP_SIZE].tolist()
        short_stocks = ranked.index[-GROUP_SIZE:].tolist()
    else:
        long_stocks = ranked.index[-GROUP_SIZE:].tolist()
        short_stocks = ranked.index[:GROUP_SIZE].tolist()

    available = [
        stock for stock in long_stocks + short_stocks
        if stock in returns.columns
    ]

    if len(available) < 2 * GROUP_SIZE:
        raise ValueError(
            f"Insufficient return data for {factor_name}."
        )

    print(f"\n{factor_name}:")
    print(f"  Long group: {long_stocks}")
    print(f"  Short group: {short_stocks}")

    return long_short_return(
        returns,
        long_stocks,
        short_stocks,
    ).rename(factor_name)


def main() -> None:
    prices = load_prices()

    # Returns used to calculate factor performance.
    returns = prices.pct_change(fill_method=None)

    fundamentals = get_latest_fundamentals()

    momentum = build_momentum_factor(prices)

    value = build_static_fundamental_factor(
        returns=returns,
        fundamentals=fundamentals,
        characteristic="Price to Book",
        long_is_low=True,
        factor_name="Value",
    )

    size = build_static_fundamental_factor(
        returns=returns,
        fundamentals=fundamentals,
        characteristic="Market Cap",
        long_is_low=True,
        factor_name="Size",
    )

    market = yf.download(
        "^NSEI",
        period="5y",
        interval="1d",
        auto_adjust=True,
        progress=False,
    )

    if market.empty or "Close" not in market.columns:
        raise RuntimeError("Could not download Nifty 50 prices.")

    market_returns = market["Close"].pct_change(fill_method=None)
    market_returns.name = "Market"

    factors = pd.concat(
        [market_returns, momentum, value, size],
        axis=1,
    )

    factors.index.name = "Date"

    output_file = OUTPUT_DIR / "factor_returns.csv"

    factors.to_csv(
        output_file,
        float_format="%.10f",
    )

    print("\n--- Factor Returns: Latest Rows ---")
    print(factors.tail().to_string())

    print(f"\nSaved factor returns to: {output_file}")
    print(
        "\nWARNING: Value and Size use current fundamentals "
        "applied retrospectively. Do not use these results as "
        "a valid historical backtest."
    )


if __name__ == "__main__":
    main()
