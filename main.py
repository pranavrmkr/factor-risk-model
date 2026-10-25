
import pandas as pd
import numpy as np


def main():
    # Example historical closing prices.
    # These are made-up values, not real market data.
    prices = pd.Series(
        [100, 102, 101, 105, 108],
        name="Example Stock"
    )

    # Calculate the return between consecutive prices.
    returns = prices.pct_change().dropna()

    print("Factor Risk Model")
    print("-" * 30)

    print("\nExample historical prices:")
    print(prices.to_list())

    print("\nReturns:")
    print((returns * 100).round(2).astype(str) + "%")

    print("\nAverage return per observation:")
    print(f"{returns.mean() * 100:.2f}%")

    print("\nReturn volatility (standard deviation):")
    print(f"{returns.std() * 100:.2f}%")


if __name__ == "__main__":
    main()
