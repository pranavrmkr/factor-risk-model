
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FILE = ROOT / "data" / "processed" / "factor_returns.csv"

factors = pd.read_csv(FILE, index_col="Date", parse_dates=True)
factors = factors[["Market", "Momentum", "Value", "Size"]].dropna()

print("Factor correlation matrix:")
print(factors.corr().round(4))

print("\nValue and Size relationship:")
print("Correlation:", factors["Value"].corr(factors["Size"]))
print(
    "Maximum absolute error in Value + Size:",
    (factors["Value"] + factors["Size"]).abs().max()
)

print("\nFactor standard deviations:")
print(factors.std().round(6))
