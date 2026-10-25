
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "daily_returns.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Only the stocks are PCA inputs. The nifty 50 remains as a benchmark
STOCKS = [
    "RELIANCE.NS",
    "TCS.NS",
    "INFY.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS",
]


def load_stock_returns() -> pd.DataFrame:
    """Load and validate the stock return matrix."""

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

    missing = [stock for stock in STOCKS if stock not in returns.columns]

    if missing:
        raise ValueError(
            f"Missing stock columns: {missing}. "
            "Check your downloaded price data."
        )

    stock_returns = returns[STOCKS].apply(
        pd.to_numeric,
        errors="coerce",
    )

    stock_returns = stock_returns.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    #remove any listings with improper data
    stock_returns = stock_returns.dropna(how="any")

    if len(stock_returns) < 30:
        raise ValueError(
            f"Only {len(stock_returns)} complete observations. "
            "At least 30 are required for this initial analysis."
        )

    if (stock_returns.std() == 0).any():
        raise ValueError(
            "At least one stock has zero return variance. "
            "Check the underlying data."
        )

    print(f"Using {len(stock_returns)} complete observations.")
    print(
        f"Date range: {stock_returns.index.min().date()} "
        f"to {stock_returns.index.max().date()}"
    )

    return stock_returns


def run_pca(
    returns: pd.DataFrame,
) -> tuple[PCA, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit PCA to standardized stock returns."""

    # standardize each stock so its scale does not screw up PCA
    scaler = StandardScaler()
    standardized = scaler.fit_transform(returns)

    
    pca = PCA()
    scores_array = pca.fit_transform(standardized)

    component_names = [
        f"PC{i + 1}" for i in range(pca.n_components_)
    ]

    # fraction of total standardized variance captured by each PC.
    variance_summary = pd.DataFrame(
        {
            "Component": component_names,
            "Explained Variance Ratio": pca.explained_variance_ratio_,
            "Explained Variance (%)": (
                pca.explained_variance_ratio_ * 100
            ),
            "Cumulative Variance (%)": (
                np.cumsum(pca.explained_variance_ratio_) * 100
            ),
        }
    )

    # this decribes how each stock contributes to each component.
    loadings = pd.DataFrame(
        pca.components_.T,
        index=returns.columns,
        columns=component_names,
    )
    loadings.index.name = "Stock"

    # Scores describe each date's position along each component.
    scores = pd.DataFrame(
        scores_array,
        index=returns.index,
        columns=component_names,
    )
    scores.index.name = "Date"

    return pca, variance_summary, loadings, scores


def save_explained_variance_chart(
    variance_summary: pd.DataFrame,
) -> None:
    """Plot individual and cumulative explained variance."""

    components = variance_summary["Component"]
    individual = variance_summary["Explained Variance (%)"]
    cumulative = variance_summary["Cumulative Variance (%)"]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(
        components,
        individual,
        alpha=0.8,
        label="Individual explained variance",
    )

    ax.plot(
        components,
        cumulative,
        marker="o",
        linewidth=2,
        label="Cumulative explained variance",
    )

    ax.set_title("PCA: Explained Variance of Stock Returns")
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Explained variance (%)")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()

    fig.tight_layout()

    output_file = OUTPUT_DIR / "pca_explained_variance.png"
    fig.savefig(output_file, dpi=150)
    plt.close(fig)

    print(f"Saved chart: {output_file}")


def save_loadings_chart(loadings: pd.DataFrame) -> None:
    """Plot stock loadings for each principal component."""

    fig, ax = plt.subplots(figsize=(11, 6))

    loadings.plot(
        kind="bar",
        ax=ax,
        width=0.8,
    )

    ax.set_title("PCA Component Loadings by Stock")
    ax.set_xlabel("Stock")
    ax.set_ylabel("Component loading")
    ax.axhline(0, linewidth=0.8)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(title="Component")
    plt.xticks(rotation=35, ha="right")

    fig.tight_layout()

    output_file = OUTPUT_DIR / "pca_loadings.png"
    fig.savefig(output_file, dpi=150)
    plt.close(fig)

    print(f"Saved chart: {output_file}")


def main() -> None:
    returns = load_stock_returns()

    pca, variance_summary, loadings, scores = run_pca(returns)

    variance_summary.to_csv(
        OUTPUT_DIR / "pca_variance.csv",
        index=False,
        float_format="%.8f",
    )

    loadings.to_csv(
        OUTPUT_DIR / "pca_loadings.csv",
        float_format="%.8f",
    )

    scores.to_csv(
        OUTPUT_DIR / "pca_scores.csv",
        float_format="%.8f",
    )

    save_explained_variance_chart(variance_summary)
    save_loadings_chart(loadings)

    print("\n--- Explained Variance ---")
    print(variance_summary.to_string(index=False))

    print("\n--- Component Loadings ---")
    print(loadings.round(4).to_string())

    print(
        "\nComponents needed to explain at least 80% of variance:",
        int(
            np.searchsorted(
                np.cumsum(pca.explained_variance_ratio_),
                0.80,
            ) + 1
        ),
    )

    print(f"\nPCA analysis complete. Outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
