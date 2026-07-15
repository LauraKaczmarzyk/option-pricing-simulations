"""
Section 1 - Market stock prices.

Loads AAPL daily/weekly close prices, computes simple and logarithmic
returns, and checks the returns for normality via a histogram and a
QQ-plot against N(0, 1).

    r_i      = (S_i - S_{i-1}) / S_i          simple return
    r_hat_i  = ln(S_i / S_{i-1})               log return

For small r_i, ln(1 + r_i) ~= r_i, so the two measures only diverge
materially for large single-period moves.

Run with:
    python src/fetch_data.py   # once, to populate data/*_aapl.csv
    python src/returns_analysis.py
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "q1")
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_prices(csv_name: str) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(DATA_DIR, csv_name))
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def compute_returns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["simple_return"] = df["Close"].pct_change()
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
    return df


def plot_price_movements(df: pd.DataFrame, label: str) -> None:
    plt.figure(figsize=(10, 6))
    plt.plot(df["Date"], df["Close"], label="Close Price", linewidth=2)
    plt.title(f"{label} Price Movements Over Time")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"{label.lower()}_price_movement.png"))
    plt.close()


def plot_returns(df: pd.DataFrame, label: str) -> None:
    plt.figure(figsize=(12, 5))
    plt.plot(df["Date"], df["simple_return"], color="blue", label="Simple Return")
    plt.title(f"{label} Simple Returns Over Time")
    plt.xlabel("Date")
    plt.ylabel("Return")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"{label.lower()}_simple_return.png"))
    plt.close()

    plt.figure(figsize=(12, 5))
    plt.plot(df["Date"], df["log_return"], color="orange", label="Log Return")
    plt.title(f"{label} Logarithmic Returns Over Time")
    plt.xlabel("Date")
    plt.ylabel("Log Return")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"{label.lower()}_log_return.png"))
    plt.close()


def log_return_stats(df: pd.DataFrame, label: str, bins: int = 50) -> tuple[float, float]:
    log_returns = df["log_return"].dropna()

    mu_hat = log_returns.mean()
    sigma2_hat = log_returns.var(ddof=1)  # ddof=1 -> denominator (n - 1), unbiased sample variance
    print(f"[{label}] mean (mu_hat):      {mu_hat:.8f}")
    print(f"[{label}] variance (sigma^2): {sigma2_hat:.8f}")

    plt.figure(figsize=(10, 5))
    plt.hist(log_returns, bins=bins, density=True)
    plt.title(f"Histogram of {label} Logarithmic Returns")
    plt.xlabel("Log return")
    plt.ylabel("Density")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"{label.lower()}_log_return_hist.png"))
    plt.close()

    return mu_hat, sigma2_hat


def qq_plot(ax, df: pd.DataFrame, label: str) -> None:
    stats.probplot(df["log_return"].dropna(), dist="norm", plot=ax)
    ax.set_title(f"QQ-Plot ({label})")
    ax.set_xlabel("Theoretical Quantiles (N(0,1))")
    ax.set_ylabel("Log Returns")
    ax.grid(True)


def main():
    daily = compute_returns(load_prices("daily_aapl.csv"))
    weekly = compute_returns(load_prices("weekly_aapl.csv"))

    plot_price_movements(daily, "Daily")
    plot_returns(daily, "Daily")
    log_return_stats(daily, "Daily")
    log_return_stats(weekly, "Weekly")

    # Weekly aggregation averages away idiosyncratic daily noise, so its QQ-plot
    # should hug the N(0,1) line more closely than the daily one (CLT in action).
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    qq_plot(axes[0], daily, "Daily")
    qq_plot(axes[1], weekly, "Weekly")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "qq_daily_vs_weekly.png"))
    plt.close()


if __name__ == "__main__":
    main()
