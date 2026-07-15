"""
Section 5 - Implied volatility.

Backs out Black-Scholes implied volatility from a LULU call-option chain
(bid/ask mid-prices) via bisection, then plots the volatility smile
(IV vs strike) and the term structure (IV vs time to maturity).
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "q5")
os.makedirs(RESULTS_DIR, exist_ok=True)


def bs_call_price_diff(S, K, r, sigma, T, market_price):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call - market_price


def implied_volatility(price, S, K, r, T, tol=1e-6, max_iter=1000):
    """Bisection search for sigma such that BS call price == market price."""
    lo, hi = 1e-6, 3.0

    lo_value = bs_call_price_diff(S, K, r, lo, T, price)
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        mid_value = bs_call_price_diff(S, K, r, mid, T, price)
        if mid_value == 0 or abs(hi - lo) < tol:
            return mid
        if lo_value * mid_value < 0:
            hi = mid
        else:
            lo, lo_value = mid, mid_value
    return 0.5 * (lo + hi)


def main():
    df = pd.read_csv(os.path.join(DATA_DIR, "lulu_options.csv"))

    K = np.array(df["Strike"], dtype=float)
    C_market = np.array(0.5 * (df["Bid"] + df["Ask"]))

    expiry = pd.to_datetime(df["Expiry_Date"])
    current_date = pd.to_datetime("2025-10-24")
    T = np.array((expiry - current_date).dt.days / 365.0)

    S = 178.0
    r = 0.04

    imp_vol = [implied_volatility(c, S, k, r, t) for k, c, t in zip(K, C_market, T)]

    plt.figure(figsize=(8, 5))
    plt.plot(K, imp_vol, marker="o")
    plt.axvline(S, color="gray", linestyle="--", label="Current price")
    plt.title("LULU - Implied Volatility Smile")
    plt.xlabel("Strike (K)")
    plt.ylabel("Implied Volatility")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "iv_smile.png"))
    plt.close()

    plt.figure(figsize=(8, 5))
    order = np.argsort(T)
    plt.plot(T[order], np.array(imp_vol)[order], "o-", label="All Strikes")
    plt.xlabel("Time to Maturity (years)")
    plt.ylabel("Implied Volatility")
    plt.title("LULU - Term Structure")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "iv_term_structure.png"))
    plt.close()


if __name__ == "__main__":
    main()
