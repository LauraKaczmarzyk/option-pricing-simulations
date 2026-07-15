"""
Section 4 - Delta hedging and the Black-Scholes model.

Builds a discretely-rebalanced replicating portfolio (stock + cash) for a
European put under GBM-with-dividend-yield dynamics and compares its
terminal value to the theoretical Black-Scholes price and to the option's
payoff. See the PDE derivation and the discrete hedging recursion in the
README for the notation (A = shares held, D = cash account, Pi = A*S + D).
"""
import math
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "q4")
os.makedirs(RESULTS_DIR, exist_ok=True)


def bs_put_price_delta(S, K, r, q, sigma, tau):
    S = np.asarray(S, dtype=float)
    tau = np.asarray(tau, dtype=float)
    tau_ = np.maximum(tau, 1e-12)

    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * tau_) / (sigma * np.sqrt(tau_))
    d2 = d1 - sigma * np.sqrt(tau_)

    disc_r = np.exp(-r * tau_)
    disc_q = np.exp(-q * tau_)

    price = K * disc_r * norm.cdf(-d2) - S * disc_q * norm.cdf(-d1)
    delta = -disc_q * norm.cdf(-d1)

    price = np.where(tau == 0.0, np.maximum(K - S, 0.0), price)
    delta = np.where(tau == 0.0, np.where(S < K, -1.0, 0.0), delta)
    return price, delta


def simulate_gbm_dividend(S0, mu, sigma, q, T, dt, seed=None):
    rng = np.random.default_rng(seed)
    n = int(round(T / dt))
    S = np.empty(n + 1, dtype=float)
    S[0] = S0
    sqrt_dt = math.sqrt(dt)

    for i in range(n):
        Z = rng.standard_normal()
        dS = (mu - q) * S[i] * dt + sigma * S[i] * sqrt_dt * Z
        S[i + 1] = S[i] + dS

    return S


def hedge_put_path(S, K, r, q, sigma, T, dt):
    n = len(S) - 1
    t = np.linspace(0.0, T, n + 1)
    tau = T - t

    V, Delta = bs_put_price_delta(S, K, r, q, sigma, tau)

    A = np.empty(n + 1)
    D = np.empty(n + 1)
    Pi = np.empty(n + 1)

    A[0] = Delta[0]
    D[0] = V[0] - A[0] * S[0]
    Pi[0] = A[0] * S[0] + D[0]

    for i in range(n):
        Pi_temp = A[i] * S[i + 1] + (1.0 + r * dt) * D[i] + q * A[i] * S[i] * dt
        A[i + 1] = Delta[i + 1]  # rebalance to the new delta
        D[i + 1] = Pi_temp - A[i + 1] * S[i + 1]
        Pi[i + 1] = A[i + 1] * S[i + 1] + D[i + 1]

    return {"t": t, "S": S, "V": V, "Delta": Delta, "A": A, "D": D, "Pi": Pi}


def plot_hedge(res, K, title, filename):
    t, S, Delta, D, Pi, V = res["t"], res["S"], res["Delta"], res["D"], res["Pi"], res["V"]

    fig, axs = plt.subplots(4, 1, figsize=(9, 9), sharex=True)
    axs[0].plot(t, S, label="S(t)")
    axs[0].axhline(K, ls="--", label="K")
    axs[0].set_ylabel("Asset path")
    axs[0].set_title(title)
    axs[0].legend()

    axs[1].plot(t, Delta)
    axs[1].set_ylabel("Delta")

    axs[2].plot(t, D)
    axs[2].set_ylabel("Cash")

    axs[3].plot(t, Pi, label="Portfolio")
    axs[3].plot(t, V, "o", ms=4, label="BS value")
    axs[3].set_ylabel("Portfolio")
    axs[3].set_xlabel("Time")
    axs[3].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, filename))
    plt.close()


def eq99_value_at_T(res, r, P0):
    """Total hedged-portfolio value at maturity: replicating portfolio plus
    the risk-free growth of the initial pricing/hedging mismatch (Eq. 9.9)."""
    Pi_T = float(res["Pi"][-1])
    Pi_0 = float(res["Pi"][0])
    T = float(res["t"][-1])
    return Pi_T + (P0 - Pi_0) * np.exp(r * T)


def simulate_eq99_many_paths(M, S0, K, r, q, sigma, T, dt, mu):
    P0, _ = bs_put_price_delta(S0, K, r, q, sigma, T)

    S_T = np.empty(M)
    Y_T = np.empty(M)
    Pay = np.empty(M)
    Err = np.empty(M)

    for m in range(M):
        S_path = simulate_gbm_dividend(S0, mu, sigma, q, T, dt)
        res = hedge_put_path(S_path, K, r, q, sigma, T, dt)

        S_T[m] = float(S_path[-1])
        Pay[m] = max(K - S_T[m], 0.0)
        Y_T[m] = eq99_value_at_T(res, r, P0)
        Err[m] = Y_T[m] - Pay[m]

    return {"S_T": S_T, "Y_T": Y_T, "Payoff": Pay, "Error": Err, "P0": float(P0)}


def run_part_c(S0=170.0, K=170.0, T=1.0, r=0.05, q=0.01, sigma=0.344, dt=0.01, M=1000, mus=(0.10, -0.10)):
    """Confirms the hedged portfolio's terminal value tracks the payoff
    independently of the real-world drift mu used to simulate S."""
    results = {}
    for mu in mus:
        data = simulate_eq99_many_paths(M, S0, K, r, q, sigma, T, dt, mu)
        results[mu] = data

        S_T, Y_T = data["S_T"], data["Y_T"]

        plt.figure(figsize=(7.5, 5.2))
        plt.plot(S_T, Y_T, ".", ms=4, alpha=0.5, color="black", label="Eq. (9.9) points")

        x = np.linspace(S_T.min() * 0.9, S_T.max() * 1.1, 300)
        plt.plot(x, np.maximum(K - x, 0.0), "k--", label="Put payoff  max(K - S_T, 0)")

        plt.xlabel(r"$S_T$")
        plt.ylabel(r"$\Pi(S_T,T) + (P(S_0,0)-\Pi(S_0,0))\,e^{rT}$")
        plt.title(f"Eq. (9.9) vs Payoff -- PUT (mu={mu:+.2f})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f"eq99_vs_payoff_mu_{mu:+.2f}.png"))
        plt.close()

    if len(mus) >= 2:
        print("\nDrift dependence (mean / stdev of hedge error):")
        for mu in mus:
            err = results[mu]["Error"]
            print(f"  mu={mu:+.2f} : mean={err.mean():.6f}  std={err.std(ddof=1):.6f}")

    return results


if __name__ == "__main__":
    K = 170.0
    T = 1.0
    r = 0.05
    dt = 0.01
    sigma = 0.344
    mu = 0.10
    q = 0.01

    S_itm = simulate_gbm_dividend(150.0, mu, sigma, q, T, dt, seed=42)  # spot below K -> put ITM
    res_itm = hedge_put_path(S_itm, K, r, q, sigma, T, dt)
    plot_hedge(res_itm, K, "Discrete hedging (PUT, ITM)", "hedge_itm.png")

    S_otm = simulate_gbm_dividend(190.0, mu, sigma, q, T, dt, seed=42)  # spot above K -> put OTM
    res_otm = hedge_put_path(S_otm, K, r, q, sigma, T, dt)
    plot_hedge(res_otm, K, "Discrete hedging (PUT, OTM)", "hedge_otm.png")

    print("\n=== Question (c): Eq. (9.9) large-scale test ===")
    run_part_c(S0=170.0, K=K, T=T, r=r, q=q, sigma=sigma, dt=dt, M=1000, mus=(0.10, -0.10))
