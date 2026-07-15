"""
Homework II, Question 2 - Monte Carlo pricing of exotic options.

Adjusts the standard Monte Carlo option-pricing approach (simulate many
price paths, average the discounted payoff) for path-dependent payoffs:

    Asian put:     v = e^(-rT) * max(E - S_avg, 0)
    Lookback put:  v = e^(-rT) * max(E - S_min, 0)

Antithetic paths (same normal draws, diffusion term negated) are used
throughout for variance reduction. Part (b) estimates the Asian put's Delta
by bumping the initial price by h and differencing, again combined with
antithetic variance reduction. Part (c) prices a snowball note that pays
out based on how often the asset stays within a band [S_lower, S_upper].
"""
import os

import numpy as np
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "q7")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Set seed for reproducibility
np.random.seed(100)

S = 1.0
E = 1.2
sigma = 0.3
r = 0.01
T = 1.0
N = 52
Dt = T/N
M = 10000

def asian():
    V = np.zeros(M)
    Vanti = np.zeros(M)

    # calculate drift and diffusion constants
    drift = (r - 0.5 * sigma ** 2) * Dt
    diffusion = sigma * np.sqrt(Dt)

    for i in range(M):
        samples = np.random.standard_normal(N)

        # Monte Carlo path
        Svals = S * np.cumprod(np.exp(drift + diffusion * samples))

        S_avg = np.mean(Svals)
        v_val = np.exp(-r*T)*max(E - S_avg, 0)

        V[i] = v_val

        # Antithetic path
        Svals2 = S * np.cumprod(np.exp(drift - diffusion * samples))

        S2_avg = np.mean(Svals2)
        v2_val = np.exp(-r*T)*max(E - S2_avg, 0)

        Vanti[i] = 0.5 * (V[i] + v2_val)

    return V, Vanti

def lookback():
    V = np.zeros(M)
    Vanti = np.zeros(M)

    # calculate drift and diffusion constants
    drift = (r - 0.5 * sigma ** 2) * Dt
    diffusion = sigma * np.sqrt(Dt)

    for i in range(M):
        samples = np.random.standard_normal(N)

        # Monte Carlo path
        Svals = S * np.cumprod(np.exp(drift + diffusion * samples))

        S_min = np.min(Svals)
        v_val = np.exp(-r * T) * max(E - S_min, 0)

        V[i] = v_val

        # Antithetic path
        Svals2 = S * np.cumprod(np.exp(drift - diffusion * samples))

        S2_min = np.min(Svals2)
        v2_val = np.exp(-r * T) * max(E - S2_min, 0)

        Vanti[i] = 0.5 * (V[i] + v2_val)

    return V, Vanti

def delta(mm=M):
    h = 0.001

    drift_factor = np.exp((r - 0.5 * sigma ** 2) * Dt)
    diffusion_scale = sigma * np.sqrt(Dt)

    deltas = np.zeros(mm)

    for i in range(mm):
        samples = np.random.standard_normal(N)


        path_m = np.cumprod(drift_factor * np.exp(diffusion_scale * samples))

        Si_path = S * path_m
        Sih_path = (S + h) * path_m


        payoff_S = np.exp(-r * T) * max(E - np.mean(Si_path), 0)
        payoff_Sh = np.exp(-r * T) * max(E - np.mean(Sih_path), 0)

        # Antithetic variance reduction
        path_a = np.cumprod(drift_factor * np.exp(-diffusion_scale * samples))

        Si_path_a = S * path_a
        Sih_path_a = (S + h) * path_a


        payoff_S_a = np.exp(-r * T) * max(E - np.mean(Si_path_a), 0)
        payoff_Sh_a = np.exp(-r * T) * max(E - np.mean(Sih_path_a), 0)


        deltas[i] = 0.5 * ((payoff_Sh - payoff_S) + (payoff_Sh_a - payoff_S_a)) / h

    return deltas

def plot_delta():
    M_list = np.logspace(1, 6, num=12, dtype=int)
    delta_means = []
    delta_err = []

    for M_i in M_list:
        delta_i = delta(M_i)
        mean_d = np.mean(delta_i)
        std_d = np.std(delta_i, ddof=1)
        delta_means.append(mean_d)
        delta_err.append(1.96 * std_d / np.sqrt(M_i))

    delta_means = np.array(delta_means)
    delta_err = np.array(delta_err)


    delta_ref = np.mean(delta(200000))

    plt.figure(figsize=(8, 6))

    plt.errorbar(
        M_list,
        np.abs(delta_means),
        yerr=delta_err,
        fmt='x',
        capsize=3,
        label='MC Delta estimate'
    )

    plt.axhline(
        abs(delta_ref),
        linestyle='--',
        linewidth=2,
        label='Reference delta'
    )

    plt.xscale('log')
    plt.yscale('log')

    plt.xlabel('Num samples')
    plt.ylabel('Delta approximation')
    plt.grid(False)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, '2b.png'))
    plt.close()

def snowball_option():
    S_lower = 0.8
    S_upper = 1.2
    A0 = 1.0
    dA = 0.2
    n = 12
    dt_sb = T / n

    V = np.zeros(M)

    drift = (r - 0.5 * sigma ** 2) * dt_sb
    diffusion = sigma * np.sqrt(dt_sb)

    for i in range(M):
        samples = np.random.standard_normal(n)

        Svals = S * np.cumprod(np.exp(drift + diffusion * samples))

        hits = np.sum((Svals >= S_lower) & (Svals <= S_upper))

        payoff = A0 + dA * hits
        V[i] = np.exp(-r * T) * payoff

    return V


if __name__ == "__main__":
    # (a) Asian and lookback puts, standard vs. antithetic Monte Carlo
    V, Vanti = asian()
    aM, bM = np.mean(V), np.std(V, ddof=1)
    aManti, bManti = np.mean(Vanti), np.std(Vanti, ddof=1)
    print(f"[Asian] Mean: {aM:.4f}")
    print(f"[Asian] 95% Confidence: [{aM - 1.96*bM/np.sqrt(M):.4f}, {aM + 1.96*bM/np.sqrt(M):.4f}]")
    print(f"[Asian] Antithetic Mean: {aManti:.4f}")
    print(f"[Asian] 95% Confidence: [{aManti - 1.96*bManti/np.sqrt(M):.4f}, {aManti + 1.96*bManti/np.sqrt(M):.4f}]")

    V, Vanti = lookback()
    aM, bM = np.mean(V), np.std(V, ddof=1)
    aManti, bManti = np.mean(Vanti), np.std(Vanti, ddof=1)
    print(f"[Lookback] Mean: {aM:.4f}")
    print(f"[Lookback] 95% Confidence: [{aM - 1.96*bM/np.sqrt(M):.4f}, {aM + 1.96*bM/np.sqrt(M):.4f}]")
    print(f"[Lookback] Antithetic Mean: {aManti:.4f}")
    print(f"[Lookback] 95% Confidence: [{aManti - 1.96*bManti/np.sqrt(M):.4f}, {aManti + 1.96*bManti/np.sqrt(M):.4f}]")

    # (b) Delta of the Asian put, and convergence of the MC estimate with sample size
    deltas = delta()
    aM = np.mean(deltas)
    bM_sq = np.var(deltas, ddof=1)
    standard_error = np.sqrt(bM_sq / M)
    print(f"Mean Delta (aM): {aM:.6f}")
    print(f"Standard Error: {standard_error:.6f}")
    plot_delta()

    # (c) Snowball option
    V = snowball_option()
    aM = np.mean(V)
    bM = np.std(V, ddof=1)
    conf = [aM - 1.96 * bM / np.sqrt(M), aM + 1.96 * bM / np.sqrt(M)]
    print(f"Standard MC Mean: {aM:.4f}")
    print(f"Standard MC 95% Conf: {conf}")
