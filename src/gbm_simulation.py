"""
Section 3 - Simulation of stock price paths.

Discretises the GBM SDE with a plain Euler scheme

    S(t_{i+1}) = S(t_i) * (1 + mu*dt + sigma*sqrt(dt)*Z),   Z ~ N(0, 1)

and compares the terminal distribution of S_T, for an increasing number of
time steps L, against the exact closed-form lognormal distribution implied
by GBM:

    S_T = S_0 * exp((mu - 0.5*sigma^2) * T + sigma*sqrt(T) * Z)

As L grows the discretisation error of the Euler scheme shrinks and the
simulated histogram converges to the theoretical lognormal curve.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import lognorm

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "q3")
os.makedirs(RESULTS_DIR, exist_ok=True)


def simulate_price_path(S0=170, mu=0.1, sigma=0.344, T=1, L=5, rng=None):
    rng = rng or np.random.default_rng()
    dt = T / L
    Z = rng.standard_normal(L)
    S = np.empty(L + 1)
    S[0] = S0
    for i in range(L):
        S[i + 1] = S[i] * (1 + mu * dt + sigma * np.sqrt(dt) * Z[i])
    return S


def plot_single_path(S0=170, mu=0.1, sigma=0.344, T=1, L=5, seed=42):
    rng = np.random.default_rng(seed)
    S = simulate_price_path(S0, mu, sigma, T, L, rng)
    t = np.linspace(0, T, L + 1)

    plt.figure(figsize=(7, 4))
    plt.plot(t, S, marker="o")
    plt.title(f"Discrete Price Path (L={L})")
    plt.xlabel("t")
    plt.ylabel("S(t)")
    plt.grid(True, alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "path_evolution.png"))
    plt.close()


def simulate_many_paths(M=5000, S0=170, mu=0.1, sigma=0.344, T=1, L=5, compare=True, seed=None):
    rng = np.random.default_rng(seed)
    ST = np.array([simulate_price_path(S0, mu, sigma, T, L, rng)[-1] for _ in range(M)])

    plt.figure(figsize=(7, 4))
    count, bins, _ = plt.hist(
        ST, bins=50, density=True, alpha=0.7, color="orange", edgecolor="k", label="Simulated $S_T$"
    )

    if compare:
        s = sigma * np.sqrt(T)
        scale = S0 * np.exp((mu - 0.5 * sigma**2) * T)
        x = np.linspace(min(bins), max(bins), 500)
        pdf = lognorm.pdf(x, s=s, scale=scale)
        plt.plot(x, pdf, "r-", label="Theoretical lognormal PDF")

    plt.title(f"Histogram of Simulated $S_T$ (M={M}, L={L})")
    plt.xlabel("$S_T$")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"terminal_distribution_L{L}.png"))
    plt.close()
    return ST


def euler_scheme_convergence(L_list=(10, 20, 50, 100), M=5000, S0=170, mu=0.1, sigma=0.344, T=1, seed=7):
    """Reproduces the terminal-distribution comparison across several step counts L."""
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    s = sigma * np.sqrt(T)
    scale = S0 * np.exp((mu - 0.5 * sigma**2) * T)

    for ax, L in zip(axes.ravel(), L_list):
        ST = np.array([simulate_price_path(S0, mu, sigma, T, L, rng)[-1] for _ in range(M)])
        count, bins, _ = ax.hist(ST, bins=50, density=True, alpha=0.7, color="orange", edgecolor="k")
        x = np.linspace(min(bins), max(bins), 500)
        ax.plot(x, lognorm.pdf(x, s=s, scale=scale), "r-")
        ax.set_title(f"Simulated $S_T$ (M={M}, L={L})")
        ax.set_xlabel("$S_T$")
        ax.set_ylabel("Density")

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "euler_convergence.png"))
    plt.close()


def running_quadratic_variation(L_list=(10, 100), n_paths=10, S0=170.0, mu=0.1, sigma=0.344, T=1.0, seed=1):
    """Sanity check: the running sum of squared log-returns should approach sigma^2 * T."""
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))

    for j, L in enumerate(L_list):
        dt = T / L
        t_grid = np.linspace(0.0, T, L + 1)

        for _ in range(n_paths):
            S = simulate_price_path(S0, mu, sigma, T, L, rng)
            r = np.diff(S) / S[:-1]
            qv = np.cumsum(r**2)

            axes[0, j].plot(t_grid, S)
            axes[1, j].plot(t_grid[1:], qv)

        axes[1, j].plot(t_grid[1:], np.full_like(t_grid[1:], (sigma**2) * T), ls=":", label=r"$\sigma^2 T$")
        axes[1, j].legend()

        axes[0, j].set_title(fr"Asset paths ($\Delta t = {dt:.3g}$, L={L})")
        axes[0, j].set_ylabel("Asset price $S_t$")
        axes[0, j].grid(alpha=0.3)

        axes[1, j].set_title("Running sum of squared returns")
        axes[1, j].set_xlabel("Time $t$")
        axes[1, j].set_ylabel("Sum-of-square increments")
        axes[1, j].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "quadratic_variation.png"))
    plt.close()


if __name__ == "__main__":
    plot_single_path(L=5, seed=42)
    simulate_many_paths(L=100, seed=42)
    euler_scheme_convergence(L_list=(10, 20, 50, 100), seed=7)
    running_quadratic_variation(L_list=(10, 100), n_paths=10, seed=1)
