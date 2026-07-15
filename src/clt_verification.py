"""
Section 2 - Central Limit Theorem.

Draws samples of size n from a non-normal population (uniform by default)
and shows that the sampling distribution of the mean converges to
N(mu, sigma / sqrt(n)) as n grows, regardless of the shape of the
underlying population.
"""
import math
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "q2")
os.makedirs(RESULTS_DIR, exist_ok=True)


def verify_clt(
    dist="uniform",
    n_list=(5, 30, 100),
    num_samples=10000,
    bins=50,
    seed=None,
    show=False,
):
    rng = np.random.default_rng(seed)

    if dist == "uniform":
        sampler = lambda m: rng.uniform(0, 10, size=m)
    elif dist == "bernoulli":
        sampler = lambda m: rng.binomial(1, 0.5, size=m)
    else:
        raise ValueError("Unsupported distribution type")

    big_sample = sampler(200_000)
    pop_mean = float(np.mean(big_sample))
    pop_var = float(np.var(big_sample, ddof=0))

    rows = len(n_list)
    fig, axes = plt.subplots(rows, 1, figsize=(7, 3 * rows), sharex=False)

    for ax, n in zip(axes, n_list):
        means = np.array([np.mean(sampler(n)) for _ in range(num_samples)])

        mu_hat = float(np.mean(means))
        sd_hat = float(np.std(means, ddof=0))

        clt_mu = pop_mean
        clt_sd = math.sqrt(pop_var / n)

        ax.hist(means, bins=bins, density=True, alpha=0.5, color="blue", edgecolor="black")
        x = np.linspace(min(means), max(means), 300)
        ax.plot(x, norm.pdf(x, loc=mu_hat, scale=sd_hat), "r-", label="Normal fit")
        ax.plot(x, norm.pdf(x, loc=clt_mu, scale=clt_sd), "k--", label=r"CLT: $N(\mu, \sigma/\sqrt{n})$")

        ax.set_title(f"Sampling distribution of the mean (n={n})")
        ax.set_ylabel("Density")
        ax.legend(loc="best")

    axes[-1].set_xlabel("Sample mean")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"{dist}.png"))
    if show:
        plt.show()
    plt.close()


if __name__ == "__main__":
    verify_clt(dist="uniform", n_list=(5, 30, 100), num_samples=10000, seed=42)
