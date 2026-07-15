# Option Pricing & Stochastic Simulation

A collection of five self-contained studies in quantitative finance, moving from
empirical return statistics up to Black-Scholes delta hedging and implied
volatility extraction. Originally developed as coursework in a stochastic
calculus / option valuation module, refactored here into a clean, runnable
project.

Everything is plain NumPy/SciPy/Matplotlib — no pricing libraries — so the
math is fully visible in the code.

## Contents

| # | Topic | Script | Concepts |
|---|-------|--------|----------|
| 1 | [Market stock prices & return distributions](#1-market-stock-prices--return-distributions) | `src/returns_analysis.py` | simple vs. log returns, sample moments, QQ-plots |
| 2 | [Central Limit Theorem](#2-central-limit-theorem) | `src/clt_verification.py` | sampling distribution of the mean |
| 3 | [GBM path simulation](#3-gbm-path-simulation) | `src/gbm_simulation.py` | Euler discretization, lognormal convergence |
| 4 | [Delta hedging & Black-Scholes](#4-delta-hedging--black-scholes) | `src/delta_hedging.py` | PDE derivation, discrete replication |
| 5 | [Implied volatility](#5-implied-volatility) | `src/implied_volatility.py` | bisection solver, smile, term structure |

## Setup

```bash
pip install -r requirements.txt
python src/fetch_data.py      # downloads AAPL daily/weekly OHLC via yfinance -> data/
```

Each `src/*.py` file can then be run standalone; figures are written to
`results/q1` ... `results/q5`.

```bash
python src/returns_analysis.py
python src/clt_verification.py
python src/gbm_simulation.py
python src/delta_hedging.py
python src/implied_volatility.py
```

## Project structure

```
option-pricing-simulations/
├── data/                    # input CSVs (fetched or provided)
├── results/                 # generated figures, one subfolder per topic
└── src/
    ├── fetch_data.py
    ├── returns_analysis.py
    ├── clt_verification.py
    ├── gbm_simulation.py
    ├── delta_hedging.py
    └── implied_volatility.py
```

---

## 1. Market stock prices & return distributions

Loads AAPL daily and weekly close prices and computes two return measures:

```
simple return:   r_i     = (S_i - S_{i-1}) / S_i
log return:      r_hat_i = ln(S_i / S_{i-1})
```

The two coincide whenever `ln(1 + r) ≈ r`, i.e. for small per-period moves —
they diverge visibly only around large single-day jumps.

Sample mean and (unbiased, `ddof=1`) variance of the daily log returns are
computed, followed by a histogram and a QQ-plot against `N(0,1)`. The same
diagnostics are repeated on weekly-aggregated data:

<p align="center">
  <img src="results/q1/qq_daily_vs_weekly.png" width="800">
</p>

The weekly QQ-plot tracks the theoretical normal line noticeably better than
the daily one — a direct empirical illustration of the Central Limit Theorem
(weekly returns are effectively 5-day sums of daily returns, so their
distribution is closer to Gaussian than the daily returns themselves, which
show the fat tails typical of financial data).

## 2. Central Limit Theorem

A synthetic check of the CLT, independent of any market data: samples of
size `n` are drawn from a uniform population and the sampling distribution
of the mean is compared, for `n = 5, 30, 100`, against:

```
N(mu, sigma / sqrt(n))
```

As `n` increases the empirical sampling distribution converges to this
Gaussian regardless of the (non-normal) shape of the underlying population.

## 3. GBM path simulation

Simulates the Euler discretization of geometric Brownian motion:

```
S(t_{i+1}) = S(t_i) * (1 + mu*dt + sigma*sqrt(dt)*Z),   Z ~ N(0, 1)
```

with `S_0 = 170`, `mu = 0.1`, `sigma = 0.344`, `T = 1`. A single path is
plotted at coarse resolution (`L = 5`), then `M = 5000` terminal values
`S_T` are simulated and histogrammed for `L = 10, 20, 50, 100` and compared
against the exact GBM solution:

```
S_T = S_0 * exp((mu - 0.5*sigma^2)*T + sigma*sqrt(T)*Z)
```

As `L` grows, the Euler scheme's discretization error shrinks and the
simulated histogram converges onto the theoretical lognormal curve. A
secondary check tracks the running sum of squared log-returns along each
path and confirms it approaches the theoretical quadratic variation
`sigma^2 * T`.

## 4. Delta hedging & Black-Scholes

**Derivation.** Starting from a self-financing portfolio of `A` shares (paying
continuous dividend yield `q`) plus a cash account `D`, matching its
instantaneous change to that of a derivative `V` via Itô's lemma and a
no-arbitrage argument yields the Black-Scholes PDE with dividends:

```
V_t + (r - q) S V_s + 0.5 sigma^2 S^2 V_ss - r V = 0
```

**Discrete hedge.** At each rebalancing step the portfolio is rolled forward
one period (earning interest `r` on cash and dividend yield `q` on the
stock position), then re-hedged to the option's new delta:

```
Pi_{i+1} = A_i S_{i+1} + (1 + r*dt) D_i + q A_i S_i dt
A_{i+1}  = dV_{i+1}/dS
D_{i+1}  = (1 + r*dt) D_i + (A_i - A_{i+1}) S_{i+1} + q A_i S_i dt
```

`delta_hedging.py` runs this for a European put (`E = S_0 = 170`, `T = 1`,
`r = 0.05`, `sigma = 0.344`, `q = 0.01`, `dt = 0.01`) along one in-the-money
and one out-of-the-money price path:

<p align="center">
  <img src="results/q4/hedge_itm.png" width="600">
</p>

The bottom panel compares the replicating portfolio's value against the
Black-Scholes price at every step — the close tracking (up to discretization
error) is the empirical content of the no-arbitrage argument.

**Terminal check (Eq. 9.9).** Running `M = 1000` simulated paths and comparing

```
Pi(S_T, T) + (P(S_0, 0) - Pi(S_0, 0)) * e^{rT}
```

against the option's actual payoff `max(K - S_T, 0)` shows the two collapse
onto the same curve, and doing this for `mu = +0.10` and `mu = -0.10`
produces statistically indistinguishable hedge-error distributions —
confirming that the hedge's effectiveness does not depend on the real-world
drift used to simulate the stock, only on `sigma`.

## 5. Implied volatility

Given a small chain of LULU call quotes (`data/lulu_options.csv`, bid/ask
mid-prices, various strikes and expiries), a bisection solver inverts the
Black-Scholes formula for `sigma` at each quote, producing:

- an **IV smile** (implied vol vs. strike), and
- a **term structure** (implied vol vs. time to maturity).

With only a handful of illiquid, noisy quotes across mixed expiries the
resulting curves are noisy rather than the smooth, monotone smile one would
see with a deep, liquid chain (e.g. index options) — a useful reminder that
implied-volatility surfaces are only as clean as the underlying option
market's liquidity.

---

## Notes & limitations

- AAPL price history is fetched fresh via `yfinance` in `fetch_data.py`
  rather than checked into the repo, so figures will vary slightly by
  fetch date/range.
- The LULU option quotes are a small, manually captured snapshot — good
  enough to demonstrate the IV-extraction machinery, not to draw
  conclusions about LULU's actual volatility surface.
- All simulations assume constant volatility and drift (plain GBM); no
  stochastic-volatility or jump dynamics are modeled.
