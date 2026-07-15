"""
Homework II, Question 1 - Binomial (CRR) model for an American / Bermudan put.

    u = e^(sigma*sqrt(dt))          up factor
    d = e^(-sigma*sqrt(dt))         down factor
    p = (e^(r*dt) - d) / (u - d)    risk-neutral probability

At each step the immediate exercise value is compared to the continuation
value and the higher of the two is taken as the option value at that node;
stepping this back from T to 0 gives the American price. The same recursion,
restricted to only exercise on a fixed subset of dates, gives the Bermudan
price (Question 1b).
"""
import os

import numpy as np
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "q6")
os.makedirs(RESULTS_DIR, exist_ok=True)

S = 1.0
E = 1.5
T = 3.0
r = 0.01
sigma = 0.3
M = 10000
dt = T / M


def put_value():
    # Calculate up and down factors
    u = np.exp(sigma * np.sqrt(dt))
    d = np.exp(-sigma * np.sqrt(dt))
    p = (np.exp(r * dt) - d)/(u - d)

    dpowers = d ** np.arange(M, -1, -1)
    upowers = u ** np.arange(0, M + 1)

    W = np.maximum(E - S * dpowers * upowers, 0)
    ex_bound = np.zeros(M + 1)
    ex_bound[M] = E

    for i in  range(M, 0, -1):
        Si = S * dpowers[M-i+1 : M+1] * upowers[0 : i]

        # Future value of the option
        con = np.exp(-r * dt) * (p * W[1:i+1] + (1 - p) * W[0:i])

        # Exercise value
        exe = np.maximum(E - Si, 0)

        ex = exe > con
        ex_bound[i - 1] = np.max(Si[ex]) if Si[ex].size > 0 else np.nan

        W = np.maximum(exe, con)

    return W, ex_bound


def plot_bound(bounds):
    times = np.linspace(0, T, M + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(times, bounds, color='black', linewidth=3)

    plt.fill_between(times, bounds, 2.5, color='red', alpha=0.3, label='Do not exercise')
    plt.fill_between(times, 0, bounds, color='green', alpha=0.3,label='Exercise')

    plt.title('Optimal Exercise Boundary')
    plt.xlabel('t')
    plt.ylabel('S')
    plt.xlim(0, T)
    plt.ylim(0, 2.0)
    plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR, '1a.png'))
    plt.close()


def ber_put_value(n):
    exercise_dates = np.linspace(0, M, n + 1, dtype=int)

    # Calculate up and down factors
    u = np.exp(sigma * np.sqrt(dt))
    d = np.exp(-sigma * np.sqrt(dt))
    p = (np.exp(r * dt) - d)/(u - d)

    dpowers = d ** np.arange(M, -1, -1)
    upowers = u ** np.arange(0, M + 1)

    W = np.maximum(E - S * dpowers * upowers, 0)

    for i in  range(M, 0, -1):
        Si = S * dpowers[M-i+1 : M+1] * upowers[0 : i]

        # Future value of the option
        con = np.exp(-r * dt) * (p * W[1:i+1] + (1 - p) * W[0:i])

        if (M - i) in exercise_dates:
            # Exercise value
            exe = np.maximum(E - Si, 0)
            W = np.maximum(exe, con)
        else:
            W = con

    return W


if __name__ == "__main__":
    W, bounds = put_value()
    print(f'Option value is {W[0]}')

    plot_bound(bounds)

    print(ber_put_value(3))
    print(ber_put_value(6))
    print(ber_put_value(12))
    print(ber_put_value(36))
