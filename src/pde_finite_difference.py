"""
Homework II, Question 3 - Finite-difference solvers for the Black-Scholes PDE.

BTCS_put():
    Prices a European put by transforming to x = ln(S) and applying a
    backward-time-centered-space (implicit) scheme, with boundary
    conditions u(x,0) = max(E - e^x, 0), u(0,tau) = E*e^(r*tau), u(L,tau) = 0
    over the asset range [1e-4, 4E].

CrankN_option():
    Prices a down-and-out European call (barrier B, dividend yield q) with
    the Crank-Nicolson scheme directly in S-space, S in [B, Smax].
"""
import os

import numpy as np
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "q8")
os.makedirs(RESULTS_DIR, exist_ok=True)


def BTCS_put():
    S0 = 10
    E = 10
    r = 0.01
    sigma = 0.3
    T = 1
    Nx = 100
    Nt = 500
    dt = T / Nt

    Smin = 1e-4
    Smax = 4 * E

    xmin = np.log(Smin)
    xmax = np.log(Smax)
    x = np.linspace(xmin, xmax, Nx + 1)
    dx = x[1] - x[0]
    xi = x[1:-1]

    nu = 0.5 * sigma**2 * dt / dx**2
    tau = np.linspace(0, T, Nt + 1)

    main_diag = (1 + 2 * nu) * np.ones(Nx - 1)
    off_diag = -nu * np.ones(Nx - 2)
    B = np.diag(main_diag) + np.diag(off_diag, 1) + np.diag(off_diag, -1)

    U = np.zeros((Nx - 1, Nt + 1))

    U[:, 0] = np.maximum(E - np.exp(xi), 0)

    # Time stepping
    for i in range(Nt):
        rhs = U[:, i].copy()
        rhs[0] += nu * E * np.exp(r * tau[i])
        U[:, i + 1] = np.linalg.solve(B, rhs)

    x0 = np.log(S0)
    idx = np.argmin(np.abs(xi - x0))

    price = np.exp(-r * T) * U[idx, -1]
    print("European put price:", price)

    bc_left = E * np.exp(r * tau)[None, :]
    bc_right = np.zeros((1, Nt + 1))
    U= np.vstack((bc_left, U, bc_right))


    return price

def CrankN_option():
    # Problem parameters
    E = 4.0
    S0 = 10.0
    B = 9.0
    sigma = 0.3
    r = 0.01
    q = 0.005
    T = 1.0

    Smax = 30.0
    Nx = 80
    Nt = 80

    k = T / Nt
    h = (Smax - B) / Nx

    # Grids
    S = np.linspace(B + h, Smax - h, Nx - 1)
    t = np.linspace(0, T, Nt + 1)

    # Matrices
    T1 = np.diag(np.ones(Nx - 2), 1) - np.diag(np.ones(Nx - 2), -1)
    T2 = -2 * np.eye(Nx - 1) + np.diag(np.ones(Nx - 2), 1) + np.diag(np.ones(Nx - 2), -1)

    D1 = np.diag(S / h)
    D2 = np.diag((S / h) ** 2)

    F = ((1 - r * k) * np.eye(Nx - 1)
         + 0.5 * k * sigma ** 2 * D2 @ T2
         + 0.5 * k * (r - q) * D1 @ T1)

    Bmat = ((1 + r * k) * np.eye(Nx - 1)
         - 0.5 * k * sigma ** 2 * D2 @ T2
         - 0.5 * k * (r - q) * D1 @ T1)

    A1 = 0.5 * (np.eye(Nx - 1) + F)
    A2 = 0.5 * (np.eye(Nx - 1) + Bmat)

    # Solution matrix
    U = np.zeros((Nx - 1, Nt + 1))

    # Initial condition (put payoff)
    U[:, 0] = np.maximum(S - E, 0)

    upper_bc = Smax * np.exp(-q * t) - E * np.exp(-r * t)

    # Time stepping
    for n in range(Nt):
        tau = n * k
        rhs = A1 @ U[:, n]
        Sj = S[-1]

        gamma = (k / 4) * (
                sigma ** 2 * Sj ** 2 / h ** 2
                + (r - q) * Sj / h)
        rhs[-1] += gamma * (upper_bc[n] + upper_bc[n + 1])
        U[:, n + 1] = np.linalg.solve(A2, rhs)

    # Boundary conditions
    bca = np.zeros(Nt + 1)
    bcb = Smax * np.exp(-q * t) - E * np.exp(-r * t)

    U_full = np.vstack([bca, U, bcb])

    S_full = np.linspace(B, Smax, Nx + 1)
    idx = np.argmin(np.abs(S_full - S0))
    price0 = U_full[idx, 0]

    # Plot
    Tmesh, Smesh = np.meshgrid(t, np.linspace(B, Smax, Nx + 1))
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.plot_surface(Tmesh, Smesh, U_full)
    ax.set_xlabel("T - t")
    ax.set_ylabel("S")
    ax.set_zlabel("Put Value")
    plt.savefig(os.path.join(RESULTS_DIR, '3d.png'))
    plt.close()

    return price0


if __name__ == "__main__":
    print(BTCS_put())
    print(CrankN_option())
