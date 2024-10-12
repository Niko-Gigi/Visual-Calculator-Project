from flask import Flask, request, jsonify
import numpy as np

app = Flask(__name__)

def simulate_wave_equation(c, n, num_steps=500):
    xmin, xmax = 0, 1
    X, dx = np.linspace(xmin, xmax, n, retstep=True)
    dt = 0.1 * dx / c

    def initial_u(x):
        return np.exp(-0.5 * np.power(((x - 0.5) / 0.08), 2))

    U = [initial_u(X)]

    for t in range(1, num_steps):
        u_prev = U[-1]
        u_new = np.zeros_like(u_prev)

        for j in range(n):
            if j == 0:
                u_new[j] = u_prev[j] + c * dt / (2 * dx) * (u_prev[j + 1] - u_prev[-1])
            elif j == n - 1:
                u_new[j] = u_prev[j] + c * dt / (2 * dx) * (u_prev[0] - u_prev[j - 1])
            else:
                u_new[j] = u_prev[j] + c * dt / (2 * dx) * (u_prev[j + 1] - u_prev[j - 1])

        U.append(u_new)

    return [
        [{"x": float(X[i]), "u": float(U[t][i])} for i in range(n)]
        for t in range(num_steps)
    ]