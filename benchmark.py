import numpy as np

'''
    File that contains benchmark functions
'''

# Sphere Function
def sphere_func(x):
    return np.sum(x**2, axis=1)

# Elliptic Function
def elliptic_func(x):
    ps, D = x.shape
    a = 1e6
    fit = np.zeros(ps)
    for i in range(D):
        fit += a**(i / (D - 1)) * x[:, i]**2
    return fit

# Rotated Elliptic Function
def elliptic_rot_func(x, M):
    x_rotated = x @ M
    return elliptic_func(x_rotated)

# Schwefel's Problem 1.2
def schwefel_func(x):
    ps, D = x.shape
    fit = np.zeros(ps)
    for i in range(D):
        fit += np.sum(x[:, :i + 1], axis=1)**2
    return fit

# Rosenbrock's Function
def rosenbrock_func(x):
    ps, D = x.shape
    return np.sum(100 * (x[:, :D-1]**2 - x[:, 1:D])**2 + (x[:, :D-1] - 1)**2, axis=1)

# Rastrigin's Function
def rastrigin_func(x):
    return np.sum(x**2 - 10 * np.cos(2 * np.pi * x) + 10, axis=1)

# Rotated Rastrigin's Function
def rastrigin_rot_func(x, M):
    x_rotated = x @ M
    return rastrigin_func(x_rotated)

# Ackley's Function
def ackley_func(x):
    ps, D = x.shape
    fit1 = -0.2 * np.sqrt(np.sum(x**2, axis=1) / D)
    fit2 = np.sum(np.cos(2 * np.pi * x), axis=1) / D
    return 20 - 20 * np.exp(fit1) - np.exp(fit2) + np.exp(1)

# Rotated Ackley's Function
def ackley_rot_func(x, M):
    x_rotated = x @ M
    return ackley_func(x_rotated)
