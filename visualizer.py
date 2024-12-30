'''
    Visualize function in 3D or see animation of algorithm works on function if algorithm param is defined
'''
import numpy as np
import matplotlib.pyplot as plt
import algorithms
import algorithms.TGA
import algorithms.WCA

def visualize(func, algo, gif=False, real_time=False):
    if algo is not None:
        print(f"Visualizing algorithm {algo} running on {func}...")

        match algo:
            case "WCA":
                config = {"LB": -5, "UB": 5, "nvars": 2, "npop": 50, "nsr": 4, "dmax": 1e-16, "max_it": 100}
                if gif:
                    algorithms.WCA.wca(func, config, gif=True)
                if real_time:
                    algorithms.WCA.wca(func, config, real_time=True)
            case "TGA":
                config = {"LB": -5, "UB": 5, "nvars": 2, "npop": 100, "N1": 40, "N2": 40, "N3": 20, "N4": 30, "lambda": 0.5, "theta": 1.1, "max_it": 100}
                if gif:
                    algorithms.TGA.tga(func, config, gif=True)
                if real_time:
                    algorithms.TGA.tga(func, config, real_time=True)
            case _:
                return f"Algorithm {algo} does not exists!"

    else:
        # TODO: Make bounds configurabile (in args?)
        bounds = [-5, 5] 
        visualize_function_3d(func, bounds)

def visualize_function_3d(func, bounds, grid_size=100, rotation_matrix=None):
    x = np.linspace(bounds[0], bounds[1], grid_size)
    y = np.linspace(bounds[0], bounds[1], grid_size)
    X, Y = np.meshgrid(x, y)
    points = np.stack([X.ravel(), Y.ravel()], axis=1)
    
    if rotation_matrix is not None:
        Z = func(points, rotation_matrix)
    else:
        Z = func(points)

    Z = Z.reshape(X.shape)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
    ax.set_title(func.__name__)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()