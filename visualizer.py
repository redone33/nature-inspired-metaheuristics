'''
    Visualize function in 3D or see animation of algorithm works on function if algorithm param is defined
'''
import numpy as np
import matplotlib.pyplot as plt
from algorithms.algorithm_config import algorithm_mapping

def visualize(func, algo, gif=False, real_time=False):
    if algo is not None:
        print(f"Visualizing algorithm {algo} running on {func}...")
        
        algo_info = algorithm_mapping.get(algo)
        if not algo_info:
            return f"Algorithm {algo} does not exist!"

        algo_func = algo_info["module"]
        config = algo_info["config"]

        algo_func(func, config, gif=gif, real_time=real_time)

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