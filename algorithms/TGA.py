'''
 Tree Growth Algorithm (TGA) (Standard Version)

 This code is prepared for single objective function (minimization), unconstrained, and continuous problems.
 Note that in order to obey the copy-right rules, please cite our published paper properly. Thank you.

 Armin Cheraghalipour, Mostafa Hajiaghaei-Keshteli - Tree Growth Algorithm (TGA): An Effective Metaheuristic Algorithm Inspired by trees' behavior
 https://www.researchgate.net/publication/320009185
 
 INPUTS: 

 objective_function:           Objective function which you wish to minimize or maximize
 LB:                           Lower bound of a problem
 UB:                           Upper bound of a problem
 nvars:                        Number of design variables
 Npop                          Population size
 N1                            Number of best trees
 N2                            Number of next best trees
 N3                            Number of worst trees (100-(N1+N2))
 N4                            New generated solution (first N1 number)
 lambda                        Linear combination factor
 theta                         Tree reduction rate of power
 max_it:                       Maximum number of iterations

 OUTPUTS:

 Xmin:                         Global optimum solution
 Fmin:                         Cost of global optimum solution
 Elapsed_Time                  Elasped time for solving an optimization problem
'''

import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter, FuncAnimation

class Tree:
    __slots__ = ("position", "cost")
    def __init__(self, position, cost):
        self.position = position
        self.cost = cost

def tga(objective_function, config, gif=False, real_time=False):

    start_time = time.time()
    
    # ------------ Read config params -------------------------------------------
    LB, UB = config.get("LB", -5), config.get("UB", 5)
    nvars = config.get("nvars", 2)
    Npop = config.get("Npop", 100)
    N1 = config.get("N1", 40)
    N2 = config.get("N2", 40)
    N3 = config.get("N3", 20)
    N4 = config.get("N4", 30)
    lambda_ = config.get("lambda", 0.5)
    theta = config.get("theta", 1.1)
    max_it = config.get("max_it", 100)

    # Ensure population size consistency
    Npop = N1 + N2 + N3

    # Initialize population
    def initialize_population():
        population = []
        for _ in range(Npop):
            position = LB + (UB - LB) * np.random.rand(nvars)
            cost = objective_function(position.reshape(1, -1))[0]
            population.append(Tree(position, cost))
        return sorted(population, key=lambda tree: tree.cost)

    population = initialize_population()

    # Divide population into groups
    best_trees = population[:N1]
    next_best_trees = population[N1:N1+N2]
    worst_trees = population[-N3:]

    for p in population:
        print(str(p.position) + " cost: " + str(p.cost))

    if gif:
        # Save visualization as .gif file

        # Set up the figure and writer
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        writer = PillowWriter(fps=1)

        x = np.linspace(LB, UB, 100)
        y = np.linspace(LB, UB, 100)
        X, Y = np.meshgrid(x, y)
        points = np.stack([X.ravel(), Y.ravel()], axis=1)
        Z = objective_function(points).reshape(X.shape)

        with writer.saving(fig, "gifs//TGA.gif", dpi=100):
            for it in range(max_it):

                # Clear the axis
                ax.clear()

                # Plot the 3D surface
                ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.8)

                # Plot global_best_tree, best_trees, next_best_trees, and worst_trees
                ax.scatter(best_trees[0].position[0], best_trees[0].position[1], best_trees[0].cost, c="red", s=80, label="Global best tree")
                for tree in best_trees:
                    ax.scatter(tree.position[0], tree.position[1], tree.cost, c="green", s=50, label="Best trees")
                for tree in next_best_trees:
                    ax.scatter(tree.position[0], tree.position[1], tree.cost, c="blue", s=20, label="Next best trees")
                for tree in worst_trees:
                    ax.scatter(tree.position[0], tree.position[1], tree.cost, c="purple", s=10, label="Worst trees")

                ax.set_title(f"Iteration {it + 1}")
                ax.set_xlabel("X1")
                ax.set_ylabel("X2")
                ax.set_zlabel("Cost")

                handles, labels = ax.get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                ax.legend(by_label.values(), by_label.keys(), loc="upper right")

                # Save the frame
                writer.grab_frame()

                population = exploration_phase(objective_function, LB, UB, nvars, Npop, lambda_, theta, population, best_trees, next_best_trees, worst_trees, N4) 
                best_trees, next_best_trees, worst_trees = population[:N1], population[N1:N1+N2], population[-N3:] 
                print(f"Iteration: {it + 1}, Position: {best_trees[0].position}, Fmin: {best_trees[0].cost}")  

    elif real_time:
        # Run algorithm with real-time visualization

        # Set up the figure
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        
        x = np.linspace(LB, UB, 100)
        y = np.linspace(LB, UB, 100)
        X, Y = np.meshgrid(x, y)
        points = np.stack([X.ravel(), Y.ravel()], axis=1)
        Z = objective_function(points).reshape(X.shape)

        def update(frame):
            nonlocal best_trees, next_best_trees, worst_trees, population

            # Clear the axis
            ax.clear()

            # Plot the 3D surface
            ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.8)

            # Update positions for the sea, rivers, and streams
            global_best_tree_position = best_trees[0].position
            best_trees_positions = [tree.position for tree in best_trees]
            next_best_trees_positions = [tree.position for tree in next_best_trees]
            worst_trees_positions = [tree.position for tree in worst_trees]

            global_best_tree_cost = best_trees[0].cost
            best_trees_costs = [tree.cost for tree in best_trees]
            next_best_trees_costs = [tree.cost for tree in next_best_trees]
            worst_trees_costs = [tree.cost for tree in worst_trees]

            # Scatter plot updates
            ax.scatter(*zip(global_best_tree_position), global_best_tree_cost, c="red", s=80, label="Global best tree")
            ax.scatter(*zip(*best_trees_positions), best_trees_costs, c="green", s=50, label="Best trees")
            ax.scatter(*zip(*next_best_trees_positions), next_best_trees_costs, c="blue", s=20, label="Next best trees")
            ax.scatter(*zip(*worst_trees_positions), worst_trees_costs, c="blue", s=10, label="Worst trees")

            ax.set_title(f"Iteration {frame + 1}")
            ax.set_xlabel("X1")
            ax.set_ylabel("X2")
            ax.set_zlabel("Cost")

            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), loc="upper right")

            population = exploration_phase(objective_function, LB, UB, nvars, Npop, lambda_, theta, population, best_trees, next_best_trees, worst_trees, N4) 
            best_trees, next_best_trees, worst_trees = population[:N1], population[N1:N1+N2], population[-N3:] 

        # Animation
        ani = FuncAnimation(fig, update, frames=max_it, repeat=False)
        plt.show()
        
    else:
        # Run algorithm without visualization\
        for it in range(max_it):
            population = exploration_phase(objective_function, LB, UB, nvars, Npop, lambda_, theta, population, best_trees, next_best_trees, worst_trees, N4) 
            best_trees, next_best_trees, worst_trees = population[:N1], population[N1:N1+N2], population[-N3:] 
            print(f"Iteration: {it + 1}, Position: {best_trees[0].position}, Fmin: {best_trees[0].cost}")

    elapsed_time = time.time() - start_time

    print(f"best tree: Position: {population[0].position}, Fmin: {population[0].cost}")
    print(f"elapsed_time: {elapsed_time}")


def exploration_phase(objective_function, LB, UB, nvars, Npop, lambda_, theta, population, best_trees, next_best_trees, worst_trees, N4):

    # Intensification: Best trees (N1) local search for 
    for tree in best_trees:
        for _ in range(5):  # Local search steps
            r = np.random.rand()
            new_position = tree.position / theta + r * tree.position
            new_position = np.clip(new_position, LB, UB)
            new_cost = objective_function(new_position.reshape(1, -1))[0]
            if new_cost < tree.cost:
                tree.position, tree.cost = new_position, new_cost

    # Diversification: Move next-best trees closer to two nearest best trees
    for tree in next_best_trees:
        distances = np.array([np.linalg.norm(tree.position - best.position) for best in best_trees]) # Calculate distances from the current tree to all best trees
        sorted_indices = np.argsort(distances)[:2] # Identify the two closest best trees
        x1, x2 = best_trees[sorted_indices[0]].position, best_trees[sorted_indices[1]].position # Linearly combine the positions of the two closest trees
        y = lambda_ * x1 + (1 - lambda_) * x2
        alpha = np.random.rand()
        tree.position += alpha * (y - tree.position) # Move the current tree towards the combined position with a random factor
        tree.position = np.clip(tree.position, LB, UB)
        tree.cost = objective_function(tree.position.reshape(1, -1))[0]


    # Replace worst trees with random positions
    for tree in worst_trees:
        tree.position = LB + (UB - LB) * np.random.rand(nvars)
        tree.cost = objective_function(tree.position.reshape(1, -1))[0]

    # Generate new solutions and apply logical mask
    new_solutions = []
    for _ in range(N4):
        new_position = LB + (UB - LB) * np.random.rand(nvars)
        mask = np.random.randint(0, 2, size=nvars)
        new_position = np.where(mask, best_trees[0].position, new_position)
        new_cost = objective_function(new_position.reshape(1, -1))[0]
        new_solutions.append(Tree(new_position, new_cost))

    population.extend(new_solutions)
    population = sorted(population, key=lambda tree: tree.cost)[:Npop]

    # update population
    population = best_trees + next_best_trees + worst_trees

    return population
