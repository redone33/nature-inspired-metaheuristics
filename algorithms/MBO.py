'''
 Monarch Butterfly Optimization (MBO) (Standard Version)

 This code is prepared for single objective function (minimization), unconstrained, and continuous problems.
 Note that in order to obey the copy-right rules, please cite our published paper properly. Thank you.

 Gai-Ge Wang, Suash Deb, and Zhihua Cui, Monarch Butterfly Optimization. Neural Computing and Applications, in press.
 DOI: 10.1007/s00521-015-1923-y
 
 https://link.springer.com/article/10.1007/s00521-015-1923-y
 
 INPUTS: 

 objective_function:           Objective function which you wish to minimize or maximize
 LB:                           Lower bound of a problem
 UB:                           Upper bound of a problem
 nvars:                        Dimension
 Npop:                         Population size
 Keep:                         Elitism parameter: how many of the best habitats to keep from one generation to the next
 p:                            Ratio of monarch butterflies in Land1 (default = 5/12 as per migration period)
 period:                       Migration period (defaul = 1.2 indicating 12 months)
 smax:                         Max step (default = 1.0)
 BAR:                          Butterfly adjusting rate (default = 5/12)
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

class Butterfly:
    __slots__ = ("position", "cost")
    def __init__(self, position, cost):
        self.position = position
        self.cost = cost

def levy_flight(step_size, dim):
    return np.sum(np.tan(np.pi * (np.random.rand(step_size, dim) - 0.5)), axis=0)


def mbo(objective_function, config, gif=False, real_time=False):

    start_time = time.time()

    # ------------ Read config params -------------------------------------------
    LB, UB = config.get("LB", -5), config.get("UB", 5)
    nvars = config.get("nvars")
    Npop = config.get("Npop", 100)
    Keep = config.get("Keep", 2)
    p = config.get("p", 0.4167)
    period = config.get("period", 1.2)
    smax = config.get("smax", 1.0)
    BAR = config.get("BAR", 0.4167)
    max_it = config.get("max_it", 100)

    # Initialize population
    num_butterfly1 = int(np.ceil(p * Npop)) # NP1 in paper
    num_butterfly2 = int(Npop - num_butterfly1) # NP2 in paper

    # Ensure population size consistency
    Npop = num_butterfly1 + num_butterfly2

    def initialize_population():
        population = []
        for _ in range(Npop):
            position = LB + (UB - LB) * np.random.rand(nvars)
            cost = objective_function(position.reshape(1, -1))[0]
            population.append(Butterfly(position, cost))
        return sorted(population, key=lambda butteffly: butteffly.cost)

    population = initialize_population()

    print("Initial population:")
    for pop in population:
        print("position" + str(pop.position) + " cost: " + str(pop.cost))

    if gif:
        # Save visualization as .gif file
        print("Generating gif...")

        # Set up the figure and writer
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        writer = PillowWriter(fps=1)

        x = np.linspace(LB, UB, 100)
        y = np.linspace(LB, UB, 100)
        X, Y = np.meshgrid(x, y)
        points = np.stack([X.ravel(), Y.ravel()], axis=1)
        Z = objective_function(points).reshape(X.shape)

        with writer.saving(fig, "gifs//MBO.gif", dpi=100):
            for gen_index in range(1, max_it + 1):

                # Clear the axis
                ax.clear()

                # Plot the 3D surface
                ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.8)

                # Plot best butterfly, population1, population2,
                ax.scatter(population[0].position[0], population[0].position[1], population[0].cost, c="red", s=80, label="Global best butterfly")
                for butterfly in population[:num_butterfly1]:
                    ax.scatter(butterfly.position[0], butterfly.position[1], butterfly.cost, c="green", s=40, label="Land1")
                for butterfly in population[num_butterfly1:]:
                    ax.scatter(butterfly.position[0], butterfly.position[1], butterfly.cost, c="blue", s=20, label="Land2")

                ax.set_title(f"Iteration {gen_index}")
                ax.set_xlabel("X1")
                ax.set_ylabel("X2")
                ax.set_zlabel("Cost")

                handles, labels = ax.get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                ax.legend(by_label.values(), by_label.keys(), loc="upper right")

                # Save the frame
                writer.grab_frame()

                population = optimization_step(gen_index, population, num_butterfly1, num_butterfly2, period, p, BAR, LB, UB, smax, max_it, Keep, objective_function)
                print(f"Iteration: {gen_index}, Position: {population[0].position}, Fmin: {population[0].cost}")

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

        gen_index = 1

        def update(frame):
            nonlocal population, gen_index

            # Clear the axis
            ax.clear()

            # Plot the 3D surface
            ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.8)

            # Update positions for the sea, rivers, and streams
            best_butterfly_pos = population[0].position
            pop1_pos = [pop.position for pop in population[:num_butterfly1]]
            pop2_pos = [pop.position for pop in population[num_butterfly1:]]

            best_butterfly_cost = population[0].cost
            pop1_cost = [pop.cost for pop in population[:num_butterfly1]]
            pop2_cost = [pop.cost for pop in population[num_butterfly1:]]

            # Scatter plot updates
            ax.scatter(*zip(best_butterfly_pos), best_butterfly_cost, c="red", s=80, label="Global best butterfly")
            ax.scatter(*zip(*pop1_pos), pop1_cost, c="green", s=40, label="Land1")
            ax.scatter(*zip(*pop2_pos), pop2_cost, c="blue", s=20, label="Land2")

            ax.set_title(f"Iteration {frame + 1}")
            ax.set_xlabel("X1")
            ax.set_ylabel("X2")
            ax.set_zlabel("Cost")

            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), loc="upper right")

            population = optimization_step(gen_index, population, num_butterfly1, num_butterfly2, period, p, BAR, LB, UB, smax, max_it, Keep, objective_function) 
            gen_index = gen_index + 1

        # Animation
        ani = FuncAnimation(fig, update, frames=max_it, repeat=False)
        plt.show()

    else: 
        # no visualization 
        for gen_index in range(1, max_it + 1):
            population = optimization_step(gen_index, population, num_butterfly1, num_butterfly2, period, p, BAR, LB, UB, smax, max_it, Keep, objective_function)
            print(f"Generation: {gen_index}, Position: {population[0].position}, Fmin: {population[0].cost}")

    elapsed_time = time.time() - start_time

    print(f"Best butterfly: Position: {population[0].position}, Fmin: {population[0].cost}")
    print(f"Elapsed_time: {elapsed_time}")

    #return best_solution, best_fitness, elapsed_time

def optimization_step(gen_index, population, num_butterfly1, num_butterfly2, period, p, BAR, LB, UB, smax, max_it, Keep, objective_function):
    nvars = len(population[0].position)

    # Save the best monarch butterflis in a temporary array.
    previous_best = []
    for j in range(Keep):
        previous_best.append(population[j])

    # Divide the whole population into Population1 (Land1) and Population2 (Land2)
    # according to their fitness.
    # The monarch butterflies in Population1 are better than or equal to Population2.
    # Of course, we can randomly divide the whole population into Population1 and Population2.
    # We do not test the different performance between two ways.
    # Divide the population into two subpopulations based on fitness
    population1 = population[:num_butterfly1]
    population2 = population[num_butterfly1:]

    # Migration operator
    new_population1 = []
    for k1 in range(num_butterfly1):
        new_position = np.copy(population1[k1].position)
        for var in range(nvars):
            r1 = np.random.rand() * period
            if r1 <= p:
                r2 = np.random.randint(0, num_butterfly1)
                new_position[var] = population1[r2].position[var]
            else:
                r3 = np.random.randint(0, num_butterfly2)
                new_position[var] = population2[r3].position[var]

        # Create new butterfly for Population1
        cost = objective_function(new_position.reshape(1, -1))[0]
        new_population1.append(Butterfly(new_position, cost))

    # Butterfly adjusting operator
    new_population2 = []
    for k2 in range(num_butterfly2):
        scale = smax / (gen_index**2)
        step_size = int(np.ceil(np.random.exponential(2 * max_it, 1)[0]))
        delta_x = levy_flight(step_size, nvars)
        new_position = np.copy(population2[k2].position)

        for var in range(nvars):
            if np.random.rand() >= p:
                new_position[var] = population1[0].position[var]
            else:
                r4 = np.random.randint(0, num_butterfly2)
                new_position[var] = population2[r4].position[var]
                if np.random.rand() > BAR:
                    new_position[var] += scale * (delta_x[var] - 0.5)
                    new_position[var] = np.clip(new_position[var], LB, UB) # Clip the value because levy can give large numbers

        cost = objective_function(new_position.reshape(1, -1))[0]
        new_population2.append(Butterfly(new_position, cost))

    # Combine two subpopulations
    population = sorted(new_population1 + new_population2, key=lambda butterfly: butterfly.cost)

    # Elitism strategy: Keep the best individuals
    for k3 in range(Keep):
        population[-(k3 + 1)] = previous_best[k3]

    return population