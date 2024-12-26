'''
 Water Cycle Algorithm (WCA) (Standard Version)

 This code is prepared for single objective function (minimization), unconstrained, and continuous problems.
 Note that in order to obey the copy-right rules, please cite our published paper properly. Thank you.

 Hadi Eskandar, Ali Sadollah, Ardeshir Bahreininejad, Mohd Hamdi “Water cycle algorithm - a novel metaheuristic optimization method for solving constrained engineering optimization problems”, Computers & Structures, 110-111 (2012) 151-166.
 https://www.sciencedirect.com/science/article/pii/S2352711016300024

 INPUTS:

 objective_function:           Objective function which you wish to minimize or maximize
 LB:                           Lower bound of a problem
 UB:                           Upper bound of a problem
 nvars:                        Number of design variables
 npop:                         Population size
 nsr:                          Number of rivers + sea
 dmax:                         Evporation condition constant
 max_it:                       Maximum number of iterations

 OUTPUTS:

 xmin:                         Global optimum solution
 fmin:                         Cost of global optimum solution
 NFEs:                         Number of function evaluations
 elapsed_time                  Elasped time for solving an optimization problem
'''
import sys
sys.path.append('../')
import benchmark
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
from matplotlib.animation import FuncAnimation

def wca(objective_function, config, gif=False, real_time=False):

    start = time.time()
    
    # ------------ Read config params -------------------------------------------
    LB, UB = config.get("LB", -5), config.get("UB", 5)
    nvars = config.get("nvars", 2)
    npop = config.get("npop", 50)
    nsr = config.get("nsr", 4)
    dmax = config.get("dmax", 1e-16)
    max_it = config.get("max_it", 100)
    
    # ------------ Create initial population ------------------------------------
    class Individual:
        __slots__ = ("position", "cost")
        def __init__(self, position, cost):
            self.position = position
            self.cost = cost

    def initialize_population():
        population = []
        for _ in range(npop):
            position = LB + (UB - LB) * np.random.rand(nvars) # ex. [0.33, 1.21, ... , n] - nx1 matrix
            cost = objective_function(position.reshape(1, -1))[0] # reshape to 1xn matrix 
            population.append(Individual(position, cost))
        return sorted(population, key=lambda ind: ind.cost)

    population = initialize_population()

    sea, rivers, streams = population[0], population[1:nsr], population[nsr:]
    
    for p in population:
        print(str(p.position) + " cost: " + str(p.cost))
    # -------------------------------------------------------------------------
   
    #---------- Alocate number of streams for each sea and river --------------

    # NSn = round(abs(Cn - max(Cs) / sum(Csr)) * streams.size())
    # NSn - Number of streams allocated to the n-th river or sea
    # Cn - Cost associated with the n-th river or sea
    # Cs - Cost of individual streams, starting from the lowest cost stream (sorted by cost)
    # Csr - Total cost of the sea and rivers combined
 
    Cs1 = streams[0].cost; # best stream by cost
    Csr = np.array([sea.cost] + [river.cost for river in rivers]) # Array of sea and river costs
   
    # Adjust stream allocation based on cost differences and normalize
    sr_adjusted = abs((Csr - Cs1) / sum(Csr))
    NS = np.round(sr_adjusted * len(streams))
    
    # Normalize NS to ensure the sum of streams allocated almost matches the total number of streams
    NS = (NS / NS.sum() * len(streams)).astype(int)

    #---------- NS Modification START -------------------------------------

    # Ensure total number of streams in NS don't exceed number of streams
    i = len(NS) # number of rivers + sea
    while(sum(NS) > len(streams)):
        if(NS[i-1] > 1):
            NS[i-1] = NS[i-1] - 1
        else:
            i -= 1

    # Ensure that the total number of streams matches the available streams, adding remaining streams to the sea
    if(sum(NS) < len(streams)):
        NS[0] = NS[0] + len(streams) - sum(NS)
    
    # Prevent any river or sea from having zero streams allocated
    trailing_zeros_start = np.count_nonzero(NS[::-1])  

    if trailing_zeros_start > 0 and trailing_zeros_start < len(NS):  # Only proceed if there are trailing zeros
        for idx in range(trailing_zeros_start, len(NS)):  
            while NS[idx] == 0:  
                for i in range(trailing_zeros_start - 1, -1, -1):  # Redistribute from earlier elements
                    if i >= 0 and i < len(NS):  
                        if NS[i] > 1:  # Ensure at least one stream remains in the source
                            redistribution = max(1, round(NS[i] / 6))  # Proportional redistribution
                            NS[idx] += redistribution 
                            NS[i] -= redistribution  
                            break  # Move to the next zero
     
    NS = np.sort(NS)[::-1]
    print(NS)

    #------ NS Modification END -----------------------------------------------

    #----------- Main Loop for WCA --------------------------------------------
    print("******************** Water Cycle Algorithm (WCA)********************")
    print("*Iterations     Function Values *")
    print("********************************************************************")

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

        with writer.saving(fig, "gifs//WCA.gif", dpi=100):
            for it in range(max_it):

                # Clear the axis
                ax.clear()

                # Plot the 3D surface
                ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.8)

                # Plot sea, rivers, and streams
                ax.scatter(sea.position[0], sea.position[1], sea.cost, c="red", s=80, label="Sea")
                for river in rivers:
                    ax.scatter(river.position[0], river.position[1], river.cost, c="green", s=50, label="River")
                for stream in streams:
                    ax.scatter(stream.position[0], stream.position[1], stream.cost, c="blue", s=10, label="Stream")

                ax.set_title(f"Iteration {it + 1}")
                ax.set_xlabel("X1")
                ax.set_ylabel("X2")
                ax.set_zlabel("Cost")

                handles, labels = ax.get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                ax.legend(by_label.values(), by_label.keys(), loc="upper right")

                # Save the frame
                writer.grab_frame()

                population, dmax = exploration_phase(objective_function, LB, UB, nvars, max_it, NS, dmax, sea, rivers, streams)
                sea, rivers, streams = population[0], population[1:nsr], population[nsr:]
                print(f"Iteration: {it + 1}, Position: {sea.position}, Fmin: {sea.cost}")

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
            nonlocal sea, rivers, streams, dmax

            # Clear the axis
            ax.clear()

            # Plot the 3D surface
            ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.8)

            # Update positions for the sea, rivers, and streams
            sea_positions = [sea.position]
            river_positions = [river.position for river in rivers]
            stream_positions = [stream.position for stream in streams]

            sea_costs = [sea.cost]
            river_costs = [river.cost for river in rivers]
            stream_costs = [stream.cost for stream in streams]

            # Scatter plot updates
            ax.scatter(*zip(*sea_positions), sea_costs, c="red", s=80, label="Sea")
            ax.scatter(*zip(*river_positions), river_costs, c="green", s=50, label="Rivers")
            ax.scatter(*zip(*stream_positions), stream_costs, c="blue", s=10, label="Streams")

            ax.set_title(f"Iteration {frame + 1}")
            ax.set_xlabel("X1")
            ax.set_ylabel("X2")
            ax.set_zlabel("Cost")

            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), loc="upper right")

            population, dmax = exploration_phase(objective_function, LB, UB, nvars, max_it, NS, dmax, sea, rivers, streams)
            sea, rivers, streams = population[0], population[1:nsr], population[nsr:]

        # Animation
        ani = FuncAnimation(fig, update, frames=max_it, repeat=False)
        plt.show()

        end = time.time()
        print(f"sea: Position: {sea.position}, Fmin: {sea.cost}")
        print(f"elapsed_time: {end - start}")

    else:
        # Run algorithm without visualization
        for it in range(max_it):
            population, dmax = exploration_phase(objective_function, LB, UB, nvars, max_it, NS, dmax, sea, rivers, streams) 
            sea, rivers, streams = population[0], population[1:nsr], population[nsr:] 
            print(f"Iteration: {it + 1}, Position: {sea.position}, Fmin: {sea.cost}")
    
    end = time.time()
    print(f"sea: Position: {sea.position}, Fmin: {sea.cost}")
    print(f"elapsed_time: {end - start}")



def exploration_phase(objective_function, LB, UB, nvars, max_it, NS, dmax, sea, rivers, streams):
        
    #-------- Moving stream to sea ----------------------------------------
        for i in range(0, NS[0]):
            streams[i].position += np.random.rand() * 2 * (sea.position - streams[i].position)
            streams[i].position = np.clip(streams[i].position, LB, UB)
            streams[i].cost = objective_function(streams[i].position.reshape(1, -1))[0]
            if (streams[i].cost < sea.cost):
                new_sea = streams[i]
                streams[i] = sea
                sea = new_sea

        #------- Moving streams to rivers ---------------------------------------
        for k in range(0, len(rivers)):
            for j in range(sum(NS[0:k+1]), sum(NS[0:k+1]) + NS[k+1]):
                streams[j].position += np.random.rand() * 2 * (rivers[k].position - streams[j].position)
                streams[j].position = np.clip(streams[j].position, LB, UB)
                streams[j].cost = objective_function(streams[j].position.reshape(1, -1))[0]

                if (streams[j].cost < rivers[k].cost):
                    new_river = streams[j]
                    streams[j] = rivers[k]
                    rivers[k] = new_river

                    if (rivers[k].cost < sea.cost):
                        new_sea = rivers[k]
                        rivers[k] = sea
                        sea = new_sea

        #------- Moving rivers to sea --------------------------------------------
        for i in range(0, len(rivers)):
            rivers[i].position += np.random.rand() * 2 * (sea.position - rivers[i].position)
            rivers[i].position = np.clip(rivers[i].position, LB, UB)
            rivers[i].cost = objective_function(rivers[i].position.reshape(1, -1))[0]
            if (rivers[i].cost < sea.cost):
                new_sea = rivers[i]
                rivers[i] = sea
                sea = new_sea

        #---------------- Evaporation condition and raining process----------------
        
        # Evaporation condition for rivers and sea
        for k in range(len(rivers)):
            # Check the evaporation condition
            if (np.linalg.norm(rivers[k].position - sea.position) < dmax or np.random.rand() < 0.1):
                # Reset positions of streams allocated to the k-th river (raining process)
                start_idx = sum(NS[:k+1])  # Starting index of streams for this river
                end_idx = start_idx + NS[k+1]  # Ending index of streams for this river
                for j in range(start_idx, end_idx):
                    streams[j].position = LB + np.random.rand(nvars) * (UB - LB)
                    streams[j].cost = objective_function(streams[j].position.reshape(1, -1))[0]

        # Evaporation condition for streams and sea
        for j in range(NS[0]):  # Streams allocated to the sea
            if np.linalg.norm(streams[j].position - sea.position) < dmax:
                streams[j].position = LB + np.random.rand(nvars) * (UB - LB)
                streams[j].cost = objective_function(streams[j].position.reshape(1, -1))[0]

        # Gradually decrease dmax
        dmax -= dmax / max_it

        # update population
        population = [sea] + rivers + streams
        population = sorted(population, key=lambda ind: ind.cost)
        #sea, rivers, streams = population[0], population[1:nsr], population[nsr:]
        
        return population, dmax
