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

# TODO: add wca_config and use it for different params
def wca(objective_function, LB, UB, nvars, npop=50, nsr=20, dmax=1e-16, max_it=1000):

    class Individual:
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

    sea = population[0]
    rivers = population[1:nsr]
    streams = population[nsr:]

    for p in population:
        print(p.position)
        print("cost: " + str(p.cost))
    
###### Alocate number of streams for each sea and river START ######

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
    #print(NS)
    
    # Normalize NS to ensure the sum of streams allocated almost matches the total number of streams
    NS = (NS / NS.sum() * len(streams)).astype(int)
    #print(NS)

    ###### NS Modification START

    # Ensure total number of streams in NS don't exceed number of streams
    i = len(NS) # number of rivers + sea
    while(sum(NS) > len(streams)):
        if(NS[i-1] > 1):
            NS[i-1] = NS[i-1] - 1
        else:
            i -= 1
    #print(NS)

    # Ensure that the total number of streams matches the available streams, adding remaining streams to the sea
    if(sum(NS) < len(streams)):
        NS[0] = NS[0] + len(streams) - sum(NS)
    #print(NS)
    
    # Prevent any river or sea from having zero streams allocated
    trailing_zeros_start = np.count_nonzero(NS[::-1])  # Start index of trailing zeros

    if trailing_zeros_start > 0 and trailing_zeros_start < len(NS):  # Only proceed if there are trailing zeros
        for idx in range(trailing_zeros_start, len(NS)):  # Loop through trailing zeros
            while NS[idx] == 0:  # Fix each zero element
                for i in range(trailing_zeros_start - 1, -1, -1):  # Redistribute from earlier elements
                    if i >= 0 and i < len(NS):  
                        if NS[i] > 1:  # Ensure at least one stream remains in the source
                            redistribution = max(1, round(NS[i] / 6))  # Proportional redistribution
                            NS[idx] += redistribution 
                            NS[i] -= redistribution  
                            break  # Move to the next zero
    
    #print("No more rivers with zero streams allocated")
    #print(NS)
    
    NS = np.sort(NS)[::-1]
    #print(NS)

    ###### NS Modification END

    ###### Alocate number of streams for each sea and river END ######

print(wca(benchmark.sphere_func,-5,5,2))    