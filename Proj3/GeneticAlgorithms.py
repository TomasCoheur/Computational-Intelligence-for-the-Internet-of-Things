import random

from deap import base
from deap import creator
from deap import tools
import pandas as pd
import time

start_time = time.perf_counter()
ecopoints_to_visit = pd.read_csv("ListEcoPoints.csv")
filename = "Project2_DistancesMatrix.xlsx"
distance_file = pd.read_excel(filename, index_col=[0])
list_of_ecopoints = ecopoints_to_visit["EcoPoints"].values.tolist()
list_of_ecopoints.sort()
nr_ecopoints_to_visit = len(list_of_ecopoints)
list_of_indexes = list(range(0, nr_ecopoints_to_visit))
first_ecopoint = 'C'
counter = 0


def shuffle_list():     # shuffle the ecopoints list to have different individuals
    random.shuffle(list_of_indexes)
    return list_of_ecopoints


def get_ecopoint():     # returns the next ecopoint from the list
    global counter  # counter to know when the individual has all ecopoints in its list
    if counter == nr_ecopoints_to_visit:    # we have to create a new individual
        shuffle_list()
        counter = 1
        return list_of_indexes[0]
    counter += 1
    return list_of_indexes[counter-1]


creator.create("FitnessMin", base.Fitness, weights=(-1.0,))     # we want the smallest value that our algorithm provides
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
# Attribute generator
toolbox.register("distances", get_ecopoint)
# Structure initializers
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.distances, nr_ecopoints_to_visit) # individual is a list of ecopoints
toolbox.register("population", tools.initRepeat, list, toolbox.individual)  # population is a list of individuals


def get_distance(actual_ecopoint, next_ecopoint):
    return distance_file[actual_ecopoint][next_ecopoint]


# ir buscar os valores das distancias e somar as distancias
def eval_distance(individual):
    actual_ecopoint = first_ecopoint
    distance = 0
    for eco in individual:
        next_ecopoint = list_of_ecopoints[eco]
        distance += get_distance(actual_ecopoint, next_ecopoint)
        actual_ecopoint = next_ecopoint
    distance += get_distance(actual_ecopoint, first_ecopoint)
    return int(distance),


toolbox.register("evaluate", eval_distance)
toolbox.register("mate", tools.cxOrdered)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)

def main():

    pop = toolbox.population(n=1000)
    # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    # CXPB  is the probability with which two individuals
    #       are crossed
    #
    # MUTPB is the probability for mutating an individual
    CXPB, MUTPB = 0.5, 0.2

    # Extracting all the fitnesses of
    fits = [ind.fitness.values[0] for ind in pop]

    # Variable keeping track of the number of generations
    g = 0

    # Begin the evolution
    while g < 500:
        # A new generation
        g = g + 1
        print("-- Generation %i --" % g)
        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        #print(offspring)
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))
        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        #print(offspring)
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        pop[:] = offspring

        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in pop]

        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x*x for x in fits)
        std = abs(sum2 / length - mean**2)**0.5

        print("  Min %s" % min(fits))
        print("  Max %s" % max(fits))
        print("  Avg %s" % mean)
        print("  Std %s" % std)

main()
end_time = time.perf_counter()
print("Time calculating: ", end_time - start_time, "seconds")