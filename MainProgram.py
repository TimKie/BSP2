from __future__ import absolute_import, division, print_function, unicode_literals
from Neural_Network import NeuralNetwork
import random
import itertools

activation_functions = ["sigmoid", "tanh", "relu", "softmax"]
number_of_neurons = [128]  # depends on the input of the NN
optimizers = ["adam"]
dropout_values = [0.2, 0.3, 0.4, 0.5]


def generate_individual():
    final_set = {"activation_function": random.choice(activation_functions), "number_of_neurons" : random.choice(number_of_neurons), "optimizer": random.choice(optimizers), "dropout": random.choice(dropout_values)}
    return final_set


def get_fitness(individual):
    nn = NeuralNetwork(individual["activation_function"], individual["number_of_neurons"], individual["optimizer"], individual["dropout"])
    fitness = nn.build()
    return fitness

# With this approach, we receive a fitness value between 0 and 1, thus we don't have to calculate the fitness of the whole population to then afterwards
# calculate the fitness of each individual between 0 and 1. (we can remove lines 56 and 57 in "guessPassword1" and modify line 58 such that we can store
# directly the fitness score received by this function ("get_fitness") without calculating anything)


def generate_population(popSize):
    pop = []
    for i in range(popSize):
        p = generate_individual()
        pop.append(p)
    return pop


def mutation(pop):
    for individual in pop:
        mutProb = random.uniform(0, 0.5)                                # at most 50% of the genes can be changed because the mutation probability will be generated between 0% and 50%
        for i in range(1,len(individual)+1):                            # i will iterate for 1 to the len of the individual (number of hyper-parameters that could be changed)
            if (mutProb * len(individual)) < i:                         # i represents the number of genes (hyper-parameters) which will be changed
                for j in range(i):                                      # we change as many genes as the values of i is (e.g. if i=2 then 2 hyper-parameters will be changed)
                    g = random.choice(list(individual.keys()))          # we take one random gene (hyper-parameter) of the chromosome to change it 
                    if g == "activation_function":
                        individual[g] = random.choice(activation_functions)
                    if g == "number_of_neurons":
                        individual[g] = random.choice(number_of_neurons)
                    if g == "optimizer":
                        individual[g] = random.choice(optimizers)
                    if g == "dropout":
                        individual[g] = random.choice(dropout_values)
                print(individual)
                break


def crossover(individual1, individual2):
    crossPoint = random.randrange(1, len(individual1)+1)
    child1_part1 = dict(list(individual1.items())[0: crossPoint])               # part from individual 1 for child 1 (genes before crossPoint)
    child1_part2 = dict(list(individual2.items())[crossPoint::])                # part from individual 2 for child 1 (genes after crossPoint)
    child1 = {**child1_part1, **child1_part2}                                   # combining the two parts to create child 1

    child2_part1 = dict(list(individual2.items())[0: crossPoint])               # part from individual 2 for child 2 (genes before crossPoint)
    child2_part2 = dict(list(individual1.items())[crossPoint::])                # part from individual 1 for child 2 (genes after crossPoint)
    child2 = {**child2_part1, **child2_part2}                                   # combining the two parts to create child 2
    
    return [child1, child2]



# Test Code

pop_size = 2
p = generate_population(pop_size)
total_fit = []
"""for ind in p:
    total_fit.append(get_fitness(ind))
print("The fitness values of the individuals in the population are:",total_fit)"""

print("population before mutation:")
for ind in p:
    print(ind)
    
print()
print("population after mutation:")
mutation(p)                                         # often we don't see what has changed because we don't have much hyper-parameters at the moment
                                                    # and each hyper-parameter doesn't have a lot of different values, thus the function often changes
                                                    # the values of a hyper-parameter to the same one again
print()

for i in range(0, len(p), 2):
    ind1 = p[i]
    ind2 = p[i+1]
    print("Individual 1:", ind1)
    print("Individual 2:", ind2)
    print("children:", crossover(ind1, ind2))
    