from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from Neural_Network import NeuralNetwork
import random
import matplotlib.pyplot as plt

activation_functions = ["sigmoid", "tanh", "relu", "softmax"]
number_of_neurons = [32, 64, 128, 256, 512, 1024]
optimizers = [tf.keras.optimizers.Adam, tf.keras.optimizers.SGD]
dropout_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
epoch_size = [2, 3, 4, 5, 6, 7, 8]
learning_rate = [0.1, 0.01, 0.001, 0.0001]
batch_size = [32, 64, 128, 256, 512]

def generate_individual():
    return {"activation_function": random.choice(activation_functions), "number_of_neurons" : random.choice(number_of_neurons),
            "optimizer": random.choice(optimizers), "dropout": random.choice(dropout_values), "epoch_size": random.choice(epoch_size),
            "learning_rate": random.choice(learning_rate), "batch_size": random.choice(batch_size)}


fitness_stored = dict()
def get_fitness(individual):
    if ''.join(str(e) for e in list(individual.values())) in fitness_stored:                        # I take the values of of one individual since they define the individual and convert them into a list
        fitness = fitness_stored[''.join(str(e) for e in list(individual.values()))]                # which I then convert into a string using list comprehension such that I can use this string as the key
    else:                                                                                           # where the corresponding value is the fitness value of the individual
        nn = NeuralNetwork(individual["activation_function"], individual["number_of_neurons"], individual["optimizer"], individual["dropout"], individual["epoch_size"], individual["learning_rate"], individual["batch_size"])
        fitness = nn.build()
        fitness_stored[''.join(str(e) for e in list(individual.values()))] = fitness                # I store the fitness value of the individual which is not already in the stored_fitness dictionary
    return fitness


def generate_population(popSize):
    pop = []
    for i in range(popSize):
        p = generate_individual()
        pop.append(p)
    return pop


def mutation(pop):
    mutated_pop = []
    for individual in pop:
        mutProb = random.uniform(0, 0.5)                                                                    # at most 50% of the genes can be changed because the mutation probability will be generated between 0% and 50%
        for i in range(1, len(individual)+1):                                                               # i will iterate from 1 to the len of the individual (number of hyper-parameters that could be changed)
            if (mutProb * len(individual)) < i:                                                             # i represents the number of genes (hyper-parameters) which will be changed
                n = i
                break
        for j in range(n):                                                                                  # we change as many genes as the values of i is (e.g. if =2 then 2 hyper-parameters will be changed)
            g = random.choice(list(individual.keys()))                                                      # we take one random gene (hyper-parameter) of the chromosome to change it
            if g == "activation_function":
                individual[g] = random.choice(activation_functions)
            if g == "number_of_neurons":
                individual[g] = random.choice(number_of_neurons)
            if g == "optimizer":
                individual[g] = random.choice(optimizers)
            if g == "dropout":
                individual[g] = random.choice(dropout_values)
            if g == "epoch_size":
                individual[g] = random.choice(epoch_size)
            if g == "learning_rate":
                individual[g] = random.choice(learning_rate)
            if g == "batch_size":
                individual[g] = random.choice(batch_size)
        mutated_pop.append(individual)
    return mutated_pop


def roulette_wheel_selection(pop, num_of_parents):
    parents = []
    fitness_sum = 0
    for elem in pop:
        fitness_sum += get_fitness(elem)                                                                    # get the fitness of the whole population (total fitness)
    fitness_of_individuals = [get_fitness(elem) / fitness_sum for elem in pop]                              # store the fitness of each individual divided by the total fitness in a list (number between 0 and 1)
    probabilities = [sum(fitness_of_individuals[:i + 1]) for i in range(len(fitness_of_individuals))]       # make a list with intervals where the individuals can be selected later (individuals with higher fitness have greater interval)
    for i in range(num_of_parents):
        r = random.random()                                                                                 # a number between 0 and 1
        for (n, individual) in enumerate(pop):                                                              # for every individual in the population (i is the index of the individual)
            if r <= probabilities[n]:                                                                       # the individual is selected when r is in the interval of the individual
                parents.append(individual)
                break                                                                                       # after one parent is found and added to the final list, we break the loop
    return parents


def crossover(individual1, individual2):
    crossPoint = random.randrange(1, len(individual1)+1)
    child1_part1 = dict(list(individual1.items())[0: crossPoint])               # part from individual 1 for child 1 (genes before crossPoint)
    child1_part2 = dict(list(individual2.items())[crossPoint::])                # part from individual 2 for child 1 (genes after crossPoint)
    child1 = {**child1_part1, **child1_part2}                                   # combining the two parts to create child 1

    child2_part1 = dict(list(individual2.items())[0: crossPoint])               # part from individual 2 for child 2 (genes before crossPoint)
    child2_part2 = dict(list(individual1.items())[crossPoint::])                # part from individual 1 for child 2 (genes after crossPoint)
    child2 = {**child2_part1, **child2_part2}                                   # combining the two parts to create child 2

    return [child1, child2]


def crossover_population(pop, cross_prob):
    new_pop = []
    num = int(len(pop) * cross_prob)
    for i in range(num // 2):
        parent1, parent2 = random.sample(pop, 2)
        new_pop += crossover(parent1, parent2)
        pop.remove(parent1)
        pop.remove(parent2)
    return pop + new_pop


"""for i in range(len(parents) // 2):                   # we take in range "len(pop) // 2" because there are always two elements removed
    pop.remove(parents[0])                              # we delete the two parents that we chose out of the population such that
    if parents[1] in pop:
        pop.remove(parents[1])                          # they cannot be selected again as parents (sometimes I get an error that parents[1] is not in the list)
    new_pop += crossover(parents[0], parents[1])        # add the offspring to the list which contains the new generation
    return pop + new_pop                                # combine the new children with the individuals that were not used as parents"""


# Main Code

popSize = 10
number_of_generations = 5
crossover_prob = 0.5
best_fitness_values = []

population = generate_population(popSize)

for i in range(number_of_generations):                                          # how often the following actions are repeated
    parents = roulette_wheel_selection(population, popSize)
    population = crossover_population(parents, crossover_prob)
    print("population after crossover", population)
    population = mutation(population)
    print("population after mutation:", population)
    fitness_values_of_pop = []
    for ind in population:
        fitness_values_of_pop.append(get_fitness(ind))
    best_fitness_values.append(max(fitness_values_of_pop))

print("Best fitness values of individuals in each generation:", best_fitness_values)
print("All fitness values that were computed:", list(fitness_stored.values()))

# plotting results
fig, ax1 = plt.subplots()

ax1.set_xlabel("Generations", color="r")
ax1.set_ylabel("Fitness values")
ax1.plot([i for i in range(1, len(best_fitness_values)+1)], best_fitness_values, label= "Best fitness values", color="r")
ax1.tick_params(axis='x', labelcolor="r")
plt.xticks([i for i in range(1, len(best_fitness_values)+1)])

ax2 = ax1.twiny()

ax2.set_xlabel("Number of fitness values", color="b")
ax2.plot([i for i in range(1, len(list(fitness_stored.values()))+1)], list(fitness_stored.values()), label= "All fitness values", color="b")
ax2.tick_params(axis='x', labelcolor="b")

plt.show()
