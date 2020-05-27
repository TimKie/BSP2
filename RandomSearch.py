from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from Neural_Network import NeuralNetwork
import random
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import statistics
random.seed(4444)

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
        fitness = fitness_stored[''.join(str(e) for e in list(individual.values()))]                # which I then convert into a string using list comprehension such that I can use this string as the
    else:                                                                                           # key where the corresponding value is the fitness value of the individual                                                          
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


# Main Code

popSize = 30
number_of_generations = 30
max_fitness_values = []
min_fitness_values = []
median_fitness_values = []
best_sets_of_hyper_para = []
fitness_values_of_pop = dict()


time_before_execution = datetime.now()
print("Current Time:", time_before_execution.strftime("%H:%M:%S"))

for i in range(number_of_generations):
    population = generate_population(popSize)

    fitness_values_of_gen = []
    sets_of_ind_of_pop = []
    for ind in population:
        fitness_values_of_gen.append(get_fitness(ind))
        sets_of_ind_of_pop.append(ind)

    fitness_values_of_pop[str(i)] = fitness_values_of_gen
    max_fitness_values.append(max(fitness_values_of_gen))
    min_fitness_values.append(min(fitness_values_of_gen))
    median_fitness_values.append(statistics.median(fitness_values_of_gen))

    # get the index of the individual with the best fitness value of this generation and using this index to find the
    # corresponding set of hyper-parameters in the list of all sets in the generation
    best_sets_of_hyper_para.append(sets_of_ind_of_pop[fitness_values_of_gen.index(max(fitness_values_of_gen))])

time_after_execution = datetime.now()
print("Current Time:", time_after_execution.strftime("%H:%M:%S"))

print("Best fitness values of individuals in each generation:", max_fitness_values)
print("Median fitness values of individuals in each generation:", median_fitness_values)
print("Worst fitness values of individuals in each generation:", min_fitness_values)
print("All fitness values that were computed:", list(fitness_stored.values()))
print()
print("The best fitness value of the last generation is:", max_fitness_values[-1], "with the corresponding set of hyper-parameters:", best_sets_of_hyper_para[-1])
print()
print("The best fitness value overall is:", max(max_fitness_values), "with the corresponding set of hyper-parameters:", best_sets_of_hyper_para[max_fitness_values.index(max(max_fitness_values))])
print()
print("Length of fitness_stored dictionary (number of computed fitness values):", len(fitness_stored))
print("The execution time is:", (time_after_execution - time_before_execution).total_seconds(), "seconds")


# plotting results
fig, ax1 = plt.subplots()

ax1.set_xlabel("Generations", color="r")
ax1.set_ylabel("Fitness values")
bp = ax1.boxplot([fitness_values_of_pop[str(i)] for i in range (len(fitness_values_of_pop))], boxprops=dict(color="red"), capprops=dict(color="red"), whiskerprops=dict(color="red"), medianprops=dict(color="black"))
ax1.tick_params(axis='x', labelcolor="r")
plt.xticks([i for i in range(1, number_of_generations+1)])

ax2 = ax1.twiny()

x_values_for_blue_plot = []
for i in range(1, number_of_generations+1):
    x_values_for_blue_plot += [i for j in range(1, popSize+1)]

fitness_values_of_all_pop = []
for i in range(number_of_generations):
    fitness_values_of_all_pop += fitness_values_of_pop[str(i)]

ax2.set_xlabel("Number of fitness values", color="b")
ax1.plot(x_values_for_blue_plot, fitness_values_of_all_pop, '.', label= "All fitness values", color="b")
ax2.tick_params(axis='x', labelcolor="b")
plt.xticks(np.arange(0, len(fitness_stored), 100))

plt.show()
