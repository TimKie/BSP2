from __future__ import absolute_import, division, print_function, unicode_literals
from Neural_Network import NeuralNetwork
import random


def generate_individual():
    activation_functions = ["sigmoid", "tanh", "relu", "softmax"]
    number_of_neurons = [128]          # depends on the input of the NN
    optimizers = ["adam"]
    dropout_values = [0.5, 0.6, 0.7, 0.8]
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


# Test Code

pop_size = 2
p = generate_population(pop_size)
total_fit = []
for ind in p:
    total_fit.append(get_fitness(ind))
print("The fitness values of the individuals in the population are:",total_fit)
