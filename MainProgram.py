from __future__ import absolute_import, division, print_function, unicode_literals
from Neural_Network import NeuralNetwork
import random


def generate_individual():
    activation_functions = ["Sigmoid", "TanH", "ReLU", "Softmax"]
    number_of_neurons = []          # depends on the input of the NN
    optimizers = ["Gradient Descent", "Adam"]
    dropout_values = [0.5, 0.6, 0.7, 0.8]
    final_set = {"activation_function": random.choice(activation_functions), "number_of_neurons" : number_of_neurons, "optimizer": random.choice(optimizers), "dropout": random.choice(dropout_values)}
    return final_set


def get_fitness(individual):
    nn = NeuralNetwork(individual["activation_function"], individual["number_of_neurons"], individual["optimizer"], individual["dropout"])
    fitness = nn.build()
    return fitness

# With this approach, we receive a fitness value between 0 and 1, thus we don't have to calculate the fitness of the whole population to then afterwards
# calculate the fitness of each individual between 0 and 1. (we can remove lines 56 and 57 in "guessPassword1" and modify line 58 such that we can store
# directly the fitness score received by this function ("get_fitness") without calculating anything)


# Test Code
print(generate_individual())

# ------------------------------------------------------

ind = {"activation_function": "relu", "number_of_neurons" : 128, "optimizer": "adam", "dropout": 0.2}       # to test the same nn as in the digit recognition example (same hyper-parameters)
print("The fitness value of the individual",ind,"is:",get_fitness(ind))