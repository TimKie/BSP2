import random

def generate_individual():
    activation_functions = ["Sigmoid", "TanH", "ReLU", "Softmax"]
    number_of_neurons = []          # depends on the input of the NN
    optimizers = ["Gradient Descent", "Adam"]
    dropout_values = [0.5, 0.6, 0.7, 0.8]
    final_set = {"activation_function": random.choice(activation_functions), "optimizer": random.choice(optimizers), "dropout": random.choice(dropout_values)}
    return final_set


# Test Code
print(generate_individual())