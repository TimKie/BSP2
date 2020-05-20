import numpy as np
import matplotlib.pyplot as plt
import statistics

# Fixing random state for reproducibility
np.random.seed(19680801)

# fake up some data
maximum = []
minimum = []
median = []
fitness_values_of_pop = dict()

fitness_values_of_pop["0"] = [0.978, 0.968, 0.979, 0.967, 0.998, 0.936, 0.478, 0.368, 0.379, 0.389, 0.487, 0.298]
fitness_values_of_pop["1"] = [0.945, 0.868, 0.779, 0.667, 0.898, 0.736, 0.278, 0.268, 0.379, 0.348, 0.287, 0.298]

for i in range(len(fitness_values_of_pop)):
    maximum.append(max(fitness_values_of_pop[str(i)]))
    minimum.append(min(fitness_values_of_pop[str(i)]))
    median.append(statistics.median(fitness_values_of_pop[str(i)]))


print(maximum)
print(minimum)
print(median)


fig1, ax1 = plt.subplots()

ax1.set_title("Concatenated values")

bp1 = ax1.boxplot([fitness_values_of_pop[str(i)] for i in range (len(fitness_values_of_pop))])

plt.show()