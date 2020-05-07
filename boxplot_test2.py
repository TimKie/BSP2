import matplotlib.pyplot as plt
import numpy as np

max_fitness_values = [0.9751, 0.9715]
median_fitness_values = [0.822950005531311, 0.9053999781608582]
min_fitness_values = [0.6708, 0.8393]

# plotting results
fig, ax1 = plt.subplots()

ax1.set_xlabel("Generations", color="r")
ax1.set_ylabel("Fitness values")
bp = ax1.boxplot([[max_fitness_values[i], median_fitness_values[i], min_fitness_values[i]] for i in range(2)], boxprops=dict(color="red"), capprops=dict(color="red"), whiskerprops=dict(color="red"), medianprops=dict(color="black"))
ax1.tick_params(axis='x', labelcolor="r")
plt.xticks([i for i in range(1, 3+1)])

ax2 = ax1.twiny()

ax2.set_xlabel("Number of fitness values", color="b")
ax2.plot([1, 2], [0.45, 0.76], '.', label= "All fitness values", color="b")
ax2.tick_params(axis='x', labelcolor="b")

plt.show()


