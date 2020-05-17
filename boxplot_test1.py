import numpy as np
import matplotlib.pyplot as plt
import statistics

# Fixing random state for reproducibility
np.random.seed(19680801)

# fake up some data
maximum = [0.978, 0.968, 0.979, 0.967, 0.998, 0.936]
minimum = [0.478, 0.368, 0.379, 0.389, 0.487, 0.298]
median = [statistics.median([maximum[i], minimum[i]]) for i in range(len(maximum))]

data1 = np.concatenate((median, maximum, minimum))
data2 = [[maximum[i], median[i], minimum[i]] for i in range(len(maximum))]

fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()

ax1.set_title("Concatenated values")
ax2.set_title("One value per generation")

bp1 = ax1.boxplot(data1)
bp2 = ax2.boxplot(data2)

plt.show()