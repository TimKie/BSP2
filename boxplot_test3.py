

fitness_values_of_pop = [0.8445000052452087, 0.9455000162124634, 0.9325000047683716, 0.5526000261306763, 0.9332000017166138, 0.18520000576972961, 0.8920000195503235, 0.9247000217437744, 0.9456999897956848, 0.9416000247001648]

# plotting results
fig, ax1 = plt.subplots()

ax1.set_xlabel("Generations", color="r")
ax1.set_ylabel("Fitness values")
bp = ax1.boxplot(fitness_values_of_pop, boxprops=dict(color="red"), capprops=dict(color="red"), whiskerprops=dict(color="red"), medianprops=dict(color="black"))
ax1.tick_params(axis='x', labelcolor="r")
plt.xticks([i for i in range(1, len(max_fitness_values)+1)])

ax2 = ax1.twiny()

ax2.set_xlabel("Number of fitness values", color="b")
ax2.plot([i for i in range(1, len(list(fitness_stored.values()))+1)], list(fitness_stored.values()), '.', label= "All fitness values", color="b")
ax2.tick_params(axis='x', labelcolor="b")

plt.show()