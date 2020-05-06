import matplotlib.pyplot as plt

malignant = 0.9784
benign = 0.6492
fig = plt.figure()
ax = fig.add_subplot(111)
ax.boxplot([malignant, benign])