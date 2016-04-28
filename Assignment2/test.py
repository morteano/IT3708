import matplotlib.pyplot as plt
bestFitnesses = [76, 57, 57, 41, 49, 58, 98, 80, 91, 72, 95, 62, 95, 61, 103, 77, 61, 45, 92, 48, 65]
x = range(0, len(bestFitnesses))
plt.plot(x, bestFitnesses)
plt.ylabel('Fitness')
plt.xlabel('Generations')
plt.ylim([0, 25])
plt.show()