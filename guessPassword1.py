import random
import datetime
import random

geneSet = " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!."
target = "Hello World!"


def generate_parent(length):
    genes = []
    while len(genes) < length:
        sampleSize = min(length - len(genes), len(geneSet))
        genes.extend(random.sample(geneSet, sampleSize))
    return ''.join(genes)


def generate_population(popSize, length):
    pop = []
    for i in range(popSize):
        p = generate_parent(length)
        pop.append(p)
    return pop


def get_fitness(guess):
    return sum(1 for expected, actual in zip(target, guess) if expected == actual)


def crossover_parents(parent1, parent2):
    crossover_point = random.randrange(0, len(parent1)+1)
    if crossover_point == 0 or crossover_point == len(parent1) or crossover_point == len(parent2):          # if the crossover point is 0 or len(parent1) or len(parent2), a new crossover point is generated
        crossover_point = random.randrange(0, len(parent1) + 1)
    print("Crossover Point:", crossover_point)                                                              # print the crossover point to the console to check if the function works correctly
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2


def mutate(parent):
    index = random.randrange(0, len(parent))
    childGenes = list(parent)
    newGene, alternate = random.sample(geneSet, 2)
    childGenes[index] = alternate if newGene == childGenes[index] else newGene
    return ''.join(childGenes)


def display(guess):
    timeDiff = datetime.datetime.now() - startTime
    fitness = get_fitness(guess)
    print("{}\t{}\t{}".format(guess, fitness, timeDiff))


# Main Program
random.seed()
startTime = datetime.datetime.now()
bestParent = generate_parent(len(target))
bestFitness = get_fitness(bestParent)
display(bestParent)

while True:
    child = mutate(bestParent)
    childFitness = get_fitness(child)
    if bestFitness >= childFitness:
        continue
    display(child)
    if childFitness >= len(bestParent):
        break
    bestFitness = childFitness
    bestParent = child


print()

# ------------------------ Testing the two new functions -------------------------------------------------------

popSize = 8
p = generate_population(popSize, len(target))
print("Population:", p)
p1 = p[random.randrange(0, popSize)]
p2 = p[random.randrange(0, popSize)]
# we generate children with two randomly picked parents from the population
print("Children:", crossover_parents(p1, p2), "from parent1 "+"\'"+p1+"\'"+" and parent2 "+"\'"+p2+"\'")