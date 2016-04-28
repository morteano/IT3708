import random
import math
import matplotlib.pyplot as plt


# GENERATIONS = 100
# GENO_LENGTH = 40
# populationSize = 20
# babyPopulationSize = 200
# Z = 21
# K = 20
# N = 10
# EPSILON = 0.1
# MUTATION_RATE = 0.05
# PROBLEM = 3
# ADULT_SELECTION = 3
# S = 2


# Generate a population of random genotypes
def randomGenotypes(length):
    genotype = []
    for i in range(length):
        genotype.append(random.randint(0, S-1))
    return genotype


def convertToPhenotypes(population):
    phenotypes = []
    for genotype in population:
        phenotypes.append(genotype)
    return phenotypes


# FITNESS
def valuateFitness(genotype):
    if PROBLEM == 1:
        return valuateFitnessOneMax(genotype)
    elif PROBLEM == 2:
        return valuateFitnessLOLZ(genotype, Z)
    elif PROBLEM == 3:
        return valuateFitnessSuprisingGlobal(genotype)
    elif PROBLEM == 4:
        return valuateFitnessSuprisingLocal(genotype)


def valuateFitnessOneMax(genotype):
    return sum(genotype)


def valuateFitnessOneMaxTest(genotype):
    best = [1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0]
    fitness = 0
    for i in range(len(genotype)):
        if genotype[i] == best[i]:
            fitness += 1
    return fitness

def valuateFitnessLOLZ(genotype, z):
    fitness = 1
    i = 1
    while genotype[i] == genotype[i-1]:
        if genotype[i] == 0 and fitness == z:
            return fitness
        fitness += 1
        i += 1
        if i == GENO_LENGTH:
            return fitness
    return fitness


def valuateFitnessSuprisingGlobal(genotype):
    unsuprises = findUnsuprisesGlobal(genotype)
    return 100-3*unsuprises


def valuateFitnessSuprisingLocal(genotype):
    unsuprises = findUnsuprisesLocal(genotype)
    return 1000-unsuprises


def getTotalFitness(population):
    totalFitness = 0
    for genotype in population:
        totalFitness += valuateFitness(genotype)
    return totalFitness


# TOURNAMENT
def tournament(population):
    parents = []
    for i in range(populationSize):
        contestants = []
        indexes = list(range(0, babyPopulationSize))
        for j in range(K):
            contestants.append(population[indexes.pop(random.randint(0, len(indexes)-1))])
        if random.random() > EPSILON:
            parents.append(contestants[findBestContestant(contestants)])
        else:
            parents.append(contestants[random.randint(0, len(contestants)-1)])
    return parents


def findBestContestant(contestants):
    bestFitness = 0
    index = 0
    for i in range(len(contestants)):
        if valuateFitness(contestants[i]) > bestFitness:
            bestFitness = valuateFitness(contestants[i])
            index = i
    return index


# Fitness proportionate
def fitnessProportionate(population):
    parents = []
    for i in range(populationSize):
        totalFitness = getTotalFitness(population)
        rand = random.random()
        fitness = 0
        i = -1
        while rand > fitness:
            i += 1
            fitness += valuateFitness(population[i])/totalFitness
        parents.append(population[i])
    return parents


# SIGMA SCALING
def sigmaScaling(population):
    parents = []
    avgFitness = getTotalFitness(population)/len(population)
    sd = getStandardDeviation(population, avgFitness)
    totalExpVal = getTotalExpVal(population, avgFitness, sd)
    for i in range(populationSize):
        rand = random.random()
        fitness = 0
        i = -1
        while rand > fitness:
            i += 1
            if sd == 0:
                fitness += 1
            else:
                fitness += (1 + (valuateFitness(population[i])-avgFitness)/(2*sd))/totalExpVal
        parents.append(population[i])
    return parents


def getTotalExpVal(population, avgFitness, sd):
    totalExpVal = 0
    for geno in population:
        if sd == 0:
            totalExpVal += 1
        else:
            totalExpVal += (1 + (valuateFitness(geno)-avgFitness)/(2*sd))
    return totalExpVal


def getStandardDeviation(population, avgFitness):
    sd = 0
    for geno in population:
        sd += (valuateFitness(geno)-avgFitness)**2
    return math.sqrt(sd/len(population))




def adultSelection(oldAdults, youngAdults):
    if (ADULT_SELECTION == 1):
        adults = youngAdults
    elif (ADULT_SELECTION == 2):
        adults = tournament(youngAdults)
    elif (ADULT_SELECTION == 3):
        adults = tournament(youngAdults + oldAdults)
    return adults


def makingBabies(adults):
    babies = []
    adults = fitnessProportionate(adults)
    while len(babies) < babyPopulationSize:
        parent1 = adults[random.randint(0, len(adults)-1)]
        parent2 = adults[random.randint(0, len(adults)-1)]
        baby = []
        for i in range(GENO_LENGTH):
            if random.random() < 0.6:
                baby.append(parent1[i])
            else:
                baby.append(parent2[i])
                temp = parent1
                parent1 = parent2
                parent2 = temp
        babies.append(baby)
    return babies


def makingBabies2(adults):
    babies = []
    adults = sigmaScaling(adults)
    while len(babies) < babyPopulationSize:
        parent1 = adults[random.randint(0, len(adults)-1)]
        parent2 = adults[random.randint(0, len(adults)-1)]
        baby = []
        for i in range(GENO_LENGTH):
            if random.random() < 0.5:
                baby.append(parent1[i])
            else:
                baby.append(parent2[i])
        babies.append(baby)
    return babies


def mutation(genotype):
    for i in range(GENO_LENGTH):
        if random.random() < MUTATION_RATE:
            genotype[i] = random.randint(0, S-1)


def mutateBabies(population):
    for genotype in population:
        mutation(genotype)

def averageFitness(population):
    totalFitness = 0
    for element in population:
        totalFitness += valuateFitness(element)
    return totalFitness/len(population)


def printData(population, i, bestFitnesses, averageFitnesses, SDs):
    best = population[findBestContestant(population)]
    bestValue = valuateFitness(best)
    avg = averageFitness(population)
    sd = getStandardDeviation(population, avg)
    print("Generation number:", i)
    print("Best:", valuateFitness(best))
    print("Average:", avg)
    print("SD", sd)
    print("Best sequel:", best)

    bestFitnesses.append(bestValue)
    averageFitnesses.append(avg)
    SDs.append(sd)
    return bestFitnesses, averageFitnesses, SDs


def findUnsuprisesGlobal(genotype):
    dict = {}
    for i in range(S):
        dict[i] = []
    for i in range(len(genotype)):
        dict[genotype[i]].append(i)
    list = []
    for i in range(S):
        for j in range(len(dict[i])):
            for k in range(j+1, len(dict[i])):
                list.append(dict[i][k]-dict[i][j])
    list = sorted(list)
    unsuprises = 0
    for i in range(1, len(list)):
        if list[i] == list[i-1]:
            unsuprises += 1
    return unsuprises


def findUnsuprisesLocal(genotype):
    unsuprises = 0
    for i in range(1, len(genotype)-2):
        first = genotype[i-1]
        second = genotype[i]
        for j in range(i, len(genotype)-1):
            if genotype[j] == first:
                if genotype[j+1] == second:
                    unsuprises += 1
    return unsuprises

ADULT_SELECTION = 1
while ADULT_SELECTION != 0:
    ADULT_SELECTION = int(input("Select adult selection (1:Full, 2:Over-production, 3:Mixing): "))
    if ADULT_SELECTION == 0:
        break
    if ADULT_SELECTION == 4:
        ADULT_SELECTION = 3
        PROBLEM = 1
        S = 2
        GENO_LENGTH = 40
        GENERATIONS = 100
        populationSize = 20
        babyPopulationSize = 20
        Z = 21
        K = 20
        N = 20
        EPSILON = 0.1
        MUTATION_RATE = 0.002
    else:
        PROBLEM = int(input("Enter problem (1:OneMAX, 2: LOLZ, 3: Global suprising, 4: Local suprising): "))
        S = int(input("Enter S: "))
        GENO_LENGTH = int(input("Enter L: "))
        GENERATIONS = int(input("Enter max generations: "))
        if GENERATIONS == 0:
            GENERATIONS = 100
            populationSize = 20
            babyPopulationSize = 200
            Z = 21
            K = 20
            N = 10
            EPSILON = 0.1
            MUTATION_RATE = 0.002
        else:
            populationSize = int(input("Enter number of potential adults: "))
            babyPopulationSize = int(input("Enter number of babies: "))
            Z = int(input("Enter Z: "))
            K = int(input("Enter K: "))
            EPSILON = float(input("Enter epsilon: "))
            MUTATION_RATE = float(input("Enter mutation rate: "))

    population = []
    adults = []
    bestFitnesses = []
    averageFitnesses = []
    SDs = []
    for i in range(babyPopulationSize):
        population.append(randomGenotypes(GENO_LENGTH))
    for i in range(GENERATIONS):
        adults = adultSelection(adults, population)
        population = makingBabies2(adults)
        mutateBabies(population)
        bestFitnesses, averageFitnesses, SDs = printData(population, i, bestFitnesses, averageFitnesses, SDs)
        if bestFitnesses[i] == 100:
            break
        # elif bestFitnesses[i] == 40 and (PROBLEM == 1 or PROBLEM == 2) and GENO_LENGTH == 40:
        #     break
    printData(population, i, bestFitnesses, averageFitnesses, SDs)
    x = range(0, len(bestFitnesses))
    plt.plot(x, bestFitnesses)
    plt.plot(x, averageFitnesses, 'r')
    plt.ylabel('Fitness')
    plt.xlabel('Generations')
    plt.show()

    plt.plot(x, SDs, 'g')
    plt.ylabel('Standard deviation')
    plt.xlabel('Generations')
    plt.show()

# geno = randomGenotypes(GENO_LENGTH)
# findUnsuprises(geno)
# print(geno)