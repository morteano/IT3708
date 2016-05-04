import openpyxl as px
import numpy as np
from random import randint, random
import math
import matplotlib.pyplot as plt
import operator

# Global variables
N = 100
TOURNAMENT_SIZE = 50
GENERATIONS = 100
EPSILON = 0.99
MUTATION_RATE = 0.01


# a class containing the cost and distance matrices
class Data:
    def __init__(self):
        self.cost, self.dist = self.readData()

    def readData(self):
        cost = self.readFile('Cost.xlsx')
        dist = self.readFile('Distance.xlsx')
        return cost, dist

    def readFile(self, filename):
        # read data
        W = px.load_workbook(filename, use_iterators = True)
        p = W.get_sheet_by_name(name = 'Sheet1')

        # store data in a list
        a=[]
        for row in p.iter_rows():
            for k in row:
                a.append(k.internal_value)

        # convert list a to a 48x48 matrix
        aa = np.resize(a, [49, 49])
        aa = np.delete(aa, 0, 0)
        aa = np.delete(aa, 0, 1)
        return aa


class EaType:
    def __init__(self, data):
        self.genotype = self.makeGenotype()
        self.phenotype = self.genotype
        self.dist, self.cost = self.fitness(data)
        self.rank = None
        self.n = None
        self.dominates = []
        self.crowdDist = 0

    def makeGenotype(self):
        genotype = []
        cities = list(range(0, 48))
        for i in range(48):
            genotype.append(cities.pop(randint(0, 47-i)))
        genotype.append(genotype[0])
        return genotype

    def fitness(self, data):
        tourDist = 0
        tourCost = 0
        for i in range(len(self.phenotype)-1):
            tourDist += data.dist[max(self.phenotype[i], self.phenotype[i+1])][min(self.phenotype[i], self.phenotype[i+1])]
            tourCost += data.cost[max(self.phenotype[i], self.phenotype[i+1])][min(self.phenotype[i], self.phenotype[i+1])]
        return tourDist, tourCost

    def mutation(self):
        for i in range(1, len(self.genotype)-1):
            if random() < MUTATION_RATE:
                index = randint(1, len(self.genotype)-2)
                temp = self.genotype[i]
                self.genotype[i] = self.genotype[index]
                self.genotype[index] = temp

    def isDominating(self, otherEaType):
        # TODO fix this function!!!!!
        if self.dist <= otherEaType.dist and self.cost <= otherEaType.cost:
            if self.dist < otherEaType.dist or self.cost < otherEaType.cost:
                return True
        return False


class Population:
    def __init__(self, data, size):
        self.population = []
        for i in range(size):
            self.population.append(EaType(data))

    def bestFitness(self):
        bestDist = float("inf")
        bestCost = float("inf")
        bestDistEa = None
        for eaType in self.population:
            if eaType.dist < bestDist:
                bestDist = eaType.dist
                bestDistEa = eaType
            if eaType.cost < bestCost:
                bestCost = eaType.cost
                bestCostEa = eaType
        return bestDistEa, bestCostEa

    def plotPopulation(self, end=False):
        plt.axis([0, 180000, 0, 2000])
        for eaType in self.population:
            if self.isParetoOptimal(eaType):
                plt.plot(eaType.dist, eaType.cost, 'ro')
            else:
                plt.plot(eaType.dist, eaType.cost, 'bo')
        plt.xlabel('Distance')
        plt.ylabel('Cost')
        plt.draw()
        plt.pause(0.001)
        plt.plot()
        plt.show()
        if end:
            input("Press [enter] to end.")
        plt.clf()

        # plt.xlim([0, 10])
        # plt.ylim([0, 10])
        # plt.grid()
        # plt.show(block=False)

    def isParetoOptimal(self, testEaType):
        for eaType in self.population:
            if eaType.dist < testEaType.dist and eaType.cost < testEaType.cost:
                return False
        return True

    def nonDomSort(self):
        F = []
        F0 = []
        for eaType in self.population:
            eaType.dominates = []
            eaType.n = 0
            for otherEaType in self.population:
                if eaType.isDominating(otherEaType):
                        eaType.dominates.append(otherEaType)
                elif otherEaType.isDominating(eaType):
                    eaType.n += 1
            if eaType.n == 0:
                eaType.rank = 1
                F0.append(eaType)
        F.append(F0)
        i = 1
        while len(F[i-1]) > 0:
            Q = []
            for eaType in F[i-1]:
                for otherEaType in eaType.dominates:
                    otherEaType.n -= 1
                    if otherEaType.n == 0:
                        otherEaType.rank = i + 1
                        Q.append(otherEaType)
            i += 1
            F.append(Q)
        F.pop(-1)
        return F

    def crowdingDist(self, I):
        for eaType in I:
            eaType.crowdDist = 0
        for obj in ['dist', 'cost']:
            sortedF = sorted(I, key=operator.attrgetter(obj))
            sortedF[0].crowdDist = float("inf")
            sortedF[-1].crowdDist = float("inf")
            for i in range(1, len(I)-1):
                if obj == 'dist':
                    sortedF[i].crowdDist += (sortedF[i+1].dist-sortedF[i-1].dist)/(sortedF[-1].dist-sortedF[0].dist)
                elif obj == 'cost':
                    sortedF[i].crowdDist += (sortedF[i+1].cost-sortedF[i-1].cost)/(sortedF[-1].cost-sortedF[0].cost)

    def makeNewPop(self, data):
        nextGen = []
        for i in range(N):
            parent1 = self.tournament()
            parent2 = self.tournament()
            baby = EaType(data)
            baby.genotype = []
            for i in range(randint(1, len(parent1.genotype)-2)):
                baby.genotype.append(parent1.genotype[i])
            i = 0
            while len(baby.genotype) < len(parent2.genotype)-1:
                if i == 49:
                    print(i, len(parent2.genotype), len(baby.genotype))
                    print(parent2.genotype)
                    print(baby.genotype)
                if parent2.genotype[i] not in baby.genotype:
                    baby.genotype.append(parent2.genotype[i])
                i += 1
            baby.genotype.append(baby.genotype[0])
            baby.mutation()
            baby.phenotype = baby.genotype
            nextGen.append(baby)
        return nextGen

    def tournament(self):
        contestants = []
        for i in range(TOURNAMENT_SIZE):
            contestants.append(self.population[randint(0, N-1)])
        if random() > EPSILON:
            return contestants[randint(0, TOURNAMENT_SIZE-1)]
        else:
            return getFittest(contestants)


def getFittest(contestants):
    best = contestants[0]
    # finds the eaTypes with best rank
    bestRank = float("inf")
    for eaType in contestants:
        if eaType.rank < bestRank:
            bestRank = eaType.rank
            elite = []
            elite.append(eaType)
        elif eaType.rank == bestRank:
            elite.append(eaType)
    # finds the biggest crowd distance given the best rank
    bestCrowdDist = 0
    for eaType in elite:
        if eaType.crowdDist > bestCrowdDist:
            bestCrowdDist = eaType.crowdDist
            best = eaType
    return best


def printAsVectors(F):
    xString = "plotx = ["
    yString = "ploty = ["
    for i in range(len(F)):
        xString += str(F[i].dist)
        yString += str(F[i].cost)
        if i == len(F)-1:
            xString += "]"
            yString += "]"
        else:
            xString += ", "
            yString += ", "
    print(xString)
    print(yString)

def main():
    plt.ion()
    plt.show()
    data = Data()
    population = Population(data, N)
    population.nonDomSort()
    nextGen = population.makeNewPop(data)
    for generation in range(GENERATIONS):
        population.plotPopulation()
        population.population += nextGen
        F = population.nonDomSort()
        population = Population(data, 0)
        i = 1
        while len(population.population)+len(F[i-1]) < N:
            population.crowdingDist(F[i-1])
            population.population += F[i-1]
            i += 1
        sortedF = sorted(F[i-1], key=operator.attrgetter('crowdDist'))
        j = 0
        while len(population.population) < N:
            population.population.append(sortedF[j])
            j += 1
        # if generation%10 == 0:
        # population.plotPopulation()
        nextGen = population.makeNewPop(data)
        if generation % 10 == 0:
            print("Generation:", generation, "of", GENERATIONS)
    # F = population.nonDomSort()
    # sortedF = sorted(F[0], key=operator.attrgetter('dist'))
    # printAsVectors(sortedF)
    population.population += nextGen
    population.plotPopulation(True)



main()