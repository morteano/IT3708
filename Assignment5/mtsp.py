import openpyxl as px
import numpy as np
from random import randint
import math
import matplotlib.pyplot as plt
import operator

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

    def plotPopulation(self):
        fig = plt.gcf()
        for eaType in self.population:
            if self.isParetoOptimal(eaType):
                plt.plot(eaType.dist, eaType.cost, 'ro')
            else:
                plt.plot(eaType.dist, eaType.cost, 'bo')
        plt.xlabel('Distance')
        plt.ylabel('Cost')
        # plt.xlim([0, 10])
        # plt.ylim([0, 10])
        plt.grid()
        plt.show()

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
                if eaType.dist < otherEaType.dist and eaType.cost < otherEaType.cost:
                    eaType.dominates.append(otherEaType)
                elif eaType.dist > otherEaType.dist and eaType.cost > otherEaType.cost:
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


def main():
    data = Data()
    population = Population(data, 10)
    distEa, costEa = population.bestFitness()
    print(distEa.dist, distEa.cost)
    print(costEa.dist, costEa.cost)
    F = population.nonDomSort()
    print("Before")
    for type in F[0]:
        print(type.crowdDist)
    population.crowdingDist(F)
    print("After")
    for type in F[0]:
        print(type.crowdDist)
    population.plotPopulation()

def tempMain():
    N = 10
    data = Data()
    population = Population(data, N)
    nextGen = Population(data, N)
    for i in range(10):
        population.population += nextGen.population
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
        population.plotPopulation()
        nextGen = Population(data, N)

tempMain()