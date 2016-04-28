import openpyxl as px
import numpy as np
from random import randint
import math
import matplotlib.pyplot as plt


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


def main():
    data = Data()
    population = Population(data, 100)
    distEa, costEa = population.bestFitness()
    print(distEa.dist, distEa.cost)
    print(costEa.dist, costEa.cost)
    population.plotPopulation()

main()