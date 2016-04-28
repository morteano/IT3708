import random
import math
import copy
import pygame
import sys
import time
from graphics import *
x = 100
y = 100
import os
os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (x,y)

white = (255, 255, 255)
red = (255, 0, 0)
green = (0, 255, 0)
blue = (0, 0, 255)

WINDOW_WIDTH = 1000
WINDOW_HEIGHT = 500
LAYERS = [5, 2, 2]
WEIGHT_BITS = 8
FLATLAND_SIZE = 10
INPUT_SIZE = 6
GENO_LENGTH = 0
for i in range(len(LAYERS)-1):
    GENO_LENGTH += LAYERS[i]*LAYERS[i+1]
GENO_LENGTH *= WEIGHT_BITS
# For weights inside a layer
GENO_LENGTH += 4*2*WEIGHT_BITS
# For bias, gains and time constants
GENO_LENGTH += 12*WEIGHT_BITS
babyPopulationSize = 500
populationSize = 20
GENERATIONS = 1
ADULT_SELECTION = 3
K = 20
EPSILON = 0.001
PROBLEM = 5
MUTATION_RATE = 0.05
S = 2
treshold = 0
LAND_HEIGHT = 15
LAND_WIDTH = 30


class Tracker:
    def __init__(self):
        self.width = 5
        self.x = random.randint(0, LAND_WIDTH-self.width-1)


class Object:
    def __init__(self, x, objectSize):
        self.x = x
        self.y = 0
        self.objectSize = objectSize


class Beerland:
    def __init__(self):
        self.board = []
        for i in range(LAND_HEIGHT):
            line = []
            for j in range(LAND_WIDTH):
                line.append(0)
            self.board.append(line)

    def printBeerland(self):
        for x in range(LAND_HEIGHT):
            line = []
            for y in range(LAND_WIDTH):
                line.append(self.board[x][y])
            print(line)

    def spawnObject(self):
        objectSize = random.randint(1, 6)
        x = random.randint(0, LAND_WIDTH-objectSize)
        self.object = Object(x, objectSize)

    def spawnTracker(self):
        self.tracker = Tracker()
        for i in range(self.tracker.width):
            self.board[LAND_HEIGHT-1][self.tracker.x+i] = 2

    def getSensorData(self):
        output = []
        for i in range(self.tracker.width):
            object = 0
            for j in range(LAND_HEIGHT-1):
                if (self.board[j][(self.tracker.x+i)%15] == 1):
                    object = 1
            output.append(object)
        return output

    def move(self, movement):
        for i in range(self.tracker.width):
            self.board[LAND_HEIGHT-1][(self.tracker.x+i)] = 0
        if movement > 4:
            self.tracker.x = (self.tracker.x + 4)
        elif movement < -4:
            self.tracker.x = (self.tracker.x - 4)
        else:
            self.tracker.x = (self.tracker.x + movement)
        if self.tracker.x < 0:
            self.tracker.x = 0
        elif self.tracker.x > LAND_WIDTH - self.tracker.width-1:
            self.tracker.x = LAND_WIDTH - self.tracker.width-1
        for i in range(self.tracker.width):
            self.board[LAND_HEIGHT-1][(self.tracker.x+i)] = 2

    def moveObject(self):
        self.object.y += 1
        if self.object.y < LAND_HEIGHT-1:
            for i in range(self.object.objectSize):
                self.board[self.object.y-1][self.object.x+i] = 0
                self.board[self.object.y][self.object.x+i] = 1
        else:
            returnValue = 0
            hits = 0
            for i in range(self.object.objectSize):
                self.board[self.object.y-1][self.object.x+i] = 0
                if self.board[self.object.y][self.object.x+i] == 2:
                    hits += 1
            # Catch
            if hits == self.object.objectSize and self.object.objectSize < 5:
                returnValue = 1
            # Miss
            elif hits < self.object.objectSize < 5:
                returnValue = 2
            # Avoidance
            elif hits == 0 and hits >= self.object.objectSize:
                returnValue = 3
            # Hit
            elif hits > 0 and self.object.objectSize >= 5:
                returnValue = 4
            self.spawnObject()
            return returnValue


class ANN:
    # def __init__(self, layers):
    #     self.layers = layers
    #     self.weights = []
    #     for i in range(len(layers)-1):
    #         self.weights.append([])
    #         for j in range(layers[i]*layers[i+1]):
    #             self.weights[i].append(random.random())

    def __init__(self, layers, weights):
        self.layers = layers
        self.weights = weights
        self.neurons = [[0, 0], [0, 0]]

    def getOutput(self, inputs):
        for layerLevel in range(len(LAYERS)-1):
            # print("")
            neurons = [0, 0]
            # print(neurons)
            for i in range(LAYERS[layerLevel+1]):
                # Sum of inputs times weights, s
                for j in range(LAYERS[layerLevel]):
                    neurons[i] += inputs[j]*self.weights[layerLevel][i*len(inputs)+j]
                # print(neurons)
                neurons[i] += self.neurons[layerLevel][i]*self.weights[layerLevel][len(inputs)*2+i]
                neurons[i] += self.neurons[layerLevel][(i+1)%2]*self.weights[layerLevel][len(inputs)*2+2+i]
                # print(neurons)
                # Bias
                neurons[i] += 1 * self.weights[2+layerLevel][i]
                # print(neurons)
                # y + dy/dt
                neurons[i] += (-self.neurons[layerLevel][i]+neurons[i])/self.weights[6+layerLevel][i]
                # print("Last usable value?")
                # print(neurons)
                # Get neuron output
                if layerLevel == 1:
                    returnValue = neurons[0]
                elif layerLevel == 0:
                    neurons[i] = 1/(1+math.exp(-self.weights[4+layerLevel][i]*neurons[i]))
                # print(neurons)
                # time.sleep(3)
            for i in range(2):
                self.neurons[layerLevel][i] = (neurons[i]%1000000)
            inputs = list(self.neurons[layerLevel])
        if neurons[0] > neurons[1]:
            direction = -1
        else:
            direction = 1
        # result = int(max(neurons[0], neurons[1])*direction)
        return int(returnValue)



class EAtype:
    def __init__(self, length):
        self.genotype = self.randomGenotypes(length)
        self.phenotype = []

    # Generate a population of random genotypes
    def randomGenotypes(self, length):
        genotype = []
        for i in range(length):
            genotype.append(random.randint(0, 1))
        return genotype

def convertToPhenotypes(population):
    for EAtype in population:
        phenotype = []
        # Weights
        for i in range(len(LAYERS)-1):
            weights = []
            j = 0
            for j in range(LAYERS[i]*LAYERS[i+1]+4):
                gray = []
                for k in range(WEIGHT_BITS):
                    gray.append(EAtype.genotype[i*14*WEIGHT_BITS+j*WEIGHT_BITS+k])
                weights.append(gray2Weight(gray, -5, 5))
            phenotype.append(weights)
        # Bias weights
        for i in range(2):
            bias = []
            for j in range(2):
                gray = []
                for k in range(WEIGHT_BITS):
                    gray.append((EAtype.genotype[(22+2*i+j)*WEIGHT_BITS+k]))
                bias.append(gray2Weight(gray, -10, 0))
            phenotype.append(bias)
        # Gains
        for i in range(2):
            gains = []
            for j in range(2):
                gray = []
                for k in range(WEIGHT_BITS):
                    gray.append((EAtype.genotype[(26+2*i+j)*WEIGHT_BITS+k]))
                gains.append(gray2Weight(gray, 1, 5))
            phenotype.append(gains)
        # Time constant
        for i in range(2):
            timeConstants = []
            for j in range(2):
                gray = []
                for k in range(WEIGHT_BITS):
                    gray.append((EAtype.genotype[(30+2*i+j)*WEIGHT_BITS+k]))
                timeConstants.append(gray2Weight(gray, 1, 2))
            phenotype.append(timeConstants)
        EAtype.phenotype = phenotype


def gray2Bin(gray):
    binary = [gray[0]]
    for i in range(1, len(gray)):
        binary.append((binary[i-1]+gray[i])%2)
    return binary


def bin2Dec(bin):
    dec = 0
    for i in range(len(bin)):
        dec += bin[i]*2**(len(bin)-1-i)
    return dec


def gray2Weight(gray, minNum, maxNum):
    bin = gray2Bin(gray)
    dec = bin2Dec(bin)
    maxDec = 2**len(gray)
    intervalStart = (maxNum-minNum)*dec/maxDec
    intervalEnd = (maxNum-minNum)*(dec+1)/maxDec
    return random.random()*(intervalEnd-intervalStart)+intervalStart+minNum

# FITNESS
def valuateFitness(EAtype):
    if PROBLEM == 1:
        return valuateFitnessOneMax(EAtype)
    elif PROBLEM == 2:
        return valuateFitnessLOLZ(EAtype, Z)
    elif PROBLEM == 3:
        return valuateFitnessSuprisingGlobal(EAtype)
    elif PROBLEM == 4:
        return valuateFitnessSuprisingLocal(EAtype)
    elif PROBLEM == 5:
        return valuateFitnessBeerland(EAtype.phenotype)


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


def valuateFitnessFlatland(phenotype):
    ann = ANN(LAYERS, phenotype)
    flatland = copy.deepcopy(mainFlatland)
    flatland.spawnAgent(5, 5)
    poison = 0
    food = 0
    for i in range(50):
        sensorData = flatland.getSensorData()
        direction = ann.getOutput(sensorData)
        eaten = flatland.move(direction)
        if eaten == 1:
            food += 1
        elif eaten == 2:
            poison += 1
    return food - 2*poison

def valuateFitnessBeerland(phenotype):
    ann = ANN(LAYERS, phenotype)
    beerland = Beerland()
    beerland.spawnTracker()
    beerland.spawnObject()
    catch = 0
    miss = 0
    avoidance = 0
    hit = 0
    for i in range(300):
        sensorData = beerland.getSensorData()
        movement = ann.getOutput(sensorData)
        beerland.move(movement)
        result = beerland.moveObject()
        if result == 1:
            catch += 1
        elif result == 2:
            miss += 1
        elif result == 3:
            avoidance += 1
        elif result == 4:
            hit += 1
    return catch #+ avoidance #- miss - hit


def getTotalFitness(population):
    totalFitness = 0
    for EAtype in population:
        totalFitness += valuateFitness(EAtype)
    return totalFitness


# TOURNAMENT
def tournament(population):
    parents = []
    for i in range(populationSize):
        contestants = []
        indexes = list(range(0, len(population)))
        for j in range(K):
            index = indexes.pop(random.randint(0, len(indexes)-1))
            object = copy.deepcopy(population[index])
            contestants.append(object)
        if random.random() > EPSILON:
            parents.append(contestants[findBestContestant(contestants)])
        else:
            parents.append(contestants[random.randint(0, len(contestants)-1)])
    return parents


def findBestContestant(contestants):
    bestFitness = -1000000
    index = 0
    for i in range(len(contestants)):
        fitness = valuateFitness(contestants[i])
        if fitness > bestFitness:
            bestFitness = fitness
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
                EAtype = population[i]
                fitness += (1 + (valuateFitness(EAtype)-avgFitness)/(2*sd))/totalExpVal
        parents.append(population[i])
    return parents


def getTotalExpVal(population, avgFitness, sd):
    totalExpVal = 0
    for EAtype in population:
        if sd == 0:
            totalExpVal += 1
        else:
            totalExpVal += (1 + (valuateFitness(EAtype)-avgFitness)/(2*sd))
    return totalExpVal


def getStandardDeviation(population, avgFitness):
    sd = 0
    for EAtype in population:
        sd += (valuateFitness(EAtype)-avgFitness)**2
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
    index = findBestContestant(adults)
    for i in range(5):
        babies.append(adults[index])
    # adults = sigmaScaling(adults)
    while len(babies) < babyPopulationSize:
        parent1 = adults[random.randint(0, len(adults)-1)]
        parent2 = adults[random.randint(0, len(adults)-1)]
        baby = []
        for i in range(int(GENO_LENGTH/WEIGHT_BITS)):
            rand = random.random()
            for j in range(WEIGHT_BITS):
                if rand < 0.5:
                    baby.append(parent1.genotype[i])
                else:
                    baby.append(parent2.genotype[i])
        babyEAtype = EAtype(GENO_LENGTH)
        babyEAtype.genotype = baby
        babies.append(babyEAtype)
    return babies


def mutation(EAtype):
    for i in range(GENO_LENGTH):
        if random.random() < MUTATION_RATE:
            EAtype.genotype[i] = random.randint(0, S-1)


def mutateBabies(population):
    for EAtype in population:
        mutation(EAtype)

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
    print("Best:", bestValue)
    print("Average:", avg)
    print("SD", sd)
    print("Best sequel:", best.phenotype)

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


def visualizeRun(EAtype):
    ann = ANN(LAYERS, EAtype.phenotype)
    beerland = Beerland()
    beerland.spawnTracker()
    beerland.spawnObject()
    catch = 0
    miss = 0
    avoidance = 0
    hit = 0

    # initialize
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    for gameTime in range(300):
        screen.fill(white)
        for x in range(LAND_WIDTH):
            for y in range(LAND_HEIGHT):
                if beerland.board[y][x] == 1:
                    pygame.draw.rect(screen, blue,
                                     (x*int(WINDOW_WIDTH/LAND_WIDTH), y*int(WINDOW_HEIGHT/LAND_HEIGHT),
                                      int(WINDOW_WIDTH/LAND_WIDTH), int(WINDOW_HEIGHT/LAND_HEIGHT)),
                                     0)
                elif beerland.board[y][x] == 2:
                    pygame.draw.rect(screen, red,
                                     (x*int(WINDOW_WIDTH/LAND_WIDTH), y*int(WINDOW_HEIGHT/LAND_HEIGHT),
                                      int(WINDOW_WIDTH/LAND_WIDTH), int(WINDOW_HEIGHT/LAND_HEIGHT)),
                                     0)
        sensorData = beerland.getSensorData()
        movement = ann.getOutput(sensorData)
        print("Move right:", movement)
        beerland.move(movement)
        result = beerland.moveObject()
        if result == 1:
            catch += 1
            print("Points:", catch+avoidance)
        elif result == 2:
            miss += 1
        elif result == 3:
            avoidance += 1
            print("Points:", catch+avoidance)
        elif result == 4:
            hit += 1
        pygame.display.update()
        time.sleep(0.2)



population = []
adults = []
bestFitnesses = []
averageFitnesses = []
SDs = []
for i in range(babyPopulationSize):
    population.append(EAtype(GENO_LENGTH))
for i in range(GENERATIONS):
    convertToPhenotypes(population)
    adults = adultSelection(adults, population)
    bestFitnesses, averageFitnesses, SDs = printData(population, i, bestFitnesses, averageFitnesses, SDs)
    print("Total population fitness", getTotalFitness(population))
    population = makingBabies2(adults)
    mutateBabies(population)
#
convertToPhenotypes(population)
best = findBestContestant(population)
print(population[best])
visualizeRun(population[best])

# land = Beerland()
# land.spawnObject()
# land.spawnTracker()
# land.printBeerland()
# print(land.getSensorData())


