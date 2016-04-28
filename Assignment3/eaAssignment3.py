import random
import math
import copy
import pygame
import sys
import time
from graphics import *

x = 0
y = 0
import os
os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (x,y)

white = (255, 255, 255)
red = (255, 0, 0)
green = (0, 255, 0)
blue = (0, 0, 255)

WINDOW_WIDTH = 600
WINDOW_HEIGHT = 800
LAYERS = [6, 3]
WEIGHT_BITS = 100
FLATLAND_SIZE = 10
INPUT_SIZE = 6
GENO_LENGTH = 0
for i in range(len(LAYERS)-1):
    GENO_LENGTH += LAYERS[i]*LAYERS[i+1]
GENO_LENGTH *= WEIGHT_BITS
babyPopulationSize = 200
populationSize = 20
GENERATIONS = 5
ADULT_SELECTION = 3
K = 10
EPSILON = 0.001
PROBLEM = 5
MUTATION_RATE = 0.05
S = 2
treshold = 0.2


class Agent:
    def __init__(self):
        self.x = random.randint(0, FLATLAND_SIZE)
        self.y = random.randint(0, FLATLAND_SIZE)
        self.direction = 2

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.direction = 2


class Flatland:
    def __init__(self, N):
        self.boardSize = N
        self.board = []
        for i in range(N):
            line = []
            for j in range(N):
                line.append([])
            self.board.append(line)
        self.fillFlatland(1/3, 1/3)

    def fillFlatland(self, f, p):
        for i in range(len(self.board)):
            for j in range(len(self.board[i])):
                rand = random.random()
                if rand < f:
                    self.board[i][j] = 1
                elif rand < f+(1-f)*p:
                    self.board[i][j] = 2
                else:
                    self.board[i][j] = 0

    def printFlatland(self):
        for y in range(self.boardSize):
            line = []
            for x in range(self.boardSize):
                line.append(self.board[x][y])
            print(line)

    def spawnAgent(self):
        x = random.randint(0, self.boardSize-1)
        y = random.randint(0, self.boardSize-1)
        self.board[x][y] = 3
        self.agent = Agent(x, y)

    def spawnAgent(self, x, y):
        self.board[x][y] = 3
        self.agent = Agent(x, y)

    def getSensorData(self):
        output = []
        neighbour = [self.board[self.agent.x][(self.agent.y-1)%self.boardSize],
                     self.board[(self.agent.x+1)%self.boardSize][self.agent.y],
                     self.board[self.agent.x][(self.agent.y+1)%self.boardSize],
                     self.board[(self.agent.x-1)%self.boardSize][self.agent.y]]
        for i in range(1, 3):
            if neighbour[self.agent.direction] == i:
                output.append(1)
            else:
                output.append(0)
            if neighbour[(self.agent.direction-1) % 4] == i:
                output.append(1)
            else:
                output.append(0)
            if neighbour[(self.agent.direction+1) % 4] == i:
                output.append(1)
            else:
                output.append(0)
        return output

    def move(self, direction):
        # If agent shall turn left
        if direction == 1:
            self.agent.direction = (self.agent.direction - 1) % 4
        # If agent shall turn right
        elif direction == 2:
            self.agent.direction = (self.agent.direction + 1) % 4
        # Set current position to empty
        self.board[self.agent.x][self.agent.y] = 0
        if self.agent.direction == 0:
            self.agent.y = (self.agent.y - 1) % FLATLAND_SIZE
        elif self.agent.direction == 1:
            self.agent.x = (self.agent.x + 1) % FLATLAND_SIZE
        elif self.agent.direction == 2:
            self.agent.y = (self.agent.y + 1) % FLATLAND_SIZE
        elif self.agent.direction == 3:
            self.agent.x = (self.agent.x - 1) % FLATLAND_SIZE
        eaten = self.board[self.agent.x][self.agent.y]
        self.board[self.agent.x][self.agent.y] = 3
        return eaten



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

    def getOutput(self, inputs):
        for i in range(len(self.layers)-1):
            output = [0]*self.layers[i+1]
            for j in range(self.layers[i+1]):
                for k in range(len(inputs)):
                    output[j] += inputs[k]*self.weights[i][k*self.layers[i+1]+j]
            inputs = output
        # print(output)
        if max(output) >= treshold:
            return output.index(max(output))
        return 3


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
    # phenotypes = []
    for EAtype in population:
        phenotype = []
        for i in range(len(LAYERS)-1):
            weights = []
            j = 0
            while j < (LAYERS[i]*LAYERS[i+1]):
                gray = []
                for k in range(WEIGHT_BITS):
                    gray.append(EAtype.genotype[j*WEIGHT_BITS+k])
                j += 1
                weights.append(gray2Weight(gray))
                # if j == 1:
                #     print("Gray: ", gray)
                #     print("Bin: ", gray2Bin(gray))
                #     print("Dec: ", bin2Dec(gray2Bin(gray)))
                #     print("Weight: ", weights[0])
                #     print("Intervall: ", 2*(bin2Dec(gray2Bin(gray))/2**len(gray))-1, 2*((bin2Dec(gray2Bin(gray))+1)/2**len(gray))-1)
            phenotype.append(weights)
        EAtype.phenotype = phenotype
        # phenotypes.append(phenotype)
    # return phenotypes


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


def gray2Weight(gray):
    bin = gray2Bin(gray)
    dec = bin2Dec(bin)
    maxDec = 2**len(gray)
    intervalStart = 2*dec/maxDec
    intervalEnd = 2*(dec+1)/maxDec
    return random.random()*(intervalEnd-intervalStart)+intervalStart-1

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
        return valuateFitnessFlatland(EAtype.phenotype)


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
    score = 0
    for i in range(len(mainFlatland)):
        flatland = copy.deepcopy(mainFlatland[i])
        flatland.spawnAgent(5, 5)
        for i in range(50):
            sensorData = flatland.getSensorData()
            direction = ann.getOutput(sensorData)
            # print(direction)
            eaten = flatland.move(direction)
            if eaten == 1:
                score += 1
            elif eaten == 2:
                score -= 2
                # return 0
    return score


def getTotalFitness(population):
    totalFitness = 0
    for EAtype in population:
        totalFitness += valuateFitness(EAtype)
    return totalFitness


# TOURNAMENT
def tournament(population):
    parents = []
    for i in range(populationSize):
        # print("Child nr: ", i)
        contestants = []
        indexes = list(range(0, len(population)))
        # print("Len pop:", len(population))
        # print("Total fitness: ", getTotalFitness(population))
        # print("Total fitness again: ", getTotalFitness(population))
        for j in range(K):
            index = indexes.pop(random.randint(0, len(indexes)-1))
            # print("Total fitness before copy: ", getTotalFitness(population))
            object = copy.deepcopy(population[index])
            # print("Total fitness before con: ", getTotalFitness(population))
            contestants.append(object)
            # print("Total fitness after con: ", getTotalFitness(population))
        if random.random() > EPSILON:
            parents.append(contestants[findBestContestant(contestants)])
            # print("Total fitness in if: ", getTotalFitness(population))
        else:
            parents.append(contestants[random.randint(0, len(contestants)-1)])
            # print("Total fitness in else: ", getTotalFitness(population))
        # print(len(parents))
    return parents


def findBestContestant(contestants):
    bestFitness = -1000000
    index = 0
    for i in range(len(contestants)):
        fitness = valuateFitness(contestants[i])
        if fitness > bestFitness:
            bestFitness = fitness
            # print("Temporary best fitness", bestFitness)
            index = i
    # print("Best fitness:", bestFitness)
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
    babies.append(adults[findBestContestant(adults)])
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
    win = GraphWin("Test", WINDOW_HEIGHT, WINDOW_WIDTH)
    # pygame.init()
    # clock = pygame.time.Clock()
    # screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    ann = ANN(LAYERS, EAtype.phenotype)
    # flatland = Flatland(FLATLAND_SIZE)
    flatland = copy.deepcopy(mainFlatland[0])
    flatland.spawnAgent(5, 5)
    poison = 0
    food = 0
    diRect = None

    # screen.fill(blue)
    for i in range(FLATLAND_SIZE):
        for j in range(FLATLAND_SIZE):
            rect = Rectangle(Point(j*WINDOW_HEIGHT/FLATLAND_SIZE, i*WINDOW_WIDTH/FLATLAND_SIZE),
                             Point((j+1)*WINDOW_HEIGHT/FLATLAND_SIZE, (i+1)*WINDOW_WIDTH/FLATLAND_SIZE))
            if flatland.board[j][i] == 1:
                rect.setFill('green')
                rect.draw(win)
                # pygame.draw.rect(screen, green, (j*FLATLAND_SIZEWINDOW_WIDTH, i*FLATLAND_SIZE/WINDOW_HEIGHT,
                #                                  FLATLAND_SIZE/WINDOW_WIDTH,FLATLAND_SIZE/WINDOW_HEIGHT))
            elif flatland.board[j][i] == 2:
                rect.setFill('red')
                rect.draw(win)
                # pygame.draw.rect(screen, red, (j*FLATLAND_SIZE/WINDOW_WIDTH, i*FLATLAND_SIZE/WINDOW_HEIGHT,
                #                                FLATLAND_SIZE/WINDOW_WIDTH, FLATLAND_SIZE/WINDOW_HEIGHT))
            elif flatland.board[j][i] == 3:
                rect.setFill('black')
                rect.draw(win)
                # if flatland.agent.direction == 0:
                diRect = Rectangle(Point(j*WINDOW_HEIGHT/FLATLAND_SIZE, i*WINDOW_WIDTH/FLATLAND_SIZE),
                             Point((j+1)*WINDOW_HEIGHT/FLATLAND_SIZE, (i+1)*WINDOW_WIDTH/FLATLAND_SIZE))
                # Rectangle(Point((j+1/2)*WINDOW_HEIGHT/FLATLAND_SIZE-10, i*WINDOW_WIDTH/FLATLAND_SIZE+10),
                #                    Point((j+1/2)*WINDOW_HEIGHT/FLATLAND_SIZE+10, i+30*WINDOW_WIDTH/FLATLAND_SIZE))

    if diRect is not None:
        diRect.setFill('white')
        diRect.draw(win)
    # flatland.printFlatland()
    autorun = 0
    poison = 0
    food = 0
    for i in range(50):
        # flatland.printFlatland()
        # print(" ")
        # time.sleep(0.1)
        rect = Rectangle(Point(flatland.agent.x*WINDOW_HEIGHT/FLATLAND_SIZE, flatland.agent.y*WINDOW_WIDTH/FLATLAND_SIZE),
                             Point((flatland.agent.x+1)*WINDOW_HEIGHT/FLATLAND_SIZE, (flatland.agent.y+1)*WINDOW_WIDTH/FLATLAND_SIZE))
        rect.setFill('white')
        rect.draw(win)
        sensorData = flatland.getSensorData()
        direction = ann.getOutput(sensorData)
        # print(direction)
        eaten = flatland.move(direction)
        if eaten == 1:
            food += 1
            print("Food", food)
        elif eaten == 2:
            poison += 1
            print("Poison", poison)

        rect = Rectangle(Point(flatland.agent.x*WINDOW_HEIGHT/FLATLAND_SIZE, flatland.agent.y*WINDOW_WIDTH/FLATLAND_SIZE),
                             Point((flatland.agent.x+1)*WINDOW_HEIGHT/FLATLAND_SIZE, (flatland.agent.y+1)*WINDOW_WIDTH/FLATLAND_SIZE))
        rect.setFill('black')
        rect.draw(win)
        if flatland.agent.direction == 0:
            diRect = Rectangle(Point((flatland.agent.x+1/2)*WINDOW_HEIGHT/FLATLAND_SIZE-5, flatland.agent.y*WINDOW_WIDTH/FLATLAND_SIZE),
                                 Point((flatland.agent.x+1/2)*WINDOW_HEIGHT/FLATLAND_SIZE+5, flatland.agent.y*WINDOW_WIDTH/FLATLAND_SIZE+10))
        elif flatland.agent.direction == 1:
            diRect = Rectangle(Point((flatland.agent.x+1)*WINDOW_HEIGHT/FLATLAND_SIZE-10, (flatland.agent.y+1/2)*WINDOW_WIDTH/FLATLAND_SIZE-5),
                                 Point((flatland.agent.x+1)*WINDOW_HEIGHT/FLATLAND_SIZE, (flatland.agent.y+1/2)*WINDOW_WIDTH/FLATLAND_SIZE+5))
        elif flatland.agent.direction == 2:
            diRect = Rectangle(Point((flatland.agent.x+1/2)*WINDOW_HEIGHT/FLATLAND_SIZE-5, (flatland.agent.y+1)*WINDOW_WIDTH/FLATLAND_SIZE-10),
                                 Point((flatland.agent.x+1/2)*WINDOW_HEIGHT/FLATLAND_SIZE+5, (flatland.agent.y+1)*WINDOW_WIDTH/FLATLAND_SIZE))
        elif flatland.agent.direction == 3:
            diRect = Rectangle(Point(flatland.agent.x*WINDOW_HEIGHT/FLATLAND_SIZE, (flatland.agent.y+1/2)*WINDOW_WIDTH/FLATLAND_SIZE-5),
                                 Point(flatland.agent.x*WINDOW_HEIGHT/FLATLAND_SIZE+10, (flatland.agent.y+1/2)*WINDOW_WIDTH/FLATLAND_SIZE+5))
        diRect.setFill('white')
        diRect.draw(win)
        # print("")
        # flatland.printFlatland()
        if autorun != "1":
            autorun = input("")
        else:
            time.sleep(0.3)
    print("Ate ", poison, " poison, and ", food, " food")

population = []
adults = []
bestFitnesses = []
averageFitnesses = []
SDs = []
mainFlatland = []
for i in range(5):
    mainFlatland.append(Flatland(FLATLAND_SIZE))
for i in range(babyPopulationSize):
    population.append(EAtype(GENO_LENGTH))
for i in range(GENERATIONS):
    # mainFlatland = []
    # for i in range(5):
    #     mainFlatland.append(Flatland(FLATLAND_SIZE))
    convertToPhenotypes(population)
    adults = adultSelection(adults, population)
    bestFitnesses, averageFitnesses, SDs = printData(population, i, bestFitnesses, averageFitnesses, SDs)
    print("Total population fitness", getTotalFitness(population))
    population = makingBabies2(adults)
    mutateBabies(population)

convertToPhenotypes(population)
bestFitnesses, averageFitnesses, SDs = printData(population, GENERATIONS+1, bestFitnesses, averageFitnesses, SDs)
mainFlatland = [Flatland(FLATLAND_SIZE)]
run = input("Press enter to view run")
best = population[findBestContestant(population)]
visualizeRun(best)

print(bestFitnesses)


