# Imports
from random import randint
import math
import pygame
import sys

# Killable
KILLABLE = False

# Weights
SEPARATION_WEIGHT = 30
ALIGNMENT_WEIGHT = 14
COHESION_WEIGHT = 8
FLEEING_WEIGHT = 2000
AVVOIDANCE_WEIGHT = 4
NEIGHBOR_RADIUS = 50
DANGER_RADIUS = 10

# Maximum component speed
MAX_SPEED = 10
PREDATOR_MAX_SPEED = 12

# Window size
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600

# Colors
black = (0, 0, 0)
white = (255, 255, 255)
red = (255, 0, 0)
blue = (0, 0, 255)

# Boid size
BOID_COUNT = 200
BOID_RADIUS = 4
BOID_THICKNESS = 3

# Obstacle radius
OBSTACLE_RADIUS = 15

# FPS
fps = 30


class Boid:
    def __init__(self, x, y):
        self.position = [x, y]
        self.velocity = [randint(-MAX_SPEED, MAX_SPEED), randint(-MAX_SPEED, MAX_SPEED)]
        self.angle = 0
        self.dead = False

    def updateBoid (self, neighbors, predators, obstacles):
        # basic flock behaviour
        sep = [0, 0]
        align = [0, 0]
        coh = [0, 0]
        if len(neighbors) > 0:
            sep = self.calculateSeparationForce(neighbors)
            align = self.calculateAlignmentForce(neighbors)
            coh = self.calculateCohesionForce(neighbors)

        # flee from predators
        flee = [0, 0]
        for predator in predators:
            if self.distance(predator.position) < NEIGHBOR_RADIUS:
                if self.distance(predator.position) < DANGER_RADIUS and KILLABLE:
                    self.dead = True
                flee = self.calculateFleeingForce(predators)

        # obstacle avoidance
        col = self.collisionAvvoidance()

        # add forces to get resulting force
        self.velocity = [sum(x) for x in zip(self.velocity, sep, coh, align, flee, col)]
        self.reduceToMaxSpeed()

        # move boid with the resulting force
        oldPos = [self.position[0], self.position[1]]
        self.position[0] = math.floor(self.position[0] + self.velocity[0]) % WINDOW_WIDTH
        self.position[1] = math.floor(self.position[1] + self.velocity[1]) % WINDOW_HEIGHT

        # update the boid's direction
        if self.position[0] - oldPos[0] != 0:
            self.angle = math.pi/2 - math.atan((self.position[1] - oldPos[1])/(self.position[0] - oldPos[0]))
            if (self.position[0] - oldPos[0]) < 0:
                self.angle += math.pi

    def calculateSeparationForce(self, neighbors):
        force = [0, 0]
        for neighbor in neighbors:
            force[0] += (self.position[0]-neighbor.position[0])*SEPARATION_WEIGHT/self.distance(neighbor.position)
            force[1] += (self.position[1]-neighbor.position[1])*SEPARATION_WEIGHT/self.distance(neighbor.position)
        return force

    def calculateAlignmentForce(self, neighbors):
        avgVelX = 0
        avgVelY = 0
        for neighbor in neighbors:
            avgVelX += neighbor.velocity[0]
            avgVelY += neighbor.velocity[1]
        return [avgVelX/len(neighbors)*ALIGNMENT_WEIGHT, avgVelY/len(neighbors)*ALIGNMENT_WEIGHT]

    def calculateCohesionForce(self, neighbors):
        meanPosition = [0, 0]
        for neighbor in neighbors:
            meanPosition = [sum(x) for x in zip(meanPosition, neighbor.position)]
        meanPosition = [meanPosition[0]/len(neighbors), meanPosition[1]/len(neighbors)]
        # if self.position[0] < NEIGHBOR_RADIUS + BOID_RADIUS or self.position[0] > WINDOW_WIDTH - (NEIGHBOR_RADIUS + BOID_RADIUS) or self.position[1] < NEIGHBOR_RADIUS + BOID_RADIUS or self.position[1] > WINDOW_HEIGHT - (NEIGHBOR_RADIUS + BOID_RADIUS):
        #     return [(meanPosition[0]-self.position[0])*COHESION_WEIGHT/5, (meanPosition[1]-self.position[1])*COHESION_WEIGHT/5]
        return [(meanPosition[0]-self.position[0])*COHESION_WEIGHT, (meanPosition[1]-self.position[1])*COHESION_WEIGHT]

    def calculateFleeingForce(self, predators):
        fleeingForce = [0, 0]
        for predator in predators:
            fleeingForce[0] += (self.position[0]-predator.position[0])*FLEEING_WEIGHT/self.distance(predator.position)**2
            fleeingForce[1] += (self.position[1]-predator.position[1])*FLEEING_WEIGHT/self.distance(predator.position)**2
        return fleeingForce

    def collisionAvvoidance(self):
        force = [0, 0]
        for obstacle in obstacles:
            if OBSTACLE_RADIUS < self.distance(obstacle.position) < (NEIGHBOR_RADIUS + OBSTACLE_RADIUS):
                if self.avvoidToLeft(obstacle):
                    force[0] -= (obstacle.position[1]-self.position[1])*AVVOIDANCE_WEIGHT
                    force[1] += (obstacle.position[0]-self.position[0])*AVVOIDANCE_WEIGHT
                else:
                    force[0] += (obstacle.position[1]-self.position[1])*AVVOIDANCE_WEIGHT
                    force[1] -= (obstacle.position[0]-self.position[0])*AVVOIDANCE_WEIGHT
                force[0] -= (obstacle.position[0]-self.position[0])*AVVOIDANCE_WEIGHT/(self.distance(obstacle.position)/5)**3
                force[1] -= (obstacle.position[1]-self.position[1])*AVVOIDANCE_WEIGHT/(self.distance(obstacle.position)/5)**3
        return force

    def avvoidToLeft(self, obstacle):
        dist = [obstacle.position[0]-self.position[0], obstacle.position[1]-self.position[1]]
        if dist[0]*self.velocity[1]-dist[1]*self.velocity[0] > 0:
            return True
        else:
            return False


    # def collisionCourse(self, obstacles):
    #     for obstacle in obstacles:
    #         if obstacle.inObstacle(self.edgeOfSight()):
    #             return True
    #     return False

    # def edgeOfSight(self):
    #     return [self.position[0] + NEIGHBOR_RADIUS * math.sin(self.angle), self.position[1] + NEIGHBOR_RADIUS * math.cos(self.angle)]

    def distance(self, pos):
        dist = math.sqrt((self.position[0]-pos[0])**2+(self.position[1]-pos[1])**2)
        if dist != 0:
            return dist
        else:
            return 1/10000

    def reduceToMaxSpeed(self):
        maxComp = max(abs(self.velocity[0]), abs(self.velocity[1]))
        if maxComp > MAX_SPEED:
            self.velocity = [self.velocity[0]/maxComp*MAX_SPEED, self.velocity[1]/maxComp*MAX_SPEED]

    def draw(self, screen):
        pygame.draw.circle(screen, blue, (self.position[0], self.position[1]), BOID_RADIUS, BOID_THICKNESS)
        pygame.draw.line(screen, blue, self.position, (self.position[0] + 1.5 * BOID_RADIUS * math.sin(self.angle), self.position[1] + 1.5 * BOID_RADIUS * math.cos(self.angle)), BOID_THICKNESS)


class Predator(Boid):
    def updatePredator(self, flock):
        for boid in flock.boids:
            if self.distance(boid.position) < NEIGHBOR_RADIUS*2:
                self.velocity[0] -= (self.position[0]-boid.position[0])*SEPARATION_WEIGHT/self.distance(boid.position)
                self.velocity[1] -= (self.position[1]-boid.position[1])*SEPARATION_WEIGHT/self.distance(boid.position)
        self.reduceToMaxSpeed()

        # update predator's position
        oldPos = [self.position[0], self.position[1]]
        self.position[0] = math.floor(self.position[0] + self.velocity[0]) % WINDOW_WIDTH
        self.position[1] = math.floor(self.position[1] + self.velocity[1]) % WINDOW_HEIGHT

        # update direction
        if self.position[0] - oldPos[0] != 0:
            self.angle = math.pi/2 - math.atan((self.position[1] - oldPos[1])/(self.position[0] - oldPos[0]))
            if (self.position[0] - oldPos[0]) < 0:
                self.angle += math.pi

    def reduceToMaxSpeed(self):
        maxComp = max(abs(self.velocity[0]), abs(self.velocity[1]))
        if maxComp > PREDATOR_MAX_SPEED:
            self.velocity = [self.velocity[0]/maxComp*PREDATOR_MAX_SPEED*0.8, self.velocity[1]/maxComp*PREDATOR_MAX_SPEED*0.8]

    def draw(self, screen):
        pygame.draw.circle(screen, red, (self.position[0], self.position[1]), 2 * BOID_RADIUS, BOID_THICKNESS)
        pygame.draw.line(screen, red, self.position, (self.position[0] + 1.5 * 2 * BOID_RADIUS * math.sin(self.angle), self.position[1] + 1.5 * 2 * BOID_RADIUS * math.cos(self.angle)), BOID_THICKNESS)

class Flock:
    def __init__(self, flockSize):
        self.flockSize = flockSize
        self.boids = []
        for i in range(flockSize):
            self.boids.append(Boid(randint(0, WINDOW_WIDTH), randint(0, WINDOW_HEIGHT)))

    def getNeighbors(self):
        neigbors = []
        for boid in self.boids:
            neigborsToBoid = []
            for potentialNeighbor in self.boids:
                if boid != potentialNeighbor:
                    distance = math.sqrt((boid.position[0]-potentialNeighbor.position[0])**2 + (boid.position[1]-potentialNeighbor.position[1])**2)
                    if distance < NEIGHBOR_RADIUS:
                        neigborsToBoid.append(potentialNeighbor)
            neigbors.append(neigborsToBoid)
        return neigbors

    def draw(self, screen, predators, obstacles):
        boidNr = 0
        neighbors = self.getNeighbors()
        toBeKilled = []
        for boid in list(self.boids):
            boid.updateBoid(neighbors[boidNr], predators, obstacles)
            if boid.dead:
                toBeKilled.append(boidNr)
            else:
                boid.draw(screen)
            boidNr += 1
        for nr in reversed(toBeKilled):
            self.boids.pop(nr)
            self.flockSize -= 1


class Obstacle:
    def __init__(self, x, y):
        self.position = [x, y]

    def inObstacle(self, pos):
        dist = math.sqrt((self.position[0]-pos[0])**2+(self.position[1]-pos[1])**2)
        if dist < OBSTACLE_RADIUS + BOID_RADIUS:
            return True
        else:
            return False

    def draw(self, screen):
        pygame.draw.circle(screen, black, (self.position[0], self.position[1]), OBSTACLE_RADIUS + 5, 0)


# initialize
pygame.init()
clock = pygame.time.Clock()
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
flock = Flock(BOID_COUNT)
obstacles = []
predators = []

while True:
    # sets fps
    msElapsed = clock.tick(fps)


    # check for key events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
             pygame.quit(); sys.exit();
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                obstacles.append(Obstacle(randint(0, WINDOW_WIDTH-1), randint(0, WINDOW_HEIGHT)))
            elif event.key == pygame.K_a:
                obstacles = []
            elif event.key == pygame.K_w:
                predators.append(Predator(randint(0, WINDOW_WIDTH), randint(0, WINDOW_HEIGHT)))
            elif event.key == pygame.K_s:
                predators = []
            elif event.key == pygame.K_e:
                SEPARATION_WEIGHT += 1
                print("Separation weight: " + str(SEPARATION_WEIGHT))
            elif event.key == pygame.K_d:
                SEPARATION_WEIGHT -= 1
                print("Separation weight: " + str(SEPARATION_WEIGHT))
            elif event.key == pygame.K_r:
                ALIGNMENT_WEIGHT += 1
                print("Alignment weight: " + str(ALIGNMENT_WEIGHT))
            elif event.key == pygame.K_f:
                ALIGNMENT_WEIGHT -= 1
                print("Alignment weight: " + str(ALIGNMENT_WEIGHT))
            elif event.key == pygame.K_t:
                COHESION_WEIGHT += 1
                print("Cohesion weight: " + str(COHESION_WEIGHT))
            elif event.key == pygame.K_g:
                COHESION_WEIGHT -= 1
                print("Cohesion weight: " + str(COHESION_WEIGHT))
            elif event.key == pygame.K_y:
                for i in range(10):
                    flock.boids.append(Boid(randint(0, WINDOW_WIDTH), randint(0, WINDOW_HEIGHT)))
                flock.flockSize += 10
            elif event.key == pygame.K_h:
                for i in range(10):
                    if flock.flockSize-1 > 0:
                        flock.boids.pop(randint(0, flock.flockSize-1))
                    elif flock.flockSize-1 == 0:
                        flock.boids.pop(0)
                    else:
                        break
                    flock.flockSize -= 1
            elif event.key == pygame.K_u:
                PREDATOR_MAX_SPEED += 1
                print("Predator max speed: " + str(PREDATOR_MAX_SPEED))
            elif event.key == pygame.K_j:
                PREDATOR_MAX_SPEED -= 1
                print("Predator max speed: " + str(PREDATOR_MAX_SPEED))
            elif event.key == pygame.K_k:
                print("KILL MODE ACTIVATED!")
                KILLABLE = not KILLABLE
        elif event.type == pygame.MOUSEBUTTONDOWN:
            obstacles.append(Obstacle(event.pos[0], event.pos[1]))

    # erase the screen
    screen.fill(white)

    # predators
    for predator in predators:
        predator.draw(screen)
        predator.updatePredator(flock)

    # obstacles
    for obstacle in obstacles:
        obstacle.draw(screen)

    # update the boids and draw the updated picture
    flock.draw(screen, predators, obstacles)

    # update the screen
    pygame.display.update()