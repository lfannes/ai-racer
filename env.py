import random
import pygame
import time
import car
import circuit
import reward
import numpy as np
import math

N_ACTIONS = 3
N_OBSERVATIONS = 7 # 8 raycast, 1 car angle

FPS = 10

pygame.init()

class RacerEnv():
    def __init__(self):
        self.step_limit = 10000
        self.steps = 0

        self.n_observations = N_OBSERVATIONS
        self.n_actions = N_ACTIONS

        self.screenWidth = 1920
        self.screenHeight = 1080

        self.car = None
        self.circuit = circuit.Circuit(N_OBSERVATIONS)
        self.rewards = reward.Reward(self.circuit)
        self.trackPositions = []

        self.clock = pygame.time.Clock()
        self.previousUpdateTime = time.time()

        self.stepCost = -1
        self.lineReward = 80
        self.linesCrossed = 0

        self.startPlace = 0

        self.window = None
        self.humanAction = ""
        self.humanMode = False

        self.start_positions = []
        self.calculate_start_positions()
        self.reset()

    def setupEnv(self):
        self.window = pygame.display.set_mode((self.screenWidth, self.screenHeight))
        pygame.display.set_caption("AI Racer")
        pygame.display.set_icon(pygame.image.load("resources/icon.png"))

    def step(self, action):
        self.steps += 1
        done = False if self.steps < self.step_limit else True
        self.car.update(action, 1/FPS)

        if self.rewards.getReward(self.car):
            reward = self.lineReward
            self.linesCrossed += 1
        else:
            reward = self.stepCost

        if self.circuit.isOffTrack(self.car):
            done = True

        observation = self.circuit.getObservation(self.car) / math.sqrt((self.screenWidth**2) - (self.screenHeight**2))
        info = observation

        return observation, reward, done, info

    def getAction(self, weights=[0.5, 0.5]): #weights removes the uniform chance of the action, weights undefined means uniform random
        weights = [weights[0], 1-weights[0]] #to avoid errors
        action = random.choices([0, 1], weights=weights)
        return action[0] #0: LEFT, 1:RIGHT

    def calculate_start_positions(self):
        for line in self.rewards.rewardLines:
            x = (line.positions[0][0] + line.positions[1][0]) / 2
            y = (line.positions[0][1] + line.positions[1][1]) / 2
            angle = line.directionVector.getAngle() - 90
            self.start_positions.append(car.StartPositions(x, y, angle, (line.lineNumber + 1) % self.rewards.numRaycasts))

    def reset(self, index=0):
        self.rewards.reset(self.start_positions[index].lineNumber)
        self.car = car.Car(self.start_positions[index])
        self.steps = 0
        return self.circuit.getObservation(self.car)

    def render(self, mode="human"):
        if mode == "human":
            if not self.humanMode:
                self.setupEnv()
                self.humanMode = True
            self.window.fill((255, 255, 255))
            self.circuit.draw(self.window)
            self.car.draw(self.window)
            self.rewards.draw(self.window)
            #self.humanAction = self.car.getAction()
            pygame.display.update()
            time.sleep(1/FPS)

    def close(self):
        pygame.quit()