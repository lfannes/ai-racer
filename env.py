import random
import pygame
import time
import car
import circuit
import reward
import numpy as np
import math

N_ACTIONS = 2
N_OBSERVATIONS = 9 # 8 raycast, 1 car angle

FPS = 10

pygame.init()

class RacerEnv():
    def __init__(self):
        self.step_limit = 300*3
        self.steps = 0
        self.sleep = 0

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
        self.lineReward = 150
        self.linesCrossed = 0

        self.startPlace = 0

        self.window = None
        self.humanAction = ""
        self.humanMode = False

        self.reset()

    def setupEnv(self):
        self.window = pygame.display.set_mode((self.screenWidth, self.screenHeight))
        pygame.display.set_caption("AI Racer")
        pygame.display.set_icon(pygame.image.load("resources/icon.png"))

    def step(self, action):
        info = str(self.car.rigidbody.getPositions())
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
        np.append(observation, (self.car.rigidbody.movement.getAngle() / 180) * math.pi)

        return observation, reward, done, info

    def getAction(self, weights=[0.5, 0.5]): #weights removes the uniform chance of the action, weights undefined means uniform random
        weights = [weights[0], 1-weights[0]] #to avoid errors
        action = random.choices([0, 1], weights=weights)
        return action[0] #0: LEFT, 1:RIGHT

    def reset(self, changePos=False):
        self.steps = 0
        self.linesCrossed = 0
        self.trackPositions = [] #points were car can start
        for i in range(len(self.rewards.rewardLines)):
            x = (self.rewards.rewardLines[i].positions[0][0] + self.rewards.rewardLines[i].positions[1][0]) / 2
            y = (self.rewards.rewardLines[i].positions[0][1] + self.rewards.rewardLines[i].positions[1][1]) / 2
            angle = self.rewards.rewardLines[i].directionVector.getAngle() + 90
            lineNumber = i + 1 if i + 1 < self.rewards.numRaycasts else 0
            self.trackPositions.append([x, y, angle, lineNumber])

        if changePos:
            self.startPlace = random.randint(0, self.rewards.numRaycasts - 1)
            self.startPlace = 8 if self.startPlace == 9 else self.startPlace

        self.car = car.Car(self.trackPositions[self.startPlace][0] + random.uniform(-5., 5.), self.trackPositions[self.startPlace][1] + random.uniform(-5., 5.), self.trackPositions[self.startPlace][2])
        self.rewards.reset()
        self.rewards.rewardIndex = self.trackPositions[self.startPlace][3]
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