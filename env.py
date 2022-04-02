import pygame
import time
import car
import circuit
import random
import numpy as np

N_ACTIONS = 2
N_OBSERVATIONS = 8

FPS = 30

pygame.init()

class RacerEnv():
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.step_limit = 1000
        self.steps = 0
        self.sleep = 0

        self.n_observations = N_OBSERVATIONS
        self.n_actions = N_ACTIONS

        self.screenWidth = 1920
        self.screenHeight = 1080

        self.car = car.Car(1178, 756)
        self.circuit = circuit.Circuit(N_OBSERVATIONS)

        self.clock = pygame.time.Clock()
        self.previousUpdateTime = time.time()

        self.rewardGain = 1

        self.window = None
        self.humanAction = ""
        self.humanMode = False

        self.reset()

    def setupEnv(self):
        self.window = pygame.display.set_mode((self.screenWidth, self.screenHeight))
        pygame.display.set_caption("AI Racer")
        pygame.display.set_icon(pygame.image.load("resources/icon.png"))

    def step(self, action):
        info = "DRIVING LIKE A PRO :)"
        self.steps += 1
        done = False if self.steps < self.step_limit else True
        reward = 0
        if not action and self.humanAction:
            action = self.humanAction

        self.car.update(action, 1/FPS)
        self.previousUpdateTime = time.time()

        if self.circuit.collidesMask(self.car) and not done:
            reward = self.rewardGain
            done = False
        else:
            print("COLLIDED WITH EDGE OF TRACK")
            info = "COLLISION"
            done = True


        observation = self.circuit.getObservation(self.car)

        return observation, reward, done, info

    def getRandomAction(self):
        return np.random.randint(0, 2) #0: LEFT, 1:RIGHT

    def reset(self):
        self.steps = 0
        self.car = car.Car(1178, 756)
        return self.circuit.getObservation(self.car)

    def render(self, mode="human"):
        if mode == "human":
            if not self.humanMode:
                self.setupEnv()
                self.humanMode = True
            self.window.fill((255, 255, 255))
            self.circuit.draw(self.window)
            self.car.draw(self.window)
            #self.humanAction = self.car.getAction()
            #pygame.display.update()
            #time.sleep(1/FPS)

    def close(self):
        pygame.quit()