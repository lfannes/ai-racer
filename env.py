import gym
from gym import spaces
import numpy as np
import pygame
import car
import circuit
import time
import design

N_ACTIONS = 3
N_OBSERVATIONS = 8
FPS = 60

pygame.init()

class env(gym.Env):
    metadata = {'render.modes' : ['human']}

    def __init__(self, screenWidth, screenHeight):
        self.action_space = spaces.Discrete(N_ACTIONS)
        self.observation_space = spaces.Box(low=0, high=1000, shape=(N_OBSERVATIONS,), dtype=np.uint8) #length of raycasts

        self.screenWidth = screenWidth
        self.screenHeight = screenHeight

        self.car = car.Car(screenWidth / 2, screenHeight / 2)
        self.circuit = circuit.Circuit()

        self.clock = pygame.time.Clock()
        self.previousUpdateTime = time.time()

        self.badReward = -10
        self.goodReward = 1

        self.reset()

        self.setupEnv(screenWidth, screenHeight)

    def setupEnv(self, screenWidth, screenHeight):
        self.window = pygame.display.set_mode((screenWidth, screenHeight))
        pygame.display.set_caption("AI Racer")
        pygame.display.set_icon(pygame.image.load("resources/icon.png"))

    def step(self, action):
        info = "DRIVING LIKE A PRO :)"
        done = False

        self.car.update(action)

        if self.circuit.collides(self.car):
            print("COLLIDED WITH EDGE OF TRACK")
            info = "COLLISION"
            reward = self.badReward
        else:
            reward = self.goodReward

        observation = self.circuit.getObservation(self.car)

        return observation, reward, done, info

    def reset(self):
        self.window.fill((255, 255, 255))
        self.car = car.Car(self.screenWidth / 2, self.screenHeight / 2)
        return self.circuit.getObservation(self.car) #returning observation

    def render(self, mode="human"):
        if mode == "human":
            self.window.fill((255, 255, 255))
            self.circuit.draw(self.window)
            self.car.draw(self.window)
            pygame.display.update()

    def close(self):
        pass