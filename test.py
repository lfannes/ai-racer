import pygame
import env
from reward import Reward
import time
from keras.models import Model, load_model
import numpy as np
import random

pygame.init()

env = env.RacerEnv()
rewards = Reward(env.circuit)

model = load_model("cartpole-dqn-1200-7.h5")
observation = env.reset()
observation = np.reshape(observation, [1, env.n_observations])

done = False


def get_action(state):
    if np.random.random() <= 0.02:
        return random.choice([0, 1, 2])
    else:
        act = np.argmax(model(state, training=False).numpy())
        print(act if act == 2 else "")
        return act

def getAction(weights=[0.5, 0.5]): #weights removes the uniform chance of the action, weights undefined means uniform random
    weights = [weights[0], 1-weights[0]] #to avoid errors
    action = random.choices([0, 1], weights=weights)
    return action[0] #0: LEFT, 1:RIGHT # 0: LEFT, 1:RIGHT


def draw():
    print("render")
    env.render()
  #pygame.display.update()

def update():
    global observation, done
    next_observation, reward, done, info = env.step(get_action(observation))
    observation = np.reshape(next_observation, [1, env.n_observations])
    if done:
        pygame.quit()
i=0
running = True
while running:
  for event in pygame.event.get():
    if event.type == pygame.QUIT:
      running = False

  draw()
  update()
  i += 1