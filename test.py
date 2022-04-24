import pygame
import env
from reward import Reward
import time
from keras.models import Model, load_model
import numpy as np

pygame.init()

env = env.RacerEnv()
rewards = Reward(env.circuit)

model = load_model("cartpole-dqn-3000.h5")
state = env.reset()
state = np.reshape(state, [1, env.n_observations])

done = False

def action():
    if np.random.random() <= 0.02:
        return env.getRandomAction()
    else:
        return np.argmax(model(state, training=False).numpy())

def draw():
  env.render()
  rewards.draw(env.window)
  pygame.display.update()

def update():
    global state, done
    next_state, reward, done, info = env.step(action())
    state = np.reshape(next_state, [1, env.n_observations])
    if done:
        pygame.quit()
    rewards.getReward(500, 1800, -1, env.car)

running = True
while running:
  for event in pygame.event.get():
    if event.type == pygame.QUIT:
      running = False

  draw()
  update()