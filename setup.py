import pygame
import env
from reward import Reward
import time

pygame.init()

env = env.RacerEnv()
obs = env.reset()

def draw():
  env.render()
  pygame.display.update()



def update():
  action = env.car.getAction()
  print(action)
  next_observation, reward, done, info = env.step(action)
  #print(info)

running = True
while running:
  for event in pygame.event.get():
    if event.type == pygame.QUIT:
      running = False

  draw()
  update()