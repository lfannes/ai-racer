import pygame
import env
from reward import Reward
import time

pygame.init()

env = env.RacerEnv()
rewards = Reward()

def draw():
  env.render("human")
  rewards.draw(env.window, env.car)
  time.sleep(1/30)
  pygame.display.update()



def update():
  next_observation, reward, done, info = env.step(env.car.getAction())

running = True
while running:
  for event in pygame.event.get():
    if event.type == pygame.QUIT:
      running = False

  draw()
  update()