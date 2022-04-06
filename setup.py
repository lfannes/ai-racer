import pygame
import env
from reward import Reward
import time

pygame.init()

env = env.RacerEnv()
rewards = Reward(env.circuit)

def draw():
  env.render("human")
  rewards.draw(env.window)
  time.sleep(1/10)
  pygame.display.update()



def update():
  next_observation, reward, done, info = env.step(env.car.getAction())
  rewards.getReward(500, 1800, -1, env.car)

running = True
while running:
  for event in pygame.event.get():
    if event.type == pygame.QUIT:
      running = False

  draw()
  update()