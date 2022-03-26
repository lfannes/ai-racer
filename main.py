import pygame
import car
import circuit
import time
import design

pygame.init()

fps = 60
(winWidth, winHeight) = (1920, 1080)
window = pygame.display.set_mode((winWidth, winHeight))

gameIcon = pygame.image.load("resources/icon.png")
pygame.display.set_caption("AI Racer")
pygame.display.set_icon(gameIcon)

car = car.Car(winWidth / 2 , winHeight / 2)
circuit = circuit.Circuit(8)
clock = pygame.time.Clock()
previousTime = time.time()
previousUpdateTime = time.time()
dt = 0

def draw():
  window.fill((255, 255, 255))
  circuit.draw(window)
  car.draw(window)


def update():
  global previousUpdateTime
  car.update("GAS", time.time() - previousUpdateTime)
  previousUpdateTime = time.time()
  circuit.update(window, car)
  circuit.collidesMask(car)
  obs = circuit.getObservation(car)

running = True
while running:
  previousTime = time.time()
  clock.tick(fps)

  for event in pygame.event.get():
    if event.type == pygame.QUIT:
      running = False

  draw()
  update()

  pygame.display.update()
  ("fps: {}".format(1 / (time.time() - previousTime)))