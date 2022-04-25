import pygame
from physics import Rigidbody
from pygame.sprite import Sprite

carImage = pygame.image.load("resources/car.png")

class Car(Sprite):
  def __init__(self, x, y, angle):
    super().__init__()
    self.vel = 120
    self.rigidbody = Rigidbody(x, y, self.vel, angle)
    self.rigidbody.setRotation(angle)
    self.width = 100
    self.height = 50
    self.maxRotation = 18
    self.image = pygame.transform.flip(pygame.transform.scale(carImage, (self.width, self.height)), False, True)
    self.mask = pygame.mask.from_surface(self.image)
    self.rect = self.image.get_rect()
    self.rotatedImage = None


  def draw(self, window):
    self.updateImage()
    window.blit(self.rotatedImage, self.rect)

    olist = self.mask.outline()
    pygame.draw.lines(window, (200, 150, 150), 1, olist)

  def updateImage(self):
    self.rotatedImage = pygame.transform.rotate(self.image, -self.rigidbody.getRotation())
    self.rect = self.rotatedImage.get_rect(center=self.image.get_rect(topleft=(self.rigidbody.x, self.rigidbody.y)).center)
    self.mask = pygame.mask.from_surface(self.rotatedImage)

  def update(self, action, dt):
    self.updateImage()
    #if (action == 0):
    self.rigidbody.movePositions(dt)
   # else:
     # self.rigidbody.stop()

    if (action == 1):
      self.rigidbody.rotate(-self.maxRotation)
    elif (action == 2):
      self.rigidbody.rotate(self.maxRotation)
      
  def getAction(self):
    # if (pygame.key.get_pressed()[pygame.K_UP] or pygame.key.get_pressed()[pygame.K_w]):
    #   return 0

    if (pygame.key.get_pressed()[pygame.K_LEFT] or pygame.key.get_pressed()[pygame.K_a]):
      return 1
    elif (pygame.key.get_pressed()[pygame.K_RIGHT] or pygame.key.get_pressed()[pygame.K_d]):
      return 2
      
    