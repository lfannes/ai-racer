import pygame
from physics import Rigidbody
from pygame.sprite import Sprite

carImage = pygame.image.load("resources/car.png")


class Car(Sprite):
  def __init__(self, x, y):
    super().__init__()
    self.vel = 400
    self.rigidbody = Rigidbody(x, y, self.vel)
    self.width = 100
    self.height = 50
    self.maxRotation = 4
    self.image = pygame.transform.flip(pygame.transform.scale(carImage, (self.width, self.height)), False, True)
    self.mask = pygame.mask.from_surface(self.image)
    self.rect = self.image.get_rect()


  def draw(self, window):
    rotatedImage = pygame.transform.rotate(self.image, -self.rigidbody.getRotation())
    self.rect = rotatedImage.get_rect(center=self.image.get_rect(topleft=(self.rigidbody.x, self.rigidbody.y)).center)
    self.mask = pygame.mask.from_surface(rotatedImage)
    window.blit(rotatedImage, self.rect)

  def update(self, action, dt):
    #print(action)
    if(pygame.key.get_pressed()[pygame.K_UP] or pygame.key.get_pressed()[pygame.K_w]):
      self.rigidbody.movePositions(dt)
    else:
      self.rigidbody.stop()

    if(pygame.key.get_pressed()[pygame.K_LEFT] or pygame.key.get_pressed()[pygame.K_a]):
      self.rigidbody.rotate(-self.maxRotation)
    elif(pygame.key.get_pressed()[pygame.K_RIGHT] or pygame.key.get_pressed()[pygame.K_d]):
      self.rigidbody.rotate(self.maxRotation)
      

      
    