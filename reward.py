import numpy as np
import pygame
from pygame.sprite import Sprite
import numpy as np
import math
from physics import Vector2

pointImage = pygame.image.load('resources/point.png')

class Reward():
    def __init__(self):
        self.rewardLines = np.array([RewardLine(500, 20, Vector2(100, 500))], dtype=RewardLine)

    def addRewardLine(self, pos1, pos2):
        length = pos1.getDistance(pos2)
        a = Vector2((pos2.x-pos1.x), (pos2.y-pos1.y))
        angle = math.atan(0) - math.atan(a.getAngle())

        rewardLine = RewardLine(length, angle, pos1)
        np.append(self.rewardLines, rewardLine)

    def draw(self, window, car):
        for line in self.rewardLines:
            line.draw(window, car)


class RewardLine(Sprite):
    def __init__(self, length, angle, pos):
        self.image = pointImage
        self.rotatedImage = None
        self.rect = self.image.get_rect()
        self.mask = pygame.mask.from_surface(self.image)
        self.height = self.image.get_height()
        self.setup(length, angle, pos)

    def __repr__(self):
        return "RewardLine :)"

    def setup(self, length, angle, pos): #rotate and scale image
        self.image = pygame.transform.scale(self.image, (length, self.height))
        #self.rotatedImage = pygame.transform.rotate(self.image, angle)
        self.rect = self.image.get_rect(center=self.image.get_rect(topleft=(pos.x, pos.y)).center)
        self.mask = pygame.mask.from_surface(self.image)

    def draw(self, window, car):
        #
        # # window.blit(self.image, (100, 100))
        # x = pygame.draw.line(window, (255, 0, 0), (self.rect.x, self.rect.y), (self.rect.x + self.rect.width, self.rect.y + self.rect.height), width=5)
        # # collision = pygame.sprite.collide_mask(self, car)
        # # print(collision)
        olist = self.mask.outline()
        pygame.draw.lines(window,(200,150,150),1,olist)
        window.blit(self.image, self.rect)