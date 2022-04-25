import numpy as np
import pygame
from pygame.sprite import Sprite
import numpy as np
import math
from physics import Vector2
from circuit import Point

pointImage = pygame.image.load('resources/point.png')

class Reward():
    def __init__(self, circuit):
        self.rewardLines = []
        self.rewardIndex = 0 #index of reward line that the car needs to cross
        self.pointsPerLine = 8
        self.numRaycasts = 12
        self.triggerDistance = 10
        self.raycastVel = 6
        self.startPos = (circuit.rect.x + circuit.rect.width/2, circuit.rect.y + circuit.rect.height/2)
        self.setup(self.startPos[0], self.startPos[1], circuit)

    def getReward(self, car):
        if self.rewardLines[self.rewardIndex].collide(car):
            self.rewardIndex = self.rewardIndex + 1 if self.rewardIndex + 1 < self.numRaycasts else 0 
            return True
        return False

    def reset(self):
        self.rewardIndex = 0

    def TrackEdgePos(self, startPos, circuit, angle):
        positions = []
        pointPos = startPos
        movingPoint = Point(pointPos[0], pointPos[1])
        direcionVector = Vector2(math.cos(math.radians(angle)), math.sin(math.radians(angle))).getMovementPos(self.raycastVel)
        direcionVector.normalize()
        while circuit.isOffTrack(movingPoint):
            pointPos = (pointPos[0] + direcionVector.get()[0],
                        pointPos[1] + direcionVector.get()[1])
            movingPoint.move(pointPos[0], pointPos[1])
        else:
            positions.append(pointPos)
            while not circuit.isOffTrack(movingPoint):
                pointPos = (pointPos[0] + direcionVector.get()[0],
                            pointPos[1] + direcionVector.get()[1])
                movingPoint.move(pointPos[0], pointPos[1])
            else:
                positions.append(pointPos)
        return positions

    def setup(self, x, y, circuit):
        anglePerStep = 360.0 / self.numRaycasts
        angle = 360.0
        for i in range(self.numRaycasts):
            self.rewardLines.append(RewardLine(self.TrackEdgePos((x, y), circuit, angle), self.pointsPerLine, angle, self.triggerDistance, i))
            angle -= anglePerStep

    def draw(self, window):
        self.rewardLines[self.rewardIndex].draw(window)


class RewardLine():
    def __init__(self, positions, pointPerLine, angle, triggerDistance, lineNumber):
        self.positions = positions
        self.pointPerLine = pointPerLine
        self.directionVector = Vector2(math.cos(math.radians(angle)), math.sin(math.radians(angle)))
        self.directionVector.normalize()
        self.points = []
        self.triggerDistance = triggerDistance
        self.lineNumber = lineNumber
        self.setup()

    def setup(self):
        length = Vector2(self.positions[0][0], self.positions[0][1]).getDistance(Vector2(self.positions[1][0], self.positions[1][1]))
        lengthPerStep = length / self.pointPerLine
        currentLenght = 0.0
        for i in range(self.pointPerLine):
            posX = self.positions[0][0] + (self.directionVector.getMovementPos(currentLenght).get()[0])
            posY = self.positions[0][1] + (self.directionVector.getMovementPos(currentLenght).get()[1])
            self.points.append(Point(posX, posY, (self.triggerDistance, self.triggerDistance)))
            currentLenght += lengthPerStep

    def collide(self, obj):
        for point in self.points:
            if point.collide(obj, self.triggerDistance):
                return True
        return False

    def draw(self, window):
        pygame.draw.line(window, (0, 0, 0), self.positions[0], self.positions[1])
        for point in self.points:
            window.blit(point.image, point.rect)

