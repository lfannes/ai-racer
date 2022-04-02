import pygame
import time
import math

class Rect():
    def __init__(self, x1, y1, x2, y2):
        self.pos1 = (x1, y1)
        self.pos2 = (x2, y2)
        self.rect = None

    def __repr__(self):
        return "Rect(" + str(self.pos1[0]) + ", " + str(self.pos1[1]) + ", " + str(self.pos2[0]) + ", " + str(self.pos2[1]) + ")"

    def draw(self, screen):
        pygame.draw.line(screen, (0, 0, 0), self.pos1, self.pos2, width=7)

    def checkCollision(self, rectObj):
        pass

class TrackDesigner():
    def __init__(self):
        self.doneDesigning = False
        self.trackEdges = [[], []]
        self.trackEdges = [[Rect(87, 97, 1768, 85), Rect(1768, 85, 1782, 892), Rect(1782, 892, 117, 891), Rect(117, 891, 87, 97)], [Rect(262, 228, 1583, 233), Rect(1583, 233, 1587, 784), Rect(1587, 784, 275, 747), Rect(275, 747, 262, 228)]]
        self.tempPos = None
        self.timeLastPress = time.time() #in seconds
        self.minDelayToPress = 0.5 # in seconds
        self.currentState = 0 #0: outer ring, 1: inner ring
        print(self.trackEdges)

    def addNewEdge(self, x, y):
        if (not self.tempPos):  # self.temp DOESN'T already exist
            self.tempPos = pygame.mouse.get_pos()
        else:
            self.trackEdges[self.currentState].append(Rect(self.tempPos[0], self.tempPos[1], x, y))
            self.tempPos = pygame.mouse.get_pos()
            print(self.trackEdges)

        self.timeLastPress = time.time()

    def update(self):
        if(pygame.mouse.get_pressed()[0] and time.time() - self.timeLastPress > self.minDelayToPress):
            self.addNewEdge(pygame.mouse.get_pos()[0], pygame.mouse.get_pos()[1])

        if(pygame.key.get_pressed()[pygame.K_RETURN] and time.time() - self.timeLastPress > self.minDelayToPress):
            self.addNewEdge(self.trackEdges[self.currentState][0].pos1[0], self.trackEdges[self.currentState][0].pos1[1])
            self.currentState = 1
            self.timeLastPress = time.time()
            self.tempPos = None

        print(self.trackEdges)



    def draw(self, screen):
        for i in range(len(self.trackEdges)):
            for edge in self.trackEdges[i]:
                edge.draw(screen)