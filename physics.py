import math
import numpy as np

class Vector2:
  def __init__(self, x, y):
    self.x = x
    self.y = y
    self.angle = 0

  def __str__(self):
    return "x: {}, y: {}".format(self.x, self.y)

  def __mul__(self, factor):
    self.x *= factor
    self.y *= factor

  def get(self):
    return (self.x, self.y)

  def normalize(self):
    v = np.array((self.x, self.y))
    normalized_v = v / np.sqrt(np.sum(v**2))
    self.x = normalized_v[0]
    self.y = normalized_v[1]
    while(abs(self.angle) >= 2*math.pi): #desize the number if it made more than one full rotation
      self.angle -= 2*math.pi * (self.angle > 0 and 1 or -1)

  def getDistance(self, pos):
    return np.sqrt((self.x - pos.x)**2 + (self.y - pos.y)**2)

  def getAngle(self): #get angle in deg
    return (math.atan2(self.y, self.x) / math.pi) * 180

  def getMovementPos(self, vel):
    movement = Vector2(self.x, self.y)
    movement.normalize()
    movement * vel
    return movement

class Rigidbody:
  def __init__(self, x, y, vel, angle=0):
    self.x = x
    self.y = y
    self.topLeft = (10, 10) #for edge collision
    self.lowerRight = (1950, 950) #for edge collision
    self.movement = Vector2(np.cos((angle / 180) * math.pi), np.sin((angle / 180) * math.pi)) #movement vector that point forward relative to the car
    self.movement.normalize()
    self.vel = vel
    self.currentVel = 0
    self.thicks = 0

  def rotate(self, angle):
    self.movement.angle += (angle / 180) * math.pi

    self.movement.y = math.sin(self.movement.angle) * self.vel
    self.movement.x = math.cos(self.movement.angle) * self.vel
    self.movement.normalize()

  def getPositions(self):
    return (self.x, self.y)

  def stop(self):
    self.thicks = 0

  def movePositions(self, dt):
    # self.thicks += 1 if self.thicks < 100 else 0
    # self.currentVel += (self.vel * dt) * (self.thicks / 100) #make an animation that it looks like the car is speeding up
    #print('vel: ', self.currentVel, " thicks: ", self.thicks, " dt: ", dt)
    # if(self.currentVel >= self.vel):
    #   self.currentVel = self.vel * dt
    self.currentVel = self.vel * dt

    deltaPos = (self.movement.getMovementPos(self.currentVel).x, self.movement.getMovementPos(self.currentVel).y)
    if(self.x + deltaPos[0] > self.topLeft[0] and self.x + deltaPos[0] < self.lowerRight[0]):
      self.x += deltaPos[0]
    else:
      self.thicks = 0
      self.currentVel = 0

    if (self.y + deltaPos[1] > self.topLeft[1] and self.y + deltaPos[1] < self.lowerRight[1]):
      self.y += deltaPos[1]
    else:
      self.thicks = 0
      self.currentVel = 0

  def getRotation(self):
    angle = (self.movement.angle / math.pi) * 180 #convert the angle back to grads
    return angle

  def setRotation(self, angle):
    self.movement.angle = (angle / 180) * math.pi