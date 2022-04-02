import numpy as np
import pygame
from pygame.sprite import Sprite
import math
import physics

circuitImage = pygame.image.load('resources/circuit.png')
pointImage = pygame.image.load('resources/point.png')

class Circuit(Sprite):
	def __init__(self, numberRaycasts):
		super().__init__()
		self.image = circuitImage
		self.mask = pygame.mask.from_surface(self.image)
		pygame.mask.Mask.invert(self.mask)
		self.rect = self.image.get_rect()
		self.angleBetweenRays = 360 / numberRaycasts
		self.numRaycasys = numberRaycasts
		self.point = Point()
		self.sizeOfPoint = 250
		self.raycastVel = 5
		self.lastObs = np.zeros((self.numRaycasys, 2))
		self.lastCar = None

	def collidesMask(self, obj):
		collision = pygame.sprite.collide_mask(self, obj)
		if not collision:
			return False
		else:
			return True

	def getObservation(self, car):
		observationPoints = np.zeros((self.numRaycasys, 2))
		observations = np.zeros(self.numRaycasys)
		carAngle = car.rigidbody.getRotation()
		currentAngle = 0
		pointPos = (car.rigidbody.getPositions()[0] + (car.width/2), car.rigidbody.getPositions()[1] + (car.height/2))
		self.point = Point()
		self.point.move(pointPos[0], pointPos[1])
		directionVector = physics.Vector2(math.sin(math.radians(carAngle - currentAngle)), -math.cos(math.radians(carAngle - currentAngle)))
		for i in range(self.numRaycasys):
			#move point until it hits the edge of the track
			while self.collidesMask(self.point):
				pointPos = (pointPos[0] + directionVector.getMovementPos(self.raycastVel).get()[0], pointPos[1] + directionVector.getMovementPos(self.raycastVel).get()[1])
				self.point.move(pointPos[0], pointPos[1])

			#save raycast in observation
			observationPoints[i][0] = self.point.x
			observationPoints[i][1] = self.point.y
			observations[i] = math.sqrt((pointPos[0] - self.point.x)**2 + (pointPos[1] - self.point.y)**2)

			#reset all the variable
			currentAngle += self.angleBetweenRays
			self.point = Point()
			pointPos = (car.rigidbody.getPositions()[0] + (car.width/2), car.rigidbody.getPositions()[1] + (car.height/2))
			self.point.move(pointPos[0], pointPos[1])
			directionVector = physics.Vector2(math.sin(math.radians(carAngle - currentAngle)), -math.cos(math.radians(carAngle - currentAngle)))

		self.lastObs = observationPoints
		self.lastCar = car
		return observations

	def draw(self, window):
		window.blit(self.image.convert_alpha(window), (0, 0))

		if not self.lastObs.any():
			return

		carCenterPos = (self.lastCar.rigidbody.getPositions()[0] + (self.lastCar.width / 2), self.lastCar.rigidbody.getPositions()[1] + (self.lastCar.height / 2))
		for i in range(self.numRaycasys):
			pygame.draw.line(window, (0, 0, 0), carCenterPos, self.lastObs[i])

class Point(Sprite):
	def __init__(self):
		self.image = pointImage
		self.mask = pygame.mask.from_surface(self.image)
		self.rect = self.image.get_rect()
		self.x = 0
		self.y = 0

	def move(self, x, y):
		self.x = int(x)
		self.y = int(y)
		self.rect = self.image.get_rect(center=self.image.get_rect(topleft=(self.x, self.y)).center)
		self.mask = pygame.mask.from_surface(self.image)