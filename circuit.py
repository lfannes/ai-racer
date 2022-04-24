import numpy as np
import pygame
from pygame.sprite import Sprite
import math
import physics

pygame.init()

circuitImage = pygame.image.load('resources/circuit.png')
pointImage = pygame.image.load('resources/point.png')
defaultFont = pygame.font.Font('resources/font.otf', 32)

class Circuit(Sprite):
	def __init__(self, numberRaycasts):
		super().__init__()
		self.image = circuitImage
		self.mask = pygame.mask.from_surface(self.image)
		pygame.mask.Mask.invert(self.mask)
		self.rect = self.image.get_rect()
		self.angleBetweenRays = 360 / numberRaycasts
		self.numRaycasts = numberRaycasts
		self.point = Point()
		self.sizeOfPoint = 250
		self.raycastVel = 5
		self.lastObs = np.zeros((self.numRaycasts, 2))
		self.lastCar = None
		self.text = defaultFont.render('', True, (255, 0, 0), (0, 255, 0))
		self.textRect = self.text.get_rect()

	def isOffTrack(self, obj):
		collision = pygame.sprite.collide_mask(self, obj)
		if collision:
			return True
		else:
			return False

	def getObservation(self, car):
		observationPoints = np.zeros((self.numRaycasts, 2))
		observations = np.zeros(self.numRaycasts)
		carAngle = car.rigidbody.getRotation()
		currentAngle = 0
		pointPos = (car.rigidbody.getPositions()[0] + (car.width/2), car.rigidbody.getPositions()[1] + (car.height/2))
		self.point = Point()
		self.point.move(pointPos[0], pointPos[1])
		directionVector = physics.Vector2(math.cos(math.radians(carAngle + currentAngle)),
										  math.sin(math.radians(carAngle + currentAngle)))
		for i in range(self.numRaycasts):
			#move point until it hits the edge of the track
			while not self.isOffTrack(self.point):
				pointPos = (pointPos[0] + directionVector.getMovementPos(self.raycastVel).get()[0], pointPos[1] + directionVector.getMovementPos(self.raycastVel).get()[1])
				self.point.move(pointPos[0], pointPos[1])

			#save raycast in observation
			observationPoints[i][0] = self.point.x
			observationPoints[i][1] = self.point.y
			observations[i] = math.sqrt((car.rigidbody.getPositions()[0] - self.point.x)**2 + (car.rigidbody.getPositions()[0] - self.point.y)**2)

			#reset all the variable
			currentAngle += self.angleBetweenRays
			self.point = Point()
			pointPos = (car.rigidbody.getPositions()[0] + (car.width/2), car.rigidbody.getPositions()[1] + (car.height/2))
			self.point.move(pointPos[0], pointPos[1])
			directionVector = physics.Vector2(math.cos(math.radians(carAngle + currentAngle)), math.sin(math.radians(carAngle + currentAngle)))

		self.lastObs = observationPoints
		self.lastCar = car
		return observations

	def draw(self, window):
		window.blit(self.image.convert_alpha(window), (0, 0))

		if not self.lastObs.any():
			return

		carCenterPos = (self.lastCar.rigidbody.getPositions()[0] + (self.lastCar.width / 2), self.lastCar.rigidbody.getPositions()[1] + (self.lastCar.height / 2))
		for i in range(self.numRaycasts):
			pygame.draw.line(window, (0, 0, 0), carCenterPos, self.lastObs[i])
			pygame.draw.circle(window, (255/(i+1), 255/(i+1), 255/(i+1)), self.lastObs[i], 5)

class Point(Sprite):
	def __init__(self, x=0, y=0, size=(0, 0)):
		self.image = pointImage
		self.x = x
		self.y = y
		if size != (0, 0):
			self.image = pygame.transform.scale(self.image, size)
		self.rect = self.image.get_rect()
		self.rect = pygame.Rect.move(self.rect, self.x, self.y)
		self.mask = pygame.mask.from_surface(self.image)

	def __repr__(self):
		return f"({self.x}, {self.y})"

	def move(self, x, y):
		self.x = int(x)
		self.y = int(y)
		self.rect = self.image.get_rect()
		self.rect = pygame.Rect.move(self.rect, self.x, self.y)
		self.mask = pygame.mask.from_surface(self.image)

	def collide(self, obj, range):
		collision = pygame.sprite.collide_rect(self, obj)
		return collision