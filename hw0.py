#!/usr/bin/python
import numpy
import random
import matplotlib.pyplot as plt

# Create line points array.
pointsLine = numpy.array([   [random.random()*0.45,  random.random()],    [random.random()*0.45+0.55,  random.random()]   ])

# Calc slope.
m = (pointsLine[1,1] - pointsLine[0,1]) / (pointsLine[1,0] - pointsLine[0,0])

# Calc intercept. y = m * x + b --> b = y - m * x, solve for first point.
b = pointsLine[0,1] - m * pointsLine[0,0]

# Create function.
def f(x):
	return m * x + b

# Calc points for x = 0 and x = 1.
pointsLineLimits = numpy.array([   [0, f(0)],   [1, f(1)]   ])

# Create test points.
nPoints = 100
pointsTest = numpy.array([   [random.random(), random.random()] for i in range(nPoints)   ])

# Create slope normal vector.
vectorSlope = pointsLine[1,:] - pointsLine[0,:]
vectorSlopeNormal = numpy.array([-vectorSlope[1], vectorSlope[0]])

# Test points.
for i in range(nPoints):
	# Get vector to current test point from first line point.
	vectorLineToPoint = pointsTest[i] - pointsLine[0]
	# Check which side of line by dot product with slope normal vector and plot.
	if numpy.dot(vectorSlopeNormal, vectorLineToPoint) >= 0:
		plt.plot(pointsTest[i,0], pointsTest[i,1], 'r.')
	else:
		plt.plot(pointsTest[i,0], pointsTest[i,1], 'b.')

# Plot line and format plot.
plt.plot(pointsLineLimits[:,0], pointsLineLimits[:,1], 'k-')
plt.plot(pointsLine[:,0], pointsLine[:,1], 'ko')
plt.gca().set_xlim(0,1)
plt.gca().set_ylim(0,1)
plt.gca().set_aspect(1)
plt.gca().grid(1)
plt.show()
