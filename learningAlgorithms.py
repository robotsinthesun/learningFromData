import numpy



 #####   ##### #####   ####   ##### ##### ###### #####   ####  ##  ##
 ##  ## ##     ##  ## ##  ## ##     ##  ##  ##   ##  ## ##  ## ### ##
 ##  ## ####   ##  ## ##     ####   ##  ##  ##   ##  ## ##  ## ######
 #####  ##     #####  ##     ##     #####   ##   #####  ##  ## ## ###
 ##     ##     ## ##  ##  ## ##     ##      ##   ## ##  ##  ## ##  ##
 ##      ##### ##  ##  ####   ##### ##      ##   ##  ##  ####  ##  ##



class perceptronAlgorithm:

	def __init__(self, W, X, Y):
		self.W = W
		self.X = X
		self.Y = Y

	# Run the perceptron algorithm.
	def learn(self):
		iterationCounter = 0
		# Mark all points as misclassified.
		pointIndicesMisclassified = range(len(self.X))
		while pointIndicesMisclassified != []:
			# Pick a random, misclassified point.
			rand = numpy.random.randint(0, len(pointIndicesMisclassified))
			pointIndex = pointIndicesMisclassified[rand]
			# Get point data.
			x = self.X[pointIndex]
			y = self.Y[pointIndex]
			# Now, apply the learning rule w + yCurrent*x.
			self.W = self.W + numpy.hstack([1,x]) * y
			# Reclassify all points according to the new weight vector.
			pointIndicesMisclassified = []
			for i in range(self.X.shape[0]):
				if self.classify(self.X[i], self.W) != self.Y[i]:
					pointIndicesMisclassified.append(i)
			iterationCounter += 1
		return self.W, iterationCounter

	def classify(self, x, W):
		x = numpy.hstack([1, x])
		return numpy.sign(W.dot(x))



 ##     ###### ##  ##  #####  ####  #####    #####   ##### #####
 ##       ##   ### ## ##     ##  ## ##  ##   ##  ## ##    ##
 ##       ##   ###### ####   ##  ## ##  ##   ##  ## ####  ##
 ##       ##   ## ### ##     ###### #####    #####  ##    ## ###
 ##       ##   ##  ## ##     ##  ## ## ##    ## ##  ##    ##  ##
 ###### ###### ##  ##  ##### ##  ## ##  ##   ##  ##  ##### ####



class linearRegressionAlgorithm:

	def __init__(self, X, Y):
		# Add column of ones to X.
		self.X = numpy.hstack([numpy.ones((X.shape[0],1)),X])
		self.Y = Y

	# Run the linear regression algorithm.
	def learn(self):
		pseudoInverseX = numpy.linalg.pinv(self.X)
		W = numpy.dot(pseudoInverseX, self.Y)
		return W




 ##  ##  ##### ##     #####   ##### #####   #####
 ##  ## ##     ##     ##  ## ##     ##  ## ##
 ###### ####   ##     ##  ## ####   ##  ##  ####
 ##  ## ##     ##     #####  ##     #####      ##
 ##  ## ##     ##     ##     ##     ## ##      ##
 ##  ##  ##### ###### ##      ##### ##  ## #####



def pointsToWeights(p1, p2):
	# Calculate slope m and intercept b.
	# x2 = m * x1 + b
	# m = dx2 / dx1
	m = (p2[1] - p1[1]) / (p2[0] - p1[0])
	# b = x2 - m * x1
	b = p1[1] - m * p1[0]
	# Return weight vector.
	return numpy.array([b, m, -1])

def weightsToPoints(W):
	m = W[1]
	b = W[0]
	# Calculate points at x1 = -1 and x1 = 1
	return numpy.array([   [-1, m * -1 + b],   [1, m * 1 + b]   ])

def classify(p, W):
	x = numpy.hstack([1, p])
	return numpy.sign(W.dot(x))
