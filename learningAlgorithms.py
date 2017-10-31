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
		# Create the pseude inverse of the input vector.
		pseudoInverseX = numpy.linalg.pinv(self.X)
		# Get the weights.
		W = numpy.dot(pseudoInverseX, self.Y)
		return W



 ##     ####   #####  ####  #####  ###### ###### ##  ## ##  ## ###### ####    #####   ##### #####
 ##    ##  ## ##     ##  ## ##  ##   ##     ##   ##  ## ######   ##  ##  ##   ##  ## ##    ##
 ##    ##  ## ##     ##  ## ##  ##   ##     ##   ###### ######   ##  ##       ##  ## ####  ##
 ##    ##  ## ## ### ###### #####    ##     ##   ##  ## ##  ##   ##  ##       #####  ##    ## ###
 ##    ##  ## ##  ## ##  ## ## ##    ##     ##   ##  ## ##  ##   ##  ##  ##   ## ##  ##    ##  ##
 ###### ####   ####  ##  ## ##  ## ######   ##   ##  ## ##  ## ###### ####    ##  ##  ##### ####



class logarithmicRegressionAlgorithm:

	def __init__(self, X, Y, eta=0.01, termDeltaE=0.01, termIterations=1000):
		# X contains the vector input vectors of dimension d for each of N data points.
		# Add column of ones to X.
		self.X = numpy.hstack([numpy.ones((X.shape[0], 1)), X])
		# Y contains a single output value for N data points.
		self.Y = Y
		# Initialize weight vector w to all zero.
		self.W = numpy.zeros(self.X.shape[1])
		# Learning rate.
		self.eta = eta
		# Termination criterion for error change rate.
		self.termIterations = termIterations
		self.termDeltaE = termDeltaE



	# Run the logarithmic regression algorithm.
	def learn(self):

		# We use stochastic gradient descent, so we compute the gradient of the
		# error function for individual points.
		deltaW = 1.
		i = 0
		while deltaW > self.termDeltaE and i < self.termIterations:
			Wprev = self.W
			permutation = numpy.random.permutation(self.X.shape[0])
			# Run through the points in the order given by the permutation.
			for j in permutation:
				# Get the training data from the dataset.
				x = self.X[j]
				y = self.Y[j]
				# Calculate the gradient of the error function we are trying to minimize.
				gradE = (-y*x) / (1+numpy.exp(y * numpy.dot(self.W,x)))
				# Update W.
				# Assign the new self.W explicitily.
				self.W = self.W - (gradE * self.eta)
			# Increment.
			i += 1
			# Calc error delta.
			# Evaluate the error function with the new and the old weights.
			# Here, the error function is the euklidean difference of the
			# old and the updated weight vectors.
			deltaW = numpy.linalg.norm(Wprev - self.W)

		print "   Learning algorithm terminated after {n:g} iterations.".format(n=i+1)
		print "   Final weight vector: " + str(self.W) + "."
		print "   Last weight change: {n:2.7f}.".format(n=deltaW)

		# Return the weight vector and the number iterations until termination.
		return self.W, i+1


	def calcCrossEntropyError(self, x,y,W):
		# Add the x0 coordinate of 1.
		x = numpy.hstack([1, x])
		return numpy.log(1+numpy.exp(-y*numpy.dot(W,x)))

	def classify(self, p, W):
		# Add the x0 coordinate of 1.
		x = numpy.hstack([1., p])
		# Classify the point according to the
		return self.theta(W.dot(x))

	def theta(self, s):
		return numpy.exp(s) / (1.+numpy.exp(s))



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
	b = W[0]
	m = W[1]
	# Calculate points at x1 = -1 and x1 = 1
	return numpy.array([   [-1, m * -1 + b],   [1, m * 1 + b]   ])

def classifySign(p, W):
	x = numpy.hstack([1, p])
	return numpy.sign(W.dot(x))
