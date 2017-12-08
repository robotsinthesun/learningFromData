#coding: utf8

import numpy
import quadprog
import cvxopt



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
	def learn(self, weightDecayLambda=0):

		# Lambda matrix for weight decay.
		lambdas = numpy.identity(self.X.shape[1]) * weightDecayLambda
		# Use numpy to solve for weight vector.
		WReg = numpy.linalg.solve(self.X.T.dot(self.X) + lambdas, self.X.T.dot(self.Y))

		''' This was without weight decay.
		# Create the pseude inverse of the input vector.
		pseudoInverseX = numpy.linalg.pinv(self.X)
		# Get the weights.
		W = numpy.dot(pseudoInverseX + lambdas, self.Y)
		'''
		return WReg

	def calcMeanSquareError(self, X, Y, W):
		X = numpy.hstack([numpy.ones((X.shape[0],1)),X])
		return numpy.mean(numpy.power((numpy.dot(X, W) - Y), 2))

	def classify(self, X, W):
		# Append zeroth coordinate of 1.
		X = numpy.vstack([numpy.ones(X.shape[0]), X.T]).T
		return numpy.sign(numpy.dot(X,W))



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





 #####  #####   #####   ##     #####  ####  #####  ##  ## ###### ##  ##  #####
 ##  ## ##  ## ##       ##    ##     ##  ## ##  ## ### ##   ##   ### ## ##
 ##  ## #####  ####     ##    ####   ##  ## ##  ## ######   ##   ###### ##
 #####  ##  ## ##       ##    ##     ###### #####  ## ###   ##   ## ### ## ###
 ## ##  ##  ## ##       ##    ##     ##  ## ## ##  ##  ##   ##   ##  ## ##  ##
 ##  ## #####  ##       ###### ##### ##  ## ##  ## ##  ## ###### ##  ##  ####



class rbf:

	def __init__(self):
		self.KCenters = []
		self.indicesKToPoints = []
		self.indicesPointsToK = []
		self.W = []


	def learn(self, X, Y, K, gamma):

		self.X = X
		self.Y = Y
		self.K = K

		# Run Lloyd's algorithm to find clusters.
		self.findClusters()
		# Test for empty clusters.
		if numpy.any(numpy.isnan(self.KCenters)):
			return False

		# Now, find the weights.
		# First, add the bias coordinate to the K centers.
		KCenters = numpy.hstack([numpy.ones((self.KCenters.shape[0],1)), self.KCenters])
		#print self.KCenters
		# Set up the rbf coefficients matrix.
		# 1st dimension is data points, 2nd dimnension is K centers, 3rd dimension is point x and y.
		X = numpy.hstack([numpy.ones((self.X.shape[0],1)), self.X])
		phi = numpy.exp(-gamma * numpy.linalg.norm(X[:, None] - KCenters, axis=2))
		# Create the pseude inverse of the coefficients matrix.
		pseudoInverseX = numpy.linalg.pinv(phi)
		# Get the weights.
		self.W = numpy.dot(pseudoInverseX, self.Y)

		return True


	# LLoyds algorithm for finding the clusters.
	def findClusters(self):
		# Find cluster centers using Lloyd's algorithm.
		boundsMin = numpy.min(self.X, axis=0)
		boundsMax = numpy.max(self.X, axis=0)
		self.KCenters = numpy.random.random((self.K, self.X.shape[1])) * (boundsMax - boundsMin) + boundsMin
		# Iterate centers.
		for i in range(1000):

			# First, find clusters. These are the groups of points that are nearer to the current center than to any other.
			# Get vectors from each point to each center.
			# See this post for how to subtract all K centers from each point.
			# https://stackoverflow.com/questions/33303348/numpy-subtract-add-1d-array-from-2d-array/33303590
			vectorsPointToCenter = self.KCenters[:,None] - self.X
			# Get distances of the K-to-point vectors.
			distancesKToPoints = numpy.linalg.norm(vectorsPointToCenter, axis=2)
			# For each K, get the point distance rank.
			# The cluster where a point belongs to has a 0 at that point's position.
			ranksKToPoints = numpy.argsort(distancesKToPoints, axis=0)
			# For each point, get the index of the corresponding cluster.
			self.indicesPointsToK = ranksKToPoints[0,:]
			# For each cluster, get the indices of the corresponding points.
			# See this post for info on argwhere: https://stackoverflow.com/questions/432112/is-there-a-numpy-function-to-return-the-first-index-of-something-in-an-array
			self.indicesKToPoints = []
			for k in range(self.K):
				self.indicesKToPoints.append(numpy.argwhere(self.indicesPointsToK==k)[:,0].tolist())

			# Then, move the cluster centers to the average of the cluster's points.
			centersOld = numpy.copy(self.KCenters)
			for k in range(self.K):
				self.KCenters[k] = numpy.average(self.X[self.indicesKToPoints[k],:], axis=0)
			if numpy.all(centersOld == self.KCenters):
				break


	def classify(self, X, gamma):
		# First, add the bias coordinate to the K centers.
		KCenters = numpy.hstack([numpy.ones((self.KCenters.shape[0],1)), self.KCenters])
		# Append zeroth coordinate of 1.
		X = numpy.vstack([numpy.ones(X.shape[0]), X.T]).T
		# Create the phi matrix.
		phi = numpy.exp(-gamma * numpy.linalg.norm(X[:, None] - KCenters, axis=2))
		return numpy.sign(numpy.dot(phi, self.W))






  ##### ##  ## #####  #####   ####  ##### ######   ##  ##  ##### #### ###### ####  #####    ##  ##  ####   ####  ##  ## ###### ##  ##  #####
 ##     ##  ## ##  ## ##  ## ##  ## ##  ##  ##     ##  ## ##    ##  ##  ##  ##  ## ##  ##   ###### ##  ## ##  ## ##  ##   ##   ### ## ##
  ####  ##  ## ##  ## ##  ## ##  ## ##  ##  ##     ##  ## ####  ##      ##  ##  ## ##  ##   ###### ##  ## ##     ######   ##   ###### ####
     ## ##  ## #####  #####  ##  ## #####   ##     ##  ## ##    ##      ##  ##  ## #####    ##  ## ###### ##     ##  ##   ##   ## ### ##
     ## ##  ## ##     ##     ##  ## ## ##   ##      ####  ##    ##  ##  ##  ##  ## ## ##    ##  ## ##  ## ##  ## ##  ##   ##   ##  ## ##
 #####   ####  ##     ##      ####  ##  ##  ##       ##    ##### ####   ##   ####  ##  ##   ##  ## ##  ##  ####  ##  ## ###### ##  ##  #####




class svmAlgorithm:
	def __init__(self):
		foo = "foo"

	def learn(self,X,Y):
		nPoints = Y.shape[0]
		coeffs = []
		# Set up quadratic coefficients.
		for n in range(nPoints):
			row = []
			for m in range(nPoints):
				row.append(Y[n]*Y[m]*numpy.dot(X[n],X[m]))
			coeffs.append(row)

		alphaCoeffs = numpy.array(coeffs)

		# Now, use quadprog to solve the optimization problem.
		# minimize 0.5 alpha.T * coeffsAlpha * alpha + (-1).T * alphas
		# subject to y*alphas = 0
	#	q = -numpy.ones(nPoints)
	#	print alphaCoeffs
	#	print q
	#	print Y

		#alphas = self.quadprog_solve_qp(P=alphaCoeffs, q=q, G=None, h=None, A=Y, b=0)
	#	alphas = self.cvxopt_solve_qp(P=alphaCoeffs, q=q, G=None, h=None, A=Y, b=0)
		'''
		qp_G = .5 * (alphaCoeffs + alphaCoeffs.T)   # Make sure P is symmetric.
		# Set up the linear term -1.T * alphas.
		qp_a = -numpy.zeros(nPoints)
		qp_C = -Y#numpy.vstack([Y, None]).T
		qp_b = -0#numpy.hstack([0, None])
		meq = Y.shape[0]
		alphas = quadprog.solve_qp(qp_G, qp_a, qp_C, qp_b, meq)[0]
		'''
	#	print alphas
		'''
		YY = numpy.repeat(Y,nPoints).reshape(-1,nPoints).T * numpy.repeat(Y,nPoints).reshape(-1,nPoints)
		XIndices = X[numpy.arange(nPoints).repeat(nPoints).reshape(-1,nPoints)]
		XX = numpy.dot(X[XIndices].T, X[XIndices].T)
		print YY
		print XX
		'''
		W = [0,0,1]
		nSupportVectors = 3
		return W, nSupportVectors

	def quadprog_solve_qp(self, P, q, G=None, h=None, A=None, b=None):
		# Taken from here: https://scaron.info/blog/quadratic-programming-in-python.html
		# Set up the quadratic term. qp_G = 0.5 *  alpha.T * coeffsAlpha * alpha.
		qp_G = .5 * (P + P.T)   # make sure P is symmetric
		qp_a = -q
		if A is not None:
			qp_C = -numpy.vstack([A, G]).T
			qp_b = -numpy.hstack([b, h])
			meq = A.shape[0]
		else:  # no equality constraint
			qp_C = -G.T
			qp_b = -h
			meq = 0
		return quadprog.solve_qp(qp_G, qp_a, qp_C, qp_b, meq)[0]

	def cvxopt_solve_qp(self, P, q, G=None, h=None, A=None, b=None):
		P = .5 * (P + P.T)  # make sure P is symmetric
		args = [numpy.matrix(P), numpy.matrix(q)]
		if G is not None:
			args.extend([numpy.matrix(G), numpy.matrix(h)])
			if A is not None:
				args.extend([numpy.matrix(A), numpy.matrix(b)])
		sol = cvxopt.solvers.qp(*args)
		if 'optimal' not in sol['status']:
			return None
		return numpy.array(sol['x']).reshape((P.shape[1],))


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

def classifySign(X, W):
	# Append zeroth coordinate of 1.
	X = numpy.vstack([numpy.ones(X.shape[0]), X.T]).T
	return numpy.sign(numpy.dot(X,W))

def calcClassificationError(X, Y, W):
	# If the signs of X·W and Y differ, their product will be negative, otherwise positive.
	# Sum up the positive products and divide by number of points
	return 1 - numpy.sum(((classifySign(X, W) * Y) + 1) / 2.) / X.shape[0]

def calcMeanSquareError(X, Y, W):
	# Append the 0 coordinate of 1.
	if X.shape == ():
		X = numpy.array([1, X])
	else:
		X = numpy.hstack([numpy.ones((X.shape[0],1)),X])
	return numpy.mean(numpy.power((numpy.dot(X, W) - Y), 2))
