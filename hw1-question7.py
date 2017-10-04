#!/usr/bin/python
import numpy
import random
import time
import matplotlib.pyplot as plt

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


# Run parameters.
# Training points per run.
nTrainingPoints = 100
# Validation points per run.
nValidationPoints = 10000
# Number of runs.
nRuns = 5000

# Test results.
iterationsPerRun = []
disagreementPerRun = []

startTime = time.time()

# Run the learning process multiple times.
for run in range(nRuns):

	runStartTime = time.time()
	# Create target function f.
	p1 = ([ random.random()*2-1, random.random()*2-1 ])
	p2 = ([ random.random()*2-1, random.random()*2-1 ])
	m = (p2[1] - p1[1]) / (p2[0] - p1[0])

	WforF = pointsToWeights(p1, p2)

	line = weightsToPoints(WforF)
	#plt.plot(line[:,0], line[:,1], 'k')
	#plt.plot([p1[0], p2[0]],[p1[1], p2[1]], 'ko')


	# Produce the training set of inputs X and outputs Y.
	X = numpy.array([   [ random.random()*2-1, random.random()*2-1 ] for i in range(nTrainingPoints)])
	Y = [1 for i in range(nTrainingPoints)]
	for i in range(nTrainingPoints):
		Y[i] = (classify(X[i], WforF))
		if Y[i] == 1:
			pass#plt.plot(X[i,0], X[i,1],  'ro')
		else:
			pass#plt.plot(X[i,0], X[i,1],  'bo')




	# Now, approximate f by learning g.
	# Initiali weight vector: [0,0,0]
	W = numpy.array([0, 0, 0])
	# Mark all points as misclassified.
	pointIndicesMisclassified = range(len(X))
	iterationCounter = 0
	while pointIndicesMisclassified != []:
		iterationCounter += 1
		# Run the perceptron algorithm.
		# Pick a random, misclassified point.
		rand = random.randint(0, len(pointIndicesMisclassified)-1)
		pointIndex = pointIndicesMisclassified[rand]
		# Get point data.
		#print "Testing point {num:g}".format(num=pointIndex)
		x = X[pointIndex]
		y = Y[pointIndex]
		#print "   x: {x1:2.4f}, {x2:2.4f}".format(x1 = x[0], x2 = x[1])
		#print "   y: {y:g}".format(y=y)
		# Classify the point according to current weights.
		#print classify(x, W)
		# Now, apply the learning rule w + yCurrent*x.
		W = W + numpy.hstack([1,x]) * y
		#print "   New weight vector: [{w0:2.4f}, {w1:2.4f}, {w2:2.4f}].".format(w0=W[0], w1=W[1], w2=W[2])

		# Reclassify all points according to the new weight vector.
		#print "   Reclassifying points."
		pointIndicesMisclassified = []
		for i in range(nTrainingPoints):
			#print classify(X[i], W)
			#print Y[i]
			if classify(X[i], W) != Y[i]:
				#print "foo"
				pointIndicesMisclassified.append(i)


	line = weightsToPoints(W)
	#plt.plot(line[:,0], line[:,1], 'g')

	iterationsPerRun.append(iterationCounter)

	print ""
	print "Run {num:g}".format(num=run)
	print "   Training completed in {num:g} iterations.".format(num=iterationCounter)

	# Produce the training set of inputs X and outputs Y.
	Xvalidation = numpy.array([   [ random.random()*2-1, random.random()*2-1 ] for i in range(nValidationPoints)])
	YvalidationF = [1 for i in range(nValidationPoints)]
	YvalidationG = [1 for i in range(nValidationPoints)]
	disagreementCounter = 0.
	for i in range(nValidationPoints):
		YvalidationF[i] = (classify(Xvalidation[i], WforF))
		YvalidationG[i] = (classify(Xvalidation[i], W))
		#print YvalidationF[i]
		#print YvalidationG[i]
		if YvalidationF[i] != YvalidationG[i]:
			disagreementCounter += 1
		if YvalidationF[i] == 1:
			pass#plt.plot(Xvalidation[i,0], Xvalidation[i,1],  'ro', mfc='none')
		else:
			pass#plt.plot(Xvalidation[i,0], Xvalidation[i,1],  'bo', mfc='none')

	disagreementPerRun.append(disagreementCounter/nValidationPoints)


	plt.gca().set_xlim(-1,1)
	plt.gca().set_ylim(-1,1)
	plt.gca().set_aspect(1)

	#plt.show()
	runDuration = time.time() - runStartTime
	totalTime = time.time() - startTime
	averageRunDuration = totalTime / (run+1.)

	print "   Disagreement: {num:2.4f}".format(num=disagreementCounter/nValidationPoints)
	print "   Current run duration: {num:g}".format(num=int(runDuration))
	print "   Average run duration: {num:g}".format(num=int(averageRunDuration))
	print "   Approximate time to finish: {num:g}".format(num=(nRuns - (run + 1)) * averageRunDuration)
	print ""
	print "Average number of iterations until convergence: {num:2.4f}.".format(num=numpy.average(numpy.array(iterationsPerRun)))
	print "Average disagreement probability: {num:2.4f}".format(num=numpy.average(numpy.array(disagreementPerRun)))




