#!/usr/bin/python
# coding: utf8

import time
import random
import numpy
import matplotlib.pyplot as plt

from learningAlgorithms import *


# Run parameters. **************************************************************
nTrainingPoints = 100
nValidationPoints = 1000
nRuns = 1000
plot = False

# Test results. ****************************************************************
iterationsPerRun = []
outOfSampleErrorPerRun = []
inSampleErrorPerRun = []




# Run the learning process multiple times. *************************************
startTime = time.time()
for run in range(nRuns):

	startTimeRun = time.time()

	# Create target function f.
	# Create random points.
	p1 = ([ random.random()*2-1, random.random()*2-1 ])
	p2 = ([ random.random()*2-1, random.random()*2-1 ])
	# Calc weights for target function.
	WforF = pointsToWeights(p1, p2)
	# Plot points and line.
	if plot:
		line = weightsToPoints(WforF)
		plt.plot(line[:,0], line[:,1], 'k')
		plt.plot([p1[0], p2[0]],[p1[1], p2[1]], 'ko')


	# Produce the training set of inputs X and outputs Y.
	X = numpy.array([   [ random.random()*2-1, random.random()*2-1 ] for i in range(nTrainingPoints)])
	Y = [1 for i in range(nTrainingPoints)]
	for i in range(nTrainingPoints):
		Y[i] = (classify(X[i], WforF))
		Y = numpy.array(Y)
		if plot:
			if Y[i] == 1:
				plt.plot(X[i,0], X[i,1],  'ro')
			else:
				plt.plot(X[i,0], X[i,1],  'bo')



	# Approximate f by learning the weights W for g using linear regression.
	linearRegression = linearRegressionAlgorithm(X, Y)
	W = linearRegression.learn()
	iterations = 1

	# Keep track of number of perceptron iterations.
	iterationsPerRun.append(iterations)

	# Calculate in-sample error E_in.
	inSampleError = numpy.array([classify(X[i], W) != Y[i] for i in range(nTrainingPoints)]).sum() / float(nTrainingPoints)
	inSampleErrorPerRun.append(inSampleError)

	if plot:
		line = weightsToPoints(W)
		plt.plot(line[:,0], line[:,1], 'g')

	print ""
	print "Run {num:g}".format(num=run)
	print "   Training completed in {num:g} iterations.".format(num=iterations)

	# Produce the training set of inputs X and outputs Y.
	Xvalidation = numpy.array([   [ random.random()*2-1, random.random()*2-1 ] for i in range(nValidationPoints)])
	YvalidationF = [1 for i in range(nValidationPoints)]
	YvalidationG = [1 for i in range(nValidationPoints)]
	outOfSampleErrorCounter = 0.
	for i in range(nValidationPoints):
		YvalidationF[i] = (classify(Xvalidation[i], WforF))
		YvalidationG[i] = (classify(Xvalidation[i], W))
		if YvalidationF[i] != YvalidationG[i]:
			outOfSampleErrorCounter += 1
		if plot:
			if YvalidationG[i] == 1:
				plt.plot(Xvalidation[i,0], Xvalidation[i,1],  'ro', mfc='none', ms='3')
			else:
				plt.plot(Xvalidation[i,0], Xvalidation[i,1],  'bo', mfc='none', ms='3')

	# Calculate out-of-sample error E_out.
	outOfSampleError = outOfSampleErrorCounter/float(nValidationPoints)
	outOfSampleErrorPerRun.append(outOfSampleError)

	# Measure run time.
	runDuration = time.time() - startTimeRun
	totalTime = time.time() - startTime
	averageRunDuration = totalTime / (run+1.)

	# Output.
	print "   Out of sample error: {num:2.4f}".format(num=outOfSampleErrorCounter/nValidationPoints)
	print "   In sample error: {num:2.4f}".format(num=inSampleError)
	print "   Current run duration: {num:g}".format(num=int(runDuration))
	print "   Average run duration: {num:g}".format(num=int(averageRunDuration))
	print "   Approximate time to finish: {num:g}".format(num=(nRuns - (run + 1)) * averageRunDuration)
	print ""
	print "Average number of iterations until convergence: {num:2.4f}.".format(num=numpy.average(numpy.array(iterationsPerRun)))
	print "Average in-sample-error: {num:2.4f}".format(num=numpy.average(numpy.array(inSampleErrorPerRun)))
	print "Average out of sample error: {num:2.4f}".format(num=numpy.average(numpy.array(outOfSampleErrorPerRun)))


	if plot:
		plt.gca().set_xlim(-1,1)
		plt.gca().set_ylim(-1,1)
		plt.gca().set_aspect(1)
		plt.show()



