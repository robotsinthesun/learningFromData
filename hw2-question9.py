#!/usr/bin/python
# coding: utf8

import time
import random
import numpy
import matplotlib.pyplot as plt

from learningAlgorithms import *


# Run parameters. **************************************************************
nTrainingPoints = 5000
noisyFraction = 0.1
nValidationPoints = 1000
nRuns = 1000
plot = False

# Test results. ****************************************************************
outOfSampleErrorPerRun = []
inSampleErrorPerRun = []
suggestedWMatchesPerRun = []


def classifyNonLinear(x):
	return numpy.sign(pow(x[0],2) + pow(x[1],2) - 0.6)


# Run the learning process multiple times.
startTime = time.time()
for run in range(nRuns):

	startTimeRun = time.time()


	# Create target function f.
	# f(x1, x2) = sign(x1² + x2² - 0.6)
	# Create random points.
#	p1 = ([ random.random()*2-1, random.random()*2-1 ])
#	p2 = ([ random.random()*2-1, random.random()*2-1 ])
	# Calc weights for target function.
#	WforF = pointsToWeights(p1, p2)
	# Plot points and line.
#	if plot:
#		line = weightsToPoints(WforF)
#		plt.plot(line[:,0], line[:,1], 'k')
#		plt.plot([p1[0], p2[0]],[p1[1], p2[1]], 'ko')


	# Produce the training set of inputs X and outputs Y.
	X = numpy.array([   [ random.random()*2-1, random.random()*2-1 ] for i in range(nTrainingPoints)])
	Y = numpy.ones(nTrainingPoints)
	for i in range(nTrainingPoints):
		Y[i] =  classifyNonLinear(X[i])	#(classify(X[i], WforF))
		# Flip y randomly within noise fraction.
		if numpy.random.rand() <= noisyFraction:
			Y[i] = Y[i] * (-1)
		if plot:
			if Y[i] == 1:
				plt.plot(X[i,0], X[i,1],  'ro')
			else:
				plt.plot(X[i,0], X[i,1],  'bo')

	# Produce feature vectors. Leave out first all-one column as the linear regression class adds it...
	Z = numpy.vstack( [   X[:,0],   X[:,1],   X[:,0] * X[:,1],   numpy.square(X[:,0]),   numpy.square(X[:,1])   ] ).T

	# Approximate f by learning the weights W for g using linear regression.
	# Note: We feed the non-linearly transformed training data Z into the algorithm.
	linearRegression = linearRegressionAlgorithm(Z, Y)
	W = linearRegression.learn()

	# Classify a randomly selected point using the learned weights.
	randomPointIndex = numpy.random.randint(0,nTrainingPoints)
	yRandomPoint = classify(Z[randomPointIndex], W)

	# Classify that same point using the weights suggested in the answers.
	suggestedW = numpy.array([	[-1, -0.05, 0.08, 0.13, 1.5, 1.5],
								[-1, -0.05, 0.08, 0.13, 1.5, 15],
								[-1, -0.05, 0.08, 0.13, 15, 1.5],
								[-1, -1.5, 0.08, 0.13, 0.05, 0.05],
								[-1, -0.05, 0.08, 1.5, 0.15, 0.15]
								])
	suggestedWMatches = []
	for WCurrent in suggestedW:
		# Test for agreement between classification by learned W and current suggested W.
		# For agreement, the product will be +1, for disagreement it will be -1.
		# Add 1 div by 2 to turn -1 into 0 and 1 into 1.
		suggestedWMatches.append(((classify(Z[randomPointIndex], WCurrent) * yRandomPoint) + 1) / 2)

	suggestedWMatchesPerRun.append(suggestedWMatches)


	print ""
	print "Run {num:g}".format(num=run)



	# Produce the test set of inputs X and outputs Y.
	# Create X and transform non-linearly.
	XValidation = numpy.array([   [ random.random()*2-1, random.random()*2-1 ] for i in range(nValidationPoints)])
	ZValidation = numpy.vstack( [   XValidation[:,0],   XValidation[:,1],   XValidation[:,0] * XValidation[:,1],   numpy.square(XValidation[:,0]),   numpy.square(XValidation[:,1])   ] ).T
	YNoiseFactor = ((numpy.random.rand(1000)>0.1) -1) * 2 + 1
	# Create
	YValidationF = numpy.array([classifyNonLinear(XValidation[i]) for i in range(nValidationPoints)]) * YNoiseFactor
	YValidationG = numpy.ones(nValidationPoints)#[1 for i in range(nValidationPoints)]
	outOfSampleErrorCounter = 0.
	# Add noise and classify.
	for i in range(nValidationPoints):
		YValidationG[i] = (classify(ZValidation[i], W))
		if YValidationF[i] != YValidationG[i]:
			outOfSampleErrorCounter += 1
		if plot:
			if YValidationG[i] == 1:
				plt.plot(XValidation[i,0], XValidation[i,1],  'ro', mfc='none', ms='3')
			else:
				plt.plot(XValidation[i,0], XValidation[i,1],  'bo', mfc='none', ms='3')

	# Calculate out-of-sample error E_out.
	outOfSampleError = outOfSampleErrorCounter/float(nValidationPoints)
	outOfSampleErrorPerRun.append(outOfSampleError)

	# Measure run time.
	runDuration = time.time() - startTimeRun
	totalTime = time.time() - startTime
	averageRunDuration = totalTime / (run+1.)

	# Output.
	print "   Out of sample error: {num:2.4f}".format(num=outOfSampleErrorCounter/nValidationPoints)
#	print "   In sample error: {num:2.4f}".format(num=inSampleError)
	print "   Current run duration: {num:g}".format(num=int(runDuration))
	print "   Average run duration: {num:g}".format(num=int(averageRunDuration))
	print "   Approximate time to finish: {num:g}".format(num=(nRuns - (run + 1)) * averageRunDuration)
	print ""
#	print "Average in-sample-error: {num:2.4f}".format(num=numpy.average(numpy.array(inSampleErrorPerRun)))
	print "Average out of sample error: {num:2.4f}".format(num=numpy.average(numpy.array(outOfSampleErrorPerRun)))
	averageAgreement = numpy.average(numpy.array(suggestedWMatchesPerRun), axis=0)
	print "Average agreement between suggested and learned weights: {h0:g}, {h1:g}, {h2:g}, {h3:g}, {h4:g}.".format(h0 = averageAgreement[0], h1 = averageAgreement[1], h2 = averageAgreement[2], h3 = averageAgreement[3], h4 = averageAgreement[4])

	if plot:
		plt.gca().set_xlim(-1,1)
		plt.gca().set_ylim(-1,1)
		plt.gca().set_aspect(1)
		plt.show()



