#!/usr/bin/python
#coding: utf8

import time
import numpy
from matplotlib import pyplot as plt

from learningAlgorithms import *

# Run parameters. **************************************************************
nTrainingPoints = 100
nValidationPoints = 1000
nRuns = 1000
plot = False


# Target parameters. ***********************************************************
nIterationsAll = []
nIterationsAverageProgress = []
crossEntropyErrorAll = []
crossEntropyErrorAverageProgress = []
outOfSampleErrorAverageProgress = []
outOfSampleErrorAll = []


# Run the learning process multiple times. *************************************
startTime = time.time()
for run in range(nRuns):
	print "Run " + str(run) + "."
	startTimeRun = time.time()

	# Create target function f.
	# Create random points.
	p1 = numpy.random.random(2)*2-1
	p2 = numpy.random.random(2)*2-1
	# Calc weights for target function.
	WforF = pointsToWeights(p1, p2)
	# Plot points and line.
	if plot:
		line = weightsToPoints(WforF)
		plt.plot(line[:,0], line[:,1], 'k')
		plt.plot([p1[0], p2[0]],[p1[1], p2[1]], 'k.')


	# Produce the training set of inputs X and outputs Y.
	X = numpy.random.random((nTrainingPoints,2))*2-1
	Y = [1 for i in range(nTrainingPoints)]
	for i in range(nTrainingPoints):
		Y[i] = classifySign(X[i], WforF)
		Y = numpy.array(Y)
		if plot:
			if Y[i] == 1:
				plt.plot(X[i,0], X[i,1],  'r.', ms='2')
			else:
				plt.plot(X[i,0], X[i,1],  'b.', ms='2')


	# Approximate f by learning the weights w for the final hypothesis g using logarithmic regression.
	logarithmicRegression = logarithmicRegressionAlgorithm(X, Y)
	W, iterations = logarithmicRegression.learn()

	# Record number of iterations to average afterwards.
	nIterationsAll.append(iterations)

	# Create validation points and classify them.
	Xvalidation = numpy.random.random((nValidationPoints,2))*2-1
	YvalidationF = [1 for i in range(nValidationPoints)]
	YvalidationG = [1 for i in range(nValidationPoints)]
	for i in range(nValidationPoints):
		YvalidationF[i] = classifySign(Xvalidation[i], WforF)
		YvalidationG[i] = logarithmicRegression.classify(Xvalidation[i], W)
		crossEntropyErrorAll.append(logarithmicRegression.calcCrossEntropyError(Xvalidation[i], YvalidationF[i], W))
		if plot:
			plt.scatter(Xvalidation[i,0], Xvalidation[i,1], c=plt.cm.bwr(YvalidationG[i]))#, norm=norm)#, mfc='none', ms='3')


	# Calculate out-of-sample error E_out.
	outOfSampleError = 	numpy.average(numpy.array(crossEntropyErrorAll))
	print "   Out of sample error: {n:2.4f}.".format(n=outOfSampleError)
	outOfSampleErrorAll.append(outOfSampleError)

	if plot:
		plt.gca().set_xlim(-1,1)
		plt.gca().set_ylim(-1,1)
		plt.gca().set_aspect(1)
		plt.show()

	nIterationsAverageProgress.append(numpy.average(numpy.array(nIterationsAll)))
	outOfSampleErrorAverageProgress.append(numpy.average(numpy.array(outOfSampleErrorAll)))


# Print results and plot.
print "Average cross entropy out of sample error: {n:2.4f}.".format(n=numpy.average(numpy.array(outOfSampleErrorAll)))
print "Average number of iterations for gradient descent: {n:g}.".format(n=numpy.average(numpy.array(nIterationsAll)))

plotIter = plt.plot(range(len(nIterationsAverageProgress)), nIterationsAverageProgress, label='Avg. No. iterations')
plt.gca().set_xlabel('Runs')
plt.gca().set_ylabel('Average number of iterations')
plt.gca().set_ylim(320, 400)
plt.yticks(numpy.arange(0, 6)*20+300)
plt.gca().grid(True)
plt.gca().twinx()
plotError = plt.plot(range(len(outOfSampleErrorAverageProgress)), outOfSampleErrorAverageProgress, 'r-', label='Avg. $E_{out}$')
plt.gca().set_ylabel('Out of sample error $E_{out}$')#, color='r')
plt.gca().set_ylim(.07, .12)
plt.yticks(numpy.arange(0, 6)*0.01+0.07)
plt.gca().grid(True)
plots = plotIter+plotError
labels = [p.get_label() for p in plots]
plt.legend(plots, labels)
plt.grid(True)
plt.show()
