#!/usr/bin/python
#coding: utf8

import time
import numpy
from matplotlib import pyplot as plt
from sklearn.svm import SVC

from learningAlgorithms import *

# Run parameters. **************************************************************
nTrainingPoints = 100	# 10 for question 8, 100 for question 9.
nTestPoints = 10000		# 1000 for question 8, 10000 for question 9.
nRuns = 1000
plot = False


# Target parameters. ***********************************************************
nIterationsAll = []
nIterationsAverageProgress = []
nSupportVectorsAll = []
classificationErrorSvmAll = []
classificationErrorPlaAll = []
nSvmBetterThanPla = 0
#crossEntropyErrorAverageProgress = []
#outOfSampleErrorAverageProgress = []
#outOfSampleErrorAll = []


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
	Y = numpy.ones(nTrainingPoints)
	Y = classifySign(X, WforF)

	XTest = numpy.random.random((nTestPoints,2))*2-1
	YTest = numpy.ones(nTestPoints)
	YTest = classifySign(XTest, WforF)


	# Break the run if all points are either zeros ore ones.
	if all(Y == -numpy.ones(nTrainingPoints)) or all(Y == numpy.ones(nTrainingPoints)):
		print "Only one class. Aborting."
	else:
		if plot:
			for i in range(nTrainingPoints):
				if Y[i] == 1:
					plt.plot(X[i,0], X[i,1],  'r.', ms='2')
				else:
					plt.plot(X[i,0], X[i,1],  'b.', ms='2')



		# Classify using perceptron learning algorithm.
		W = numpy.zeros(3)
		perceptron = perceptronAlgorithm(W, X, Y)
		WPla, iterations = perceptron.learn()
		YTestPla = classifySign(X, WPla)

		# Record number of iterations to average afterwards.
		nIterationsAll.append(iterations)


		# Classify using support vector machine.
		svc = SVC(C=10e15, kernel='linear')
		svc.fit(X,Y)
		YTestSvm = svc.predict(XTest)
		nSupportVectorsAll.append(svc.support_vectors_.shape[0])

		if plot:
			for i in range(nTrainingPoints):
				if YSvm[i] == 1:
					plt.plot(X[i,0], X[i,1],  'r+', ms='4')
				else:
					plt.plot(X[i,0], X[i,1],  'b+', ms='4')
				if YPla[i] == 1:
					plt.plot(X[i,0], X[i,1],  'ro', ms='5', markerfacecolor="None")
				else:
					plt.plot(X[i,0], X[i,1],  'bo', ms='5', markerfacecolor="None")

		classificationErrorSvm = 1 - numpy.sum(((YTestSvm * YTest) + 1) / 2.) / XTest.shape[0]
		classificationErrorPla = calcClassificationError(XTest, YTest, WPla)
		classificationErrorSvmAll.append(classificationErrorSvm)
		classificationErrorPlaAll.append(classificationErrorPla)
		if classificationErrorSvm < classificationErrorPla:
			nSvmBetterThanPla += 1.

		print "   SVM classification error: {esvm:2.2f}, PLA classification error: {epla:2.2f}".format(esvm=classificationErrorSvm, epla=classificationErrorPla)
		'''

		# Calculate out-of-sample error E_out.
		outOfSampleError = 	numpy.average(numpy.array(crossEntropyErrorAll))
		print "   Out of sample error: {n:2.4f}.".format(n=outOfSampleError)
		outOfSampleErrorAll.append(outOfSampleError)
		'''
		if plot:
			plt.gca().set_xlim(-1,1)
			plt.gca().set_ylim(-1,1)
			plt.gca().set_aspect(1)
			plt.show()
	'''
	nIterationsAverageProgress.append(numpy.average(numpy.array(nIterationsAll)))
	outOfSampleErrorAverageProgress.append(numpy.average(numpy.array(outOfSampleErrorAll)))
	'''
# Print results and plot.
print "Percentage of svm better than pla: {n:3.2f}.".format(n=(nSvmBetterThanPla / float(nRuns) * 100))
print "Average number of support vectors: {n:2.4f}.".format(n=numpy.average(numpy.array(nSupportVectorsAll)))
'''
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
'''
