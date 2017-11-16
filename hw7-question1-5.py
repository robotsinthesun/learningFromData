#!/usr/bin/python
#coding: utf8

import numpy
from matplotlib import pyplot as plt
from learningAlgorithms import *

flagPlot = False

# Import the data.
dataIn = numpy.fromfile('./hw6-dataTrain', dtype=float, sep=' ').reshape(-1,3)
dataOut = numpy.fromfile('./hw6-dataTest', dtype=float, sep=' ').reshape(-1,3)

# Loop for questions 1-2 and 3-4.
EOutAll = []
for i in range(2):
	# Split into training and validation data.
	if i == 0:
		# Question 1-2:
		print "Using 25 training points, 10 validation points."
		lenTrain = 25
		XTrain = dataIn[:lenTrain,:2]
		YTrain = dataIn[:lenTrain,2]
		XVal = dataIn[lenTrain:,:2]
		YVal = dataIn[lenTrain:,2]
		XTest = dataOut[:,:2]
		YTest = dataOut[:,2]
	elif i == 1:
		# Question 3-4:
		print "Using 10 training points, 25 validation points."
		lenVal = 25
		XTrain = dataIn[lenVal:,:2]
		YTrain = dataIn[lenVal:,2]
		XVal = dataIn[:lenVal,:2]
		YVal = dataIn[:lenVal,2]
		XTest = dataOut[:,:2]
		YTest = dataOut[:,2]


	# Non-linear transform.
	ZTrain = numpy.vstack([XTrain[:,0], XTrain[:,1], numpy.power(XTrain[:,0],2), numpy.power(XTrain[:,1],2), XTrain[:,0]*XTrain[:,1], numpy.abs(XTrain[:,0] - XTrain[:,1]), numpy.abs(XTrain[:,0] + XTrain[:,1])]).T
	ZVal = numpy.vstack([XVal[:,0], XVal[:,1], numpy.power(XVal[:,0],2), numpy.power(XVal[:,1],2), XVal[:,0]*XVal[:,1], numpy.abs(XVal[:,0] - XVal[:,1]), numpy.abs(XVal[:,0] + XVal[:,1])]).T
	ZTest = numpy.vstack([XTest[:,0], XTest[:,1], numpy.power(XTest[:,0],2), numpy.power(XTest[:,1],2), XTest[:,0]*XTest[:,1], numpy.abs(XTest[:,0] - XTest[:,1]), numpy.abs(XTest[:,0] + XTest[:,1])]).T


	# Question 1.
	# Train with models of 3rd, 4th, 5th, 6th and 7th order transform.
	componentNames = ["1", "x1", "x2", "x1²", "x2²", "x1·x2", "|x1-x2|", "x1+x2"]
	kEValMin = 0
	EValMin = 1000
	kEOutMin = 0
	EOutMin = 1000
	for k in [3,4,5,6,7]:




		# Get current hypothesis subset.
		ZTrainCut = ZTrain[:,:k]
		ZValCut = ZVal[:,:k]
		ZTestCut = ZTest[:,:k]
		if flagPlot: print "   Training with k = {n:g},   phi = {c:s}".format(n=k, c="  ".join(componentNames[0:k+1]))


		# Find the weights using linear regression.
		linReg = linearRegressionAlgorithm(ZTrainCut, YTrain)
		W = linReg.learn()



		# Calc classification error on validation set.
		EVal = calcClassificationError(ZValCut, YVal, W)
		if flagPlot: print "      Validation classificiation error: {n:2.2f}.".format(n=EVal)
		# Calc classification error on out-of-sample set.
		EOut = calcClassificationError(ZTestCut, YTest, W)
		if flagPlot: print "      Out-of-sample classificiation error: {n:2.2f}.".format(n=EOut)
		if flagPlot: print " "

		# Track minimum validation classification error.
		if EVal < EValMin:
			kEValMin = k
			EValMin = EVal

		# Track minimum out-of-sample classification error.
		if EOut < EOutMin:
			kEOutMin = k
			EOutMin = EOut


		if flagPlot:
			# Plot all in-sample data with given classification to verify the solution.
			for x, y in zip(dataIn[:,:2], dataIn[:,2]):
				if y > 0:
					plt.plot(x[0], x[1], 'ro', ms="7", markerfacecolor="None")
				else:
					plt.plot(x[0], x[1], 'bo', ms="7", markerfacecolor="None")

			# Plot training set.
			for xTrain, yTrain in zip(XTrain, classifySign(ZTrainCut, W)):
				if yTrain > 0:
					plt.plot(xTrain[0], xTrain[1], 'r.', ms="3")
				else:
					plt.plot(xTrain[0], xTrain[1], 'b.', ms="3")

			# Plot validation set.
			for xVal, yVal in zip(XVal, classifySign(ZValCut, W)):
				if yVal > 0:
					plt.plot(xVal[0], xVal[1], 'r.')
				else:
					plt.plot(xVal[0], xVal[1], 'b.')

			'''
			# Plot out-of-sample set with given classification to verify the solution.
			for xVal, yVal in zip(XTest, YTest):
				if yVal > 0:
					plt.plot(xVal[0], xVal[1], 'rs', markerfacecolor="None")
				else:
					plt.plot(xVal[0], xVal[1], 'bs', markerfacecolor="None")

			# Plot out-of-sample set with learned classification.
			for xVal, yVal in zip(XTest, classifySign(ZTestCut, W)):
				if yVal > 0:
					plt.plot(xVal[0], xVal[1], 'rs', ms="1")
				else:
					plt.plot(xVal[0], xVal[1], 'bs', ms="1")
			'''
			plt.show()

	print "   Minimum validation classificiation error: {n:2.2f} found for k = {k:g}.".format(n=EValMin, k=kEValMin)
	print "   Minimum out-of-sample classificiation error: {n:2.2f} found for k = {k:g}.".format(n=EOutMin, k=kEOutMin)

	EOutAll.append(EOutMin)

# Question 5.
choices = [	[0.0,0.1],
			[0.1,0.2],
			[0.1,0.3],
			[0.2,0.2],
			[0.2,0.3]]
dists = [100,100,100,100,100]
for choice, i in zip(choices, range(5)):
	dists[i] = numpy.linalg.norm(numpy.array(EOutAll) - numpy.array(choice))

print "Question 5: answer {a:s}.".format(a=['a', 'b', 'c', 'd', 'e'][numpy.argsort(numpy.array(dists))[0]])
