#!/usr/bin/python
#coding: utf8

import numpy
from matplotlib import pyplot as plt

from learningAlgorithms import *

# Import the data.
dataTrain = numpy.fromfile('./hw6-dataTrain', dtype=float, sep=' ').reshape(-1,3)
dataTest = numpy.fromfile('./hw6-dataTest', dtype=float, sep=' ').reshape(-1,3)

XTrain = dataTrain[:,0:2]
YTrain = dataTrain[:,2]

XTest = dataTest[:,0:2]
YTest = dataTest[:,2]

# Transform to Z-space.
ZTrain = numpy.vstack([XTrain[:,0], XTrain[:,1], numpy.power(XTrain[:,0],2), numpy.power(XTrain[:,1],2), XTrain[:,0]*XTrain[:,1], numpy.abs(XTrain[:,0] - XTrain[:,1]), numpy.abs(XTrain[:,0] + XTrain[:,1])]).T
ZTest = numpy.vstack([XTest[:,0], XTest[:,1], numpy.power(XTest[:,0],2), numpy.power(XTest[:,1],2), XTest[:,0]*XTest[:,1], numpy.abs(XTest[:,0] - XTest[:,1]), numpy.abs(XTest[:,0] + XTest[:,1])]).T


# Find the weights using linear regression.
linReg = linearRegressionAlgorithm(ZTrain, YTrain)
W = linReg.learn()


# Plot the points in Z space.
'''
for x, y in zip(XTrain, YTrain):
	if y < 0:
		plt.plot(numpy.power(x[0],2), numpy.power(x[1],2), 'b.')
	else:
		plt.plot(numpy.power(x[0],2), numpy.power(x[1],2), 'r.')
# Plot h(x) in Z space.
xplot = numpy.linspace(0,1,100)
yplot = [ W[0] + W[1] * x + W[2] * pow(x, 2) + W[3] * pow(x, 2) + W[2] * pow(x, 2) + W[2] * pow(x, 2) + W[2] * pow(x, 2) for x in xplot]
plt.plot(xplot, yplot)
plt.show()
'''

# Classify the points using the perceptron sign classifier.
print ""
print "Question 2: "
print "   In-sample classificiation error: {n:2.2f}.".format(n=calcClassificationError(ZTrain, YTrain, W))
print "   Out-of-sample classificiation error: {n:2.2f}.".format(n=calcClassificationError(ZTest, YTest, W))


# Run linear regression again using a regularized version.
# lambda = 10ek
print ""
print "Question 3, 4 and 5:"
for k in [-3., -2., -1., 0., 1., 2., 3.]:
	Wreg = linReg.learn(weightDecayLambda=pow(10, k))
	print "   k = {n:g}".format(n=k)
	print "      In-sample classificiation error: {n:2.2f}.".format(n=calcClassificationError(ZTrain, YTrain, Wreg))
	print "      Out-of-sample classificiation error: {n:2.2f}.".format(n=calcClassificationError(ZTest, YTest, Wreg))


print ""
print "Question 6:"
print "   Minimum out-of-sample error: {n:2.2f}.".format(n = numpy.min(numpy.array([calcClassificationError(ZTest, YTest, linReg.learn(weightDecayLambda=pow(10, k))) for k in numpy.linspace(-100, 100, 201)])))
