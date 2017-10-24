#!/usr/bin/python

from math import *
import numpy
import matplotlib.pyplot as plt


datasetX = numpy.random.rand(2)
datasetY = numpy.sin(numpy.pi * datasetX)
dataset = numpy.vstack([datasetX, datasetY]).T
print dataset

#dataset = numpy.array([[0.5, -0.], [0.5, 1]])



def optimizeA(dataset, nTests):
	a0 = dataset[0,1] / dataset[0,0]
	a1 = dataset[1,1] / dataset[1,0]
	aIncr = (a1 - a0) / float(nTests)
	squaredErrorOld = 100000000
	aAll = [a0+(i+1)*aIncr for i in range(nTests)]
	aOpt = 0
	for a in aAll:
#		print "Squared error 0: {n:2.6f}".format(n = pow(dataset[0,1] - dataset[0,0] * a, 2))
#		print "Squared error 1: {n:2.6f}".format(n = pow(dataset[1,1] - dataset[1,0] * a, 2))
		#plt.plot([0,1], [0,a], color='gray', linewidth='0.1')
		squaredError = numpy.average(numpy.square(dataset[:,1] - dataset[:,0] * a))
		if squaredError < squaredErrorOld:
			aOpt = a
			squaredErrorOld = squaredError
	return aOpt


def optimizeANew(dataset, nMax):
	# Choose smaller of the two data point's a as initial a.
	a0 = dataset[0,1] / dataset[0,0]
	a1 = dataset[1,1] / dataset[1,0]
	if a0 < a1:
		aInitial = a0
	else:
		aInitial = a1

	aStep = 0.5 * abs(a1-a0)
	direction = 1
	rate = 0.1
	a = aInitial
	msqOld = 2*numpy.average(numpy.square(dataset[:,1] - dataset[:,0] * a))

	for n in range(nMax):
	#	print "iteration: {i:g}".format(i=n)
		# Calc mean square error.
		msq = numpy.average(numpy.square(dataset[:,1] - dataset[:,0] * a))
	#	print "msq: {n:2.6f}".format(n=msq)
		if msq > msqOld:
			print "Changing dir"
			direction *= -1
		a += aStep * direction
		aStep *= 0.5
	#	print "a step: {n:2.6f}".format(n=aStep)
	#	print "a: {n:2.4f}".format(n=a)
		diff = abs(msqOld - msq)
		if diff < 0.000001:
			break
		msqOld = msq
	return a

aOpt = optimizeA(dataset, 1000)
aOptNew = optimizeANew(dataset, 1000)
print aOpt
print aOptNew

plt.plot(dataset[0,0], dataset[0,1], 'ro')
plt.plot(dataset[1,0], dataset[1,1], 'go')
plt.plot([0,1], [0,aOpt], linewidth='0.2', color='red')
plt.plot([0,1], [0,aOptNew], linewidth='0.2', color='green')
plt.xlim(0,1)
plt.ylim(-1,1)
plt.grid(True)
plt.show()
