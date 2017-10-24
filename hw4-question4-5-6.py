#!/usr/bin/python

from math import *
import numpy
import matplotlib.pyplot as plt

nPoints = 1000
f = numpy.vstack([numpy.linspace(-1,1,nPoints), numpy.sin(numpy.pi*numpy.linspace(-1,1,nPoints))]).T

nRuns = 100000

def optimizeA(dataset, nTests):
	a0 = dataset[0,1] / dataset[0,0]
	a1 = dataset[1,1] / dataset[1,0]
	aIncr = (a1 - a0) / float(nTests)
	squaredErrorOld = 100000000
	aAll = [a0+(i+1)*aIncr for i in range(nTests)]
	aOpt = 0
	for a in aAll:
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
			#print "Changing dir"
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


aSum = 0
aAll = []
datasets = []
aHatAll = []
for run in range(nRuns):
	# Pick two points.
	#dataset = numpy.vstack([f[numpy.random.randint(nPoints)], f[numpy.random.randint(nPoints)]])
	datasetX = numpy.random.rand(2)*2-1
	datasetY = numpy.sin(numpy.pi * datasetX)
	dataset = numpy.vstack([datasetX, datasetY]).T
	datasets.append(dataset)

	# Get the slope of a line producing the smallest mean square error.
	a = optimizeANew(dataset, 1000)

	# Sum up for average.
	aSum += a
	aAll.append(a)
	aHatAll.append(aSum / (float(run)+1))



	# Print progress and plot g lines.
	if numpy.mod(run, nRuns/100) == 0:
		print run/float(nRuns)*100
		line = [[-1, 1],[a*-1, a]]
		plt.plot(line[0], line[1], color='gray', linewidth='0.1')


# Average a.
aHat = aSum / nRuns#numpy.average(numpy.array(aAll))
print "gBar = {a:2.2f}x".format(a=aHat)

# Plot all the g's, gBar and the target function f.
plt.plot([-1, 1], [-1*aHat, aHat], color='red', linewidth='0.5')
plt.plot(f[:,0], f[:,1])



# Question 5. ******************************************************************
# Create gBar data.
gBar = numpy.vstack([numpy.linspace(-1,1,nPoints), aHat*numpy.linspace(-1,1,nPoints)]).T

# Calculate bias.
# First, calc squared error between gBar and f at a number of sample points.
# Then, integrate all the squared errors over the solution space. 2/nPoints is the dx.
# This is the mean square error between gBar and f.
squaredError = numpy.square(gBar[:,1] - f[:,1]).sum() * (2. / float(nPoints))
# To get the bias, we need to divide this by the length of the solution domain (WHY?)
bias = squaredError * (1/2.)
print "Bias: {n:2.4f}.".format(n=bias)



# Question 6. ******************************************************************
# Calculate variance.
# Variance is the average squared error between all of the individual g's coming from the data sets and gBar.
# For every g, we calculate the squared error between that g and gBar.
squaredErrorAll = []
for i in range(nPoints):
	a = aAll[i]
	g = numpy.vstack([numpy.linspace(-1,1,nPoints), a*numpy.linspace(-1,1,nPoints)]).T
	squaredErrorAll.append(numpy.square(g[:,1] - gBar[:,1]))
squaredErrorAll = numpy.vstack(squaredErrorAll)
# Then, we average the squared errors of all g's. This is the variance at every data point.
varianceAll = numpy.average(squaredErrorAll, axis=0)
plt.plot(gBar[:,0], gBar[:,0]*aHat + numpy.sqrt(varianceAll))
plt.plot(gBar[:,0], gBar[:,0]*aHat - numpy.sqrt(varianceAll))
# To get the total variance, average these over the solution space.
variance = numpy.average(varianceAll)
print "Variance: {n:2.4f}.".format(n=variance)

plt.grid(True)
plt.show()


# Plot the averaging progress to check that we have arrived in a steady region.
#plt.plot(aHatAll, linewidth='.2')
#plt.ylim(floor(aHat-0.01), ceil(aHat+0.01))
#plt.grid(True)
#plt.show()
