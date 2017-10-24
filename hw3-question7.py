#!/usr/bin/python

import numpy

# Here, we try to find the number of dichotomies for the two-intervals model by brute force.


N = 10

runs = 20000

hypothethes = numpy.zeros(N)

for run in range(runs):
	# Place the four interval end points randomly within the sample.
	firstInterval = numpy.sort(numpy.random.randint(0, N+1, 2))	# N+2 in order to *include* N+1
	secondInterval = numpy.sort(numpy.random.randint(0, N+1, 2))

	#print "First interval"
	#print firstInterval

	#print "Second interval"
	#print secondInterval

	# Create data set and set ones.
	dataSet = numpy.zeros(N)
	# First interval.
	dataSet[firstInterval[0]:firstInterval[1]] = numpy.ones(firstInterval[1] - firstInterval[0])
	# Second interval.
	dataSet[secondInterval[0]:secondInterval[1]] = numpy.ones(secondInterval[1] - secondInterval[0])

	# Add to hypothethes.
	hypothethes = numpy.vstack((hypothethes, dataSet))

print "Number of dichotomies for N = {n:g}: {d:g}.".format(n=N, d=numpy.unique(hypothethes, axis=0).shape[0])
