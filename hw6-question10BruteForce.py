#!/usr/bin/python
#coding: utf

import numpy
from matplotlib import pyplot as plt

nInputUnits = 10
nOutputUnits = 1
nHiddenUnits = 36
nHiddenLayersMax = 5

nConnectionsMax = 0
nUnitsMax = []
nConnectionsAll = []
nUnitsAll = []
nRuns = 100000

for run in range(nRuns):
	print "Run {r:g} of {n:g}.".format(r=run+1, n=nRuns)
	# Random number of layers.
	nHiddenLayers = numpy.random.randint(1,nHiddenLayersMax+1)
	print "   Number of hidden layers: {n:g}.".format(n=nHiddenLayers)

	# Units per layer.
	nUnits = [nInputUnits]
	nUnitsRemaining = nHiddenUnits-nHiddenLayers*2
	for layer in range(nHiddenLayers):
		if layer < nHiddenLayers-1:
			nUnitsNew = numpy.random.randint(0,nUnitsRemaining+1)
			nUnits.append(nUnitsNew+2)
			nUnitsRemaining -= nUnitsNew
		else:
			nUnits.append(nUnitsRemaining+2)
	nUnits.append(nOutputUnits)
	nUnits = numpy.array(nUnits)
	print "   Number of units per layer: " + str(nUnits)
	print "   Number of hidden units: " + str(numpy.sum(numpy.array(nUnits[1:-1])))


	# Calc number of connections.
	nConnections = 0
	for layer in range(nHiddenLayers):
		nConnections += nUnits[layer] * (nUnits[layer+1]-1)
		print "   Layer " + str(layer) + ": " + str(nUnits[layer] * (nUnits[layer+1]-1)) + " connections."
	nConnections += nUnits[-2] * nUnits[-1]

	if nConnections > nConnectionsMax:
		nConnectionsMax = nConnections
		nUnitsMax = nUnits
	print "   Current number of nConnections: {n:g}, max: {m:g} at " .format(n=nConnections, m=nConnectionsMax) + str(nUnitsMax)

