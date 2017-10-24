#!/usr/bin/python
# coding: utf8

import sys
import numpy
from matplotlib import pyplot as plt
from regression import regression


# Build target function.
nPoints = 1000
f = numpy.vstack([numpy.linspace(-1,1,nPoints), numpy.sin(numpy.pi*numpy.linspace(-1,1,nPoints))]).T

nRuns = 1000000

# Create hypotheses in the form of
# coefficients of the polynomial a0 + a1x + a2x^2.
hypothesisModels = [	[1, 0, 0],
			[0, 1, 0],
			[1, 1, 0],
			[0, 0, 1],
			[1, 0, 1]
			]
hypothesisModels = numpy.array(hypothesisModels)
modelNames = ['a_0', 'a_1x', 'a_1x + a_0', 'a_2x^2', 'a_2x^2 + a_0']

# Create figure for each hypothesis set and convergence.
fig, axes = plt.subplots(2,5)
axesIndex = 0

# Test the different models.
for model in hypothesisModels:
	coeffsSum = numpy.zeros(model.shape[0])
	gCoeffsAll = []
	coeffsHatProgression = []
	datasets = []
	graphs = []

	print "Testing model {c0:g}·a0 + {c1:g}·a1·x + {c2:g}·a2·x²".format(c0=model[0], c1=model[1], c2=model[2])
	for run in range(nRuns):
		# Pick two points and create data set.
		datasetX = numpy.random.rand(2)*2-1
		datasetY = numpy.sin(numpy.pi * datasetX)
		dataset = numpy.vstack([datasetX, datasetY]).T
		datasets.append(dataset)

		# Get the regression coefficients corresponding to the current hypothesis.
		coeffs = regression(dataset, model)

		# Sum up for average.
		coeffsSum += coeffs
		coeffsHatProgression.append(coeffsSum / (float(run)+1))
		gCoeffsAll.append(coeffs)

		# Print progress and plot g lines.
		if numpy.mod(run, nRuns/100) == 0:
			# Print progress.
			sys.stdout.write("   Progress: {num:3g}%\r".format(num=run/float(nRuns)*100))
			sys.stdout.flush()
			# Create graph.
			x = numpy.linspace(-1,1,100)
			y = coeffs[0] + coeffs[1]*x + coeffs[2]*numpy.power(x,numpy.ones(x.shape[0])*2.)
			graph = numpy.vstack([x,y]).T
			graphs.append(graph)

	print "   Progress: 100%."



	# Calc gBar by averaging all g.
	gBarCoeffs = numpy.average(numpy.array(gCoeffsAll), axis=0)
	gBar = numpy.vstack([f[:,0], gBarCoeffs[0] + f[:,0]*gBarCoeffs[1] + numpy.square(f[:,0])*gBarCoeffs[2]]).T
	print "   gBar(x) = {c0:2.2f} + {c1:2.2f}·x + {c2:2.2f}·x²".format(c0=gBarCoeffs[0], c1=gBarCoeffs[1], c2=gBarCoeffs[2])


	# Calculate bias.
	# First, calc squared error between gBar and f at a number of sample points.
	# Then, integrate all the squared errors over the solution space. 2/nPoints is the dx.
	# This is the mean square error between gBar and f.
	squaredError = numpy.square(gBar[:,1] - f[:,1]).sum() * (2. / float(nPoints))
	# To get the bias, we need to divide this by the length of the solution domain (WHY?)
	bias = squaredError * (1/2.)
	print "   Bias: {n:2.4f}.".format(n=bias)



	# Calculate variance.
	# Variance is the average squared error between all of the individual g's coming from the data sets and gBar.
	# For every g, we calculate the squared error between that g and gBar.
	meanSquaredError = numpy.zeros(nPoints)
	for i in range(nRuns):
		squaredError = numpy.square( (gCoeffsAll[i][0] + gCoeffsAll[i][1]*f[:,0] + gCoeffsAll[i][2]*numpy.square(f[:,0])) - gBar[:,1])
		meanSquaredError += squaredError / float(nRuns)
	# Then, we average the squared errors of all g's. This is the variance at every data point.
	varianceAll = meanSquaredError#numpy.average(squaredErrorAll, axis=0)
	# To get the total variance, average these over the solution space.
	variance = numpy.average(meanSquaredError)#numpy.average(varianceAll)
	print "   Variance: {n:2.4f}.".format(n=variance)



	# Calculate Eout = bias + variance
	print "   Eout: {n:2.4f}".format(n=bias+variance)



	# Plot target function, some of the g's and gBar
	# g's.
	for graph in graphs:
		axes[0,axesIndex].plot(graph[:,0], graph[:,1], color='gray', linewidth='0.1')
	# gBar.
	axes[0,axesIndex].plot(gBar[:,0], gBar[:,1], color='red', linewidth='0.6')
	# f.
	axes[0,axesIndex].plot(f[:,0], f[:,1], color='blue', linewidth='0.6')
	# Variance.
	axes[0,axesIndex].fill_between(gBar[:,0], gBar[:,1] + numpy.sqrt(varianceAll), gBar[:,1] - numpy.sqrt(varianceAll), facecolor='green', alpha=0.2)
	# Format plot.
	axes[0,axesIndex].set_title('${h:s}$'.format(h=modelNames[axesIndex]))
	axes[0,axesIndex].set_xlim([-1, 1])
	axes[0,axesIndex].set_ylim([-1.5, 1.5])
	axes[0,axesIndex].set_aspect(1)
	axes[0,axesIndex].grid()
	if axesIndex > 0:
		axes[0,axesIndex].set_yticklabels([])

	# Plot progress of change of gBar.
	for i in range(nRuns):
		coeffsHatProgression[i] -= gBarCoeffs
	axes[1,axesIndex].semilogy(numpy.abs(coeffsHatProgression), linewidth='.6')
	axes[1,axesIndex].plot([0,nRuns],[0,0], color='black', linewidth='.5')
	axes[1,axesIndex].set_xlim([0, nRuns])
	axes[1,axesIndex].set_ylim(10e-10, 10)
	if axesIndex > 0:
		axes[1,axesIndex].set_yticklabels([])
	# Move on to next subplot.
	axesIndex += 1

plt.show()
