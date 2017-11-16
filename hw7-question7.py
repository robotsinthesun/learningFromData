#!/usr/bin/python
#coding: utf8

import numpy
from matplotlib import pyplot as plt
from learningAlgorithms import *
from regression import *

rhos = numpy.arange(10000)*0.0005
rhoChoices = numpy.array(	[	numpy.sqrt(numpy.sqrt(3)+4),
								numpy.sqrt(numpy.sqrt(3)-1),
								numpy.sqrt(9+4*numpy.sqrt(6)),
								numpy.sqrt(9-numpy.sqrt(6))
								])
#print rhoChoices
errorDiffsAll = []
for rho in rhos:
	points = numpy.array([	[-1, 0],
							[rho, 1],
							[1,0]])

	# Leave one out three (each point once).
	EValConstAll = []
	EValLinAll = []
	#colors = ['r', 'b', 'g']
	for i in range(3):
		pointVal = points[i]
		pointsTrain = numpy.delete(points, i, axis=0)
		# Do the regression
		coeffsConst = regression(pointsTrain, coeffFactors=numpy.array([1,0]))
		coeffsLin = regression(pointsTrain, coeffFactors=numpy.array([1,1]))
		# Get the errors and sum up.
		EValConstAll.append(calcMeanSquareError(pointVal[0], pointVal[1], coeffsConst))
		EValLinAll.append(calcMeanSquareError(pointVal[0], pointVal[1], coeffsLin))
		# Plot.
		#plt.plot(pointsTrain[:,0], pointsTrain[:,1], colors[i]+'o')
		#plt.plot([-1, 1], [coeffsConst[0], coeffsConst[0]], colors[i]+'-')
	plt.show()
	# Average errors.
	EValConst = numpy.average(numpy.array(EValConstAll))
	EValLin = numpy.average(numpy.array(EValLinAll))

	errorDiffsAll.append(abs(EValLin - EValConst))
errorDiffsAll = numpy.array(errorDiffsAll)
rhoFinal = rhos[numpy.argsort(errorDiffsAll)[0]]
errorDiffFinal = numpy.min(errorDiffsAll)
print "Model validation error difference of {d:2.6f} found for rho = {r:2.4f}.".format(d=errorDiffFinal, r=rhoFinal)

plt.plot(rhos, errorDiffsAll)
for choice, label, color in zip(rhoChoices, ['a', 'b', 'c', 'd'], ['r', 'g', 'b', 'm']):
	plt.plot([choice, choice], [0,1], "{c:s}-".format(c=color), label=label)
plt.legend()
plt.show()


points = numpy.array([	[-1, 0],
						[rhoFinal, 1],
						[1,0]])
plt.plot(points[:,0], points[:,1], 'r.')
plt.plot([-1, numpy.ceil(rhoFinal)], [coeffsConst[0], coeffsConst[0]], 'g-')
plt.plot([-1,numpy.ceil(rhoFinal)], [coeffsLin[0]+coeffsLin[1]*-1, coeffsLin[0]+coeffsLin[1]*numpy.ceil(rhoFinal)], 'b-')
plt.show()
'''
	if abs(EValLin - EValConst) < 0.01:
		print "Tie!"
		print "Rho = {n:g}.".format(n=rho)

		plt.plot(points[:,0], points[:,1], 'ro')
		plt.plot([-1,1], [coeffsConst[0], coeffsConst[0]], 'g-')
		plt.plot([-1,1], [coeffsLin[0]+coeffsLin[1]*-1, coeffsLin[0]+coeffsLin[1]*1], 'b-')
		plt.show()
'''
