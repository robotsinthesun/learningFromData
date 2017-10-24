#!/usr/bin/python

from math import *
import numpy
from matplotlib import pyplot as plt

# Function that returns coefficients of a polynomial regression.
# Specify the coefficients to solve for in the coeffFactors array.
# For example, for y = a0 + a2x2 use [1,0,1].
def regression(data, coeffFactors):
	# Basic approach:
	# y = X * A + e.
	# y is response vector, X is input matrix containing all the x values,
	# A is the parameter vector and e is the error vector.
	# See here: https://youtube.com/watch?v=Qa_FI92_qo8
	# We are looking for a polynomial expression like this:
	# y = a0 + a1x + a2x^2 + ... + anx^n.
	# In matrix form, this can be written as
	# y0 = | 1 x0 x0^2 ... x0^n |     |a0|     |e0|
	# y1 = | 1 x1 x1^2 ... x1^n |  *  |a1|  +  |e1|
	# y2 = | 1 x2 x2^2 ... x2^n |     |a2|     |e2|
	# We now need to solve this for A.

	# Build the vectors and matrices we need.
	nCols = data.shape[0]
	# Response vector.
	y = data[:,1]
	# Design matrix.
	X = numpy.ones(nCols)
	for i in range(coeffFactors.shape[0]-1):
		X = numpy.vstack([X, numpy.power(data[:,0], numpy.ones(nCols)*(i+1.))])
	X = X.T
	# Here, we disable the terms with a zero in the coeffFactors input parameter.
	X = X * coeffFactors

	# Now, solve for A using the pseudo inverse.
	pseudoInverseX = numpy.linalg.pinv(X)
	A = numpy.dot(pseudoInverseX, y)

	# Return the coefficients.
	return A



if __name__ == '__main__':

	dataset = numpy.array([[0.8, 0.7], [0.2, 0.4]])

	coeffs = regression(dataset, coeffFactors=numpy.array([1,0,0]))
	print coeffs

	# Plot the result.
	nPoints = 100
	x = numpy.linspace(0, 1, nPoints)
	y = []
	for i in range(nPoints):
		xPowers = numpy.power(x[i], numpy.array(range(coeffs.shape[0])))
		y.append(numpy.dot(xPowers, coeffs))
	y = numpy.array(y)


	plt.plot(x,y, linewidth='0.1', color='red')
	plt.plot(dataset[:,0], dataset[:,1], 'k.')
	plt.xlim([0,1])
	plt.ylim([0,1])
	plt.grid()
	plt.gca().set_aspect(1)
	plt.show()
