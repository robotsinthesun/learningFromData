#!/usr/bin/python
#coding: utf8

import time
import numpy
from matplotlib import pyplot as plt

from learningAlgorithms import *

fig, axes = plt.subplots(1,2)

# Create some points that might match the classification bounds.
#numpy.random.seed(1)
X = numpy.random.rand(50,2)*2-1
Y = []
for i in range(X.shape[0]):
	if numpy.linalg.norm(numpy.array([1,0]) - numpy.abs(X[i])) > 0.7 and numpy.abs(X[i,0]) < 0.8:
		Y.append(1)
		X[i,0] *= 0.6
		axes[0].plot(X[i,0], X[i,1], 'r.')
	else:
		Y.append(-1)
		axes[0].plot(X[i,0], X[i,1], 'b.')


# Transform the points into Z-space.
Z = numpy.power(X,2)

for i in range(Z.shape[0]):
	if Y[i] > 0:
		axes[1].plot(Z[i, 0], Z[i, 1], 'r.')
	else:
		axes[1].plot(Z[i, 0], Z[i, 1], 'b.')

# Run perceptron in Z space.
W = numpy.array([1. ,0. ,0.])
perceptron = perceptronAlgorithm(W, Z, Y)
W, iterations = perceptron.learn()

print "Learned weight vector in Z space: [{w0:2.2f}, {w1:2.2f}, {w2:2.2f}].".format(w0 = W[0], w1 = W[1], w2 = W[2])

# Create some test points in Z space.
ZValidation = numpy.random.rand(100,2)
for z in ZValidation:
	yValidation = perceptron.classify(z, W)
	if yValidation > 0:
		axes[1].plot(z[0], z[1], 'r.', ms="1.5")
	else:
		axes[1].plot(z[0], z[1], 'b.', ms="1.5")

# Create some test points in X space.
XValidation = numpy.random.rand(500,2)*2-1
for x in XValidation:
	yValidation = perceptron.classify(numpy.power(x,2), W)

	if yValidation > 0:
		axes[0].plot(x[0], x[1], 'r.', ms="1.5")
	else:
		axes[0].plot(x[0], x[1], 'b.', ms="1.5")



# Transform W to the line.
#classificationBoundaryPoints = weightsToPoints(W)
#axes[1].plot(classificationBoundaryPoints[:,0], classificationBoundaryPoints[:,1])

axes[0].set_xlim(-1,1)
axes[0].set_ylim(-1,1)
axes[0].set_xlabel("$x_1$")
axes[0].set_ylabel("$x_2$")
axes[0].set_aspect(1)
axes[0].set_title('$\mathcal{X}$')
axes[0].grid()

axes[1].set_xlim(0,1)
axes[1].set_ylim(0,1)
axes[1].set_xlabel("$z_1$")
axes[1].set_ylabel("$z_2$")
axes[1].set_aspect(1)
axes[1].set_title('$\mathcal{Z}$')
axes[1].grid()
plt.tight_layout()
plt.show()

