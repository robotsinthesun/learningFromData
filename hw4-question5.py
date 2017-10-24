#!/usr/bin/python

from math import *
import numpy
import matplotlib.pyplot as plt


nPoints = 1000
a =  1.46
f = numpy.vstack([numpy.linspace(-1,1,nPoints), numpy.sin(numpy.pi*numpy.linspace(-1,1,nPoints))]).T
gBar = numpy.vstack([numpy.linspace(-1,1,nPoints), a*numpy.linspace(-1,1,nPoints)]).T

# Calculate bias.
squaredError = numpy.square(gBar[:,1] - f[:,1]).sum() * (2. / float(nPoints))
bias = squaredError * (1/2.)
print "Bias: {n:2.2f}.".format(n=bias)

# Calculate variance.




plt.plot(f[:,0], f[:,1])
plt.plot(gBar[:,0], gBar[:,1], linewidth='0.2', color='red')
plt.grid(True)
plt.show()
