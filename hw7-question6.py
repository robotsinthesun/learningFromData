#!/usr/bin/python
#coding: utf8

import numpy

nSamples = 10000000
e1 = numpy.random.rand(nSamples)
e2 = numpy.random.rand(nSamples)
print "Expected value e1: " + str(numpy.average(e1)) + "."
print "Expected value e1: " + str(numpy.average(e2)) + "."
expMin = numpy.average(numpy.min(numpy.vstack([e1, e2]), axis=0))
print "Expected value min(e1, e2): " + str(expMin) + "."
print "Distance form 0.25: " + str(abs(0.25 - expMin))
print "Distance from 0.40: " + str(abs(0.4 - expMin))

