#!/usr/bin/python

import numpy

epsilon = 0.05
Pmax = 0.03

for M in [1, 10, 100]:
	for N in range(10000):
		if (2 * M * numpy.exp(-2 * pow(epsilon,2) * N)) <= Pmax:
			print "Minimum N for M = {m:g}: {n:g}.".format(m=M, n=N)
			break
