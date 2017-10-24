#!/usr/bin/python

import numpy

def combination(n,k):
	if k > n:
		return 0
	else:
		return numpy.math.factorial(n) / (numpy.math.factorial(k) * numpy.math.factorial(n-k))

# Try what happens for different q.
for q in range(1,10):
	print "Testing q = {n:g}.".format(n=q)
	breakPoint = 1
	# Solve m for the first 10 N.
	m = []
	# Set m for N = 1.
	m.append(2)
	# Calc m for the rest of the N with given q.
	# Also, test for break point.
	for i in range(1,10):
		N = i+1
		m.append(2 * m[i-1] - combination(N, q))
		if m[-1] < pow(2,N):
			breakPoint = N
	print "m(N):"
	print m
	print "Break point at N = {b:g}, dvc = {d:g}".format(b=breakPoint, d=breakPoint-1)
	print ""
