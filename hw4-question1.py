#!/usr/bin/python

from math import *

# Define generalization error.
epsilonGoal = 0.05

# Define probability measure.
delta = 1 - 0.95
# VC dimension
dvc = 10.

# Brute force approach so we don't have to do the math.
for N in [	400000.,
			420000.,
			440000.,
			460000.,
			480000.]:

	# This is the VC bound approach.
	# See lecture 7 slide 23.
	print  epsilonGoal - sqrt(   (8/N)    *    log((4 *pow(2*N, dvc))/ delta)   )
