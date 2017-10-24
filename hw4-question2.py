#!/usr/bin/python

from math import *
import numpy
from matplotlib import pyplot as plt

# VC dimension
dvc = 50
delta = 0.05


def boundA(N):
	return sqrt((8./N) * log((4. * pow(2.*N,dvc))/delta))

def boundB(N):
	return sqrt( (2. * log(2. * N * pow(N,dvc))) / N ) + sqrt( (2./N) * log(1./delta) ) + 1./N

def boundC(N, iterations):
	epsilon = 1.

	for i in range(iterations):
		diff = epsilon - sqrt( 1./N * (2.*epsilon + log( (6. * pow(2.*N,dvc)) / delta)))
		epsilon -= 0.01 * diff
		if abs(diff) < 0.00001:
			print "   Used {n:g} iterations.".format(n=i)
			print "   Residual: {n:2.6f}.".format(n=diff)
			break
	return epsilon

def boundD(N, iterations):
	epsilon = 1.
	diffs = []
	for i in range(iterations):
		# Convert ln(4 * (N^2)^50 / delta) into ln(4) + 100*ln(N) - ln(delta) using product-, exponent- and quotient role.
		diff = epsilon - sqrt( 1./(2.*N) * (4.*epsilon * (1.+epsilon) + ( log(4.) + 100.*log(N) - log(delta)  ) ) )
		epsilon -= 0.01 * diff
		if abs(diff) < 0.00001:
			print "   Used {n:g} iterations.".format(n=i)
			print "   Residual: {n:2.6f}.".format(n=diff)
			break
	return epsilon


N = numpy.array(range(1, 10000))

print "Bounds for N = 10000:"
print "dvc: {n:2.4f}.".format(n=boundA(10000))
print "Rademacher penalty: {n:2.4f}.".format(n=boundB(10000))
print "Parrando and Van den Broek: {n:2.4f}.".format(n=boundC(10000, iterations=10000))
print "Devroye: {n:2.4f}.".format(n=boundD(10000, iterations=10000))
print ""
print "Bounds for N = 5:"
print "dvc: {n:2.4f}.".format(n=boundA(5))
print "Rademacher penalty: {n:2.4f}.".format(n=boundB(5))
print "Parrando and Van den Broek: {n:2.4f}.".format(n=boundC(5, iterations=10000))
print "Devroye: {n:2.4f}.".format(n=boundD(5, iterations=10000))

'''
plt.plot(N, [boundA(n) for n in N], label='$d_{vc}$')
plt.plot(N, [boundB(n) for n in N], label='Rademacher penalty')
plt.plot(N, [boundC(n, iterations=1000) for n in N], label='Parrando and Van den Broek')
plt.plot(N, [boundD(n, iterations=1000) for n in N], label='Devroye')
plt.grid(True)
plt.legend()
plt.show()
'''

