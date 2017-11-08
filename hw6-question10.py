#!/usr/bin/python
#coding: utf

from matplotlib import pyplot as plt

def calcNumberOfWeights(L):
	nInputs = 10
	nHiddenUnits = 36
	nOutputs = 1
	return nInputs*((nHiddenUnits/(L-1))-1) + (nHiddenUnits/(L-1)) * ((nHiddenUnits/(L-1))-1) * (L-2) + nHiddenUnits/(L-1) * nOutputs



#Ls = [2,3,4,5,7, 10, 13]
Ls = [2,3,5]
nConnections = []
for L in Ls:
	nConnections.append(calcNumberOfWeights(L))


plt.plot([L-1 for L in Ls], nConnections, 'r.')
plt.gca().set_xlabel("Number of layers")
plt.gca().set_ylabel("Number of weights")
for xy in zip([L-1 for L in Ls], nConnections):
	plt.gca().annotate(xy[1], xy=xy, textcoords='data')
plt.grid()
plt.show()
