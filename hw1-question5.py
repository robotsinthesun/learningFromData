#!/usr/bin/python

import random

nExperiments = 10000
nSamples = 1000
nMarbles = 10

allGreenCounter = 0.

for i in range(nExperiments):
	dataSet = [[random.random() <= 0.55 for m in range(nMarbles)] for s in range(nSamples)]

	for j in range(nSamples):
		if all(datum == False for datum in dataSet[j]):
			allGreenCounter += 1
			print "Found all green sample, experiment {num:g}".format(num=i)
			break

allGreenRate = allGreenCounter / nExperiments

print "Amount of experiments that had at least one all green sample: {num:2.4f}".format(num=allGreenRate)
