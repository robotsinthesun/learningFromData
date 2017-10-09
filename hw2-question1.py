#!/usr/bin/python
# coding: utf8

'''
Run a computer simulation for flipping 1,000 virtual fair coins.
- Flip each coin independently 10 times.
- Focus on 3 coins as follows:
	- c 1 is the first coin flipped
	- c rand is a coin chosen randomly from the 1,000
	- c min is the coin which had the minimum frequency of heads (pick the earlier one in case of a tie)
- Let ν 1 , ν rand , and ν min be the fraction of heads obtained for the 3 respective coins out of the 10 tosses.

- Run the experiment 100,000 times in order to get a full distribution of ν 1 , ν rand , and ν min
	(note that c rand and c min will change from run to run).
'''
import sys
import numpy

# Experiment parameters.
nTosses = 10
nCoins = 1000
nExperiments = 100000

# Output variables.
nuAll = numpy.empty((nExperiments,3))

# Run the experiment.
for i in range(nExperiments):

	# Get the output of tossing 1000 coins ten times each.
	# Head = 1, tail = 0.
	coinsAll = numpy.random.rand(nCoins, nTosses).round()

	# Grab the coins the question asks for.
	coinsChosen = numpy.vstack([	coinsAll[0],									# First coin.
									coinsAll[int(numpy.random.randint(0,nCoins))],	# Random coin.
									coinsAll[coinsAll.sum(axis=1).argmin()]				# Min heads coin. Sum tosses outcome and get index of coin with the min sum.
									])

	# Get the fraction of heads for each of the chosen coins.
	nuAll[i] = coinsChosen.sum(axis=1) / 10.

	# Print progress.
	if numpy.mod(i, nExperiments/100.) < 1e-5:
		sys.stdout.write("Progress: {num:3g}%\r".format(num=i/float(nExperiments)*100))
		sys.stdout.flush()

# Average the nus for all experiments.
nuAverage = numpy.average(nuAll, axis=0)

'''
Question 1
The average value of ν min is closest to:
	[a] 0
	[b] 0.01
	[c] 0.1
	[d] 0.5
	[e] 0.67
'''
print "Average nu for first coin:     {num:2.4f}.".format(num=nuAverage[0])
print "Average nu for random coin:    {num:2.4f}.".format(num=nuAverage[1])
print "Average nu for min heads coin: {num:2.4f}.".format(num=nuAverage[2])
print "Answer to question 1: {a:s}.".format(a=['a', 'b', 'c', 'd', 'f'][numpy.abs(numpy.array([0, 0.01, 0.1, 0.5, 0.67]) - nuAverage[2]).argmin()])

