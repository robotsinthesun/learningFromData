#!/usr/bin/python
#coding:utf8

import sys
import numpy
from matplotlib import pyplot as plt
from sklearn.svm import SVC

# For using the support vector classifier from scikitlearn with kernels, refer to this documentation:
# http://scikit-learn.org/dev/modules/svm.html#svm-kernels

nIntervals = 10
nRuns = 100

# Import the data. Data format is digit, intensity, symmetry.
dataTrain = numpy.fromfile('./digitFeaturesTrain', dtype=float, sep=' ').reshape(-1,3)
dataTest = numpy.fromfile('./digitFeaturesTest', dtype=float, sep=' ').reshape(-1,3)

CSelectionCounter = [0, 0, 0, 0, 0]
EValForCsAll = []
for run in range(nRuns):

	# Shuffle the data indices.
	dataIndices = numpy.random.permutation(len(dataTrain))

	# Partition the indices into ten intervals.
	# In contrast to split(), array_split() does not complain about non-equal division of the input array.
	dataIndicesSegmented = numpy.array_split(dataIndices, nIntervals)

	# Train while using each of the intervals once for validation.
	EValForIntervals = []
	for i in range(nIntervals):

		# Split into training and validation data.
		dataVal = dataTrain[dataIndicesSegmented[i]]
		dataTrainRest = numpy.delete(dataTrain, dataIndicesSegmented[i], axis=0)

		#print "   Validation on interval " + str(i) + " of size " + str(dataVal.shape[0]) + "."

		# Extract training data for 1 vs 5.
		picks = numpy.any([dataTrainRest[:,0]==5, dataTrainRest[:,0]==1], axis=0)
		# Features.
		XTrain1vs5 = dataTrainRest[picks,1:]
		# Labels.
		YTrain1vs5 = (dataTrainRest[picks,0] == 1) * 2 - 1

		# Extract validation data for 1 vs 5.
		picks = numpy.any([dataVal[:,0]==5, dataVal[:,0]==1], axis=0)
		# Features.
		XVal1vs5 = dataVal[picks,1:]
		# Labels.
		YVal1vs5 = (dataVal[picks,0] == 1) * 2 - 1

		# Loop through Cs.
		Cs = [0.0001, 0.001, 0.01, 0.1, 1.]
		EValForCs = []
		for C, j in zip(Cs, range(len(Cs))):
			# Train.
			Q = 2
			svc = SVC(C=C, kernel='poly', degree=Q, coef0=1, gamma=1)
			svc.fit(XTrain1vs5, YTrain1vs5)

			# Validate.
			YVal1vs5Svm = svc.predict(XVal1vs5)
			EVal = 1 - numpy.sum(((YVal1vs5Svm * YVal1vs5) + 1) / 2.) / XVal1vs5.shape[0]
			EValForCs.append(EVal)
			#print "      Training with C = {n:1.4f}, validation error {e:2.4f}.".format(n=C, e=EVal)

		# Collect validation errors of all Cs for each interval.
		EValForIntervals.append(EValForCs)

	# Calc mean validation errors for each C over all intervals.
	EValForCsAll.append(numpy.mean(numpy.array(EValForIntervals), axis=0))
	# Sort by EVal, then by C to select the smaller C in case of EVal tie.
	CSelectionCounter[numpy.lexsort((numpy.array(Cs), numpy.mean(numpy.array(EValForIntervals), axis=0)))[0]] += 1
	sys.stdout.write("Run: {num:3g}, model selection counters for 0.0001, 0.0010, 0.0100, 0.1000, 1.0000: {s0:2g}, {s1:2g}, {s2:2g}, {s3:2g}, {s4:2g}\r".format(num=run, s0=CSelectionCounter[0], s1=CSelectionCounter[1], s2=CSelectionCounter[2], s3=CSelectionCounter[3], s4=CSelectionCounter[4]))
	sys.stdout.flush()

finalSelection = numpy.argsort(numpy.array(CSelectionCounter))[-1]
print ""
print "Final selection: C = {n:1.4f} wins with {s:g} points.".format(n=Cs[finalSelection], s=CSelectionCounter[finalSelection])
print "Validation error for final selection: {n:2.6f}.".format(n=numpy.mean(numpy.array(EValForCsAll), axis=0)[finalSelection])
