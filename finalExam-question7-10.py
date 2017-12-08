#!/usr/bin/python
#coding:utf8

import sys
import numpy
from matplotlib import pyplot as plt
from learningAlgorithms import *
from sklearn.svm import SVC

plot = True

# Import the data. Data format is digit, intensity, symmetry.
dataTrain = numpy.fromfile('./digitFeaturesTrain', dtype=float, sep=' ').reshape(-1,3)
dataTest = numpy.fromfile('./digitFeaturesTest', dtype=float, sep=' ').reshape(-1,3)

'''
# Plot all data points with the corresponding digits as markers.
colors =['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
for i in range(dataTrain.shape[0]):
	plt.scatter(dataTrain[i,1], dataTrain[i,2], marker='${n:g}$'.format(n=dataTrain[i,0]), color=colors[int(dataTrain[i,0])])
plt.show()
'''



# Do questions 7--9. ***********************************************************
# Train the the model with lambda = 1 using no non-linear transform.
lam = 1
digits = range(10)
EinLinAll = []
EinNonLinAll = []
EoutLinAll = []
EoutNonLinAll = []
for digit in digits:
	print "   Testing digit {n:g}".format(n=digit)

	# Get features without and with non-linear transform.
	XTrain = dataTrain[:,1:]
	XTest = dataTest[:,1:]
	ZTrain = numpy.vstack([XTrain[:,0], XTrain[:,1], XTrain[:,0]*XTrain[:,1], numpy.power(XTrain[:,0], 2), numpy.power(XTrain[:,1], 2)]).T
	ZTest = numpy.vstack([XTest[:,0], XTest[:,1], XTest[:,0]*XTest[:,1], numpy.power(XTest[:,0], 2), numpy.power(XTest[:,1], 2)]).T
	# Generate binary label for the current digit.
	YTrain = (dataTrain[:,0] == digit) * 2 - 1
	YTest = (dataTest[:,0] == digit) * 2 - 1

	# Train linear regression classifier.
	lrcLin = linearRegressionAlgorithm(XTrain, YTrain)
	WLin = lrcLin.learn(weightDecayLambda=lam)

	# Train linear regression classifier with non-linear transform.
	lrcNonLin = linearRegressionAlgorithm(ZTrain, YTrain)
	WNonLin = lrcNonLin.learn(weightDecayLambda=lam)

	# Classify the training set without non-linear transform and calculate in-sample error.
	YTrainLrcLin = lrcLin.classify(XTrain, WLin)
	EinLin = 1 - numpy.sum(((YTrainLrcLin * YTrain) + 1) / 2.) / XTrain.shape[0]
	EinLinAll.append(EinLin)
	print "      Ein linear: {n:2.4f}.".format(n=EinLin)

	# Classify the training set with non-linear transform and calculate in-sample error.
	YTrainLrcNonLin = lrcNonLin.classify(ZTrain, WNonLin)
	EinNonLin = 1 - numpy.sum(((YTrainLrcNonLin * YTrain) + 1) / 2.) / ZTrain.shape[0]
	EinNonLinAll.append(EinNonLin)
	print "      Ein non-linear: {n:2.4f}.".format(n=EinNonLin)

	# Classify the test set without non-linear transform and calculate out-of-sample error.
	YTestLrcLin = lrcLin.classify(XTest, WLin)
	EoutLin = 1 - numpy.sum(((YTestLrcLin * YTest) + 1) / 2.) / XTest.shape[0]
	EoutLinAll.append(EoutLin)
	print "      Eout linear: {n:2.4f}.".format(n=EoutLin)

	# Classify the test set with non-linear transform and calculate out-of-sample error.
	YTestLrcNonLin = lrcNonLin.classify(ZTest, WNonLin)
	EoutNonLin = 1 - numpy.sum(((YTestLrcNonLin * YTest) + 1) / 2.) / XTest.shape[0]
	EoutNonLinAll.append(EoutNonLin)
	print "      Eout non-linear: {n:2.4f}.".format(n=EoutNonLin)

	'''
	for x, y, ysvm in zip(XTrain, YTrain, YTrainLrc):
		if y == ysvm and y == 1:
			plt.plot(x[0], x[1], 'r.')
		elif y != ysvm and y == 1:
			plt.plot(x[0], x[1], 'ro', markerfacecolor='None')
		elif y == ysvm and y == -1:
			plt.plot(x[0], x[1], 'b.')
		elif y != ysvm and y == -1:
			plt.plot(x[0], x[1], 'bo', markerfacecolor='None')
	plt.show()
	'''


# Question 7. ******************************************************************
print ""
print "Question 7"
winner = numpy.argsort(numpy.array(EinLinAll)[5:10])[0]
print "   The minimum in-sample error for digits 5 to 9 occurs for digit {d:g} with Ein = {e:2.4f}. Solution: {s:s}".format(d=digits[5:10][winner], e=EinLinAll[5:10][winner], s=['a', 'b', 'c', 'd', 'd'][winner])

if plot:
	plt.bar(numpy.delete(digits[5:10], winner), numpy.delete(EinLinAll[5:10], winner), label='$E_{in}$')
	plt.bar(digits[5:10][winner], EinLinAll[5:10][winner], color='r')
	plt.xticks(digits[5:10])
	plt.title('Question 7')
	plt.xlabel('Digit')
	plt.ylabel('$E_{in}$')
	plt.show()


# Question 8. ******************************************************************
print ""
print "Question 8"
winner = numpy.argsort(numpy.array(EoutNonLinAll)[0:5])[0]
print "   The minimum out-of-sample error for digits 0 to 4 occurs for digit {d:g} with Eout = {e:2.4f}. Solution: {s:s}".format(d=digits[0:5][winner], e=EoutNonLinAll[0:5][winner], s=['a', 'b', 'c', 'd', 'd'][winner])

if plot:
	plt.bar(numpy.delete(digits[0:5], winner), numpy.delete(EoutNonLinAll[0:5], winner), label='$E_{out}$')
	plt.bar(digits[0:5][winner], EoutNonLinAll[0:5][winner], color='r')
	plt.xticks(digits[0:5])
	plt.title('Question 8')
	plt.xlabel('Digit')
	plt.ylabel('$E_{out}$')
	plt.show()


# Question 9. ******************************************************************
print ""
print "Question 9"
print "   Change of Eout when applying non-linear transform per digit:"
EinLinAll = numpy.array(EinLinAll)
EinNonLinAll = numpy.array(EinNonLinAll)
EoutLinAll = numpy.array(EoutLinAll)
EoutNonLinAll = numpy.array(EoutNonLinAll)
inSamplePerformanceChange = (EinLinAll - EinNonLinAll) / EinNonLinAll * 100
outOfSamplePerformanceChange = (EoutLinAll - EoutNonLinAll) / EoutNonLinAll * 100

for digit in digits:
	print "      Digit {d:g}: EoutLin = {el:2.4f}, EoutNonLin = {en:2.4f}, Eout change = {ec:2.2f}%.".format(d=digit, el=EoutLinAll[digit], en=EoutNonLinAll[digit], ec=(EoutLinAll[digit] - EoutNonLinAll[digit]) / EoutNonLinAll[digit] * 100)

if numpy.all(inSamplePerformanceChange > 0) and numpy.all(outOfSamplePerformanceChange < 0):
	print "   Solution a."
if numpy.all(outOfSamplePerformanceChange >= 5):
	print "   Solution b."
if numpy.all(outOfSamplePerformanceChange == 0):
	print "   Solution c."
if numpy.all(outOfSamplePerformanceChange <= -5):
	print "   Solution d."
if outOfSamplePerformanceChange[5] > 0 and outOfSamplePerformanceChange[5] < 5:
	print "   Solution e."






# Do question 10. **************************************************************
print ""
print "Question 10"
EinLinAll = []
EinNonLinAll = []
EoutLinAll = []
EoutNonLinAll = []

# Extract training data for 1 vs 5.
picks = numpy.any([dataTrain[:,0]==5, dataTrain[:,0]==1], axis=0)
XTrain = dataTrain[picks, 1:]
YTrain = (dataTrain[picks, 0] == 1) * 2 - 1

# Extract test data for 1 vs 5.
picks = numpy.any([dataTest[:,0]==5, dataTest[:,0]==1], axis=0)
XTest = dataTest[picks, 1:]
YTest = (dataTest[picks, 0] == 1) * 2 - 1

for lam in [0.01, 1.]:
	print "   Testing lambda {n:2.2f}".format(n=lam)

	# Apply non-linear transform.
	ZTrain = numpy.vstack([XTrain[:,0], XTrain[:,1], XTrain[:,0]*XTrain[:,1], numpy.power(XTrain[:,0], 2), numpy.power(XTrain[:,1], 2)]).T
	ZTest = numpy.vstack([XTest[:,0], XTest[:,1], XTest[:,0]*XTest[:,1], numpy.power(XTest[:,0], 2), numpy.power(XTest[:,1], 2)]).T


	# Train linear regression classifier with non-linear transform.
	lrcNonLin = linearRegressionAlgorithm(ZTrain, YTrain)
	WNonLin = lrcNonLin.learn(weightDecayLambda=lam)

	# Classify the training set with non-linear transform and calculate in-sample error.
	YTrainLrcNonLin = lrcNonLin.classify(ZTrain, WNonLin)
	EinNonLin = 1 - numpy.sum(((YTrainLrcNonLin * YTrain) + 1) / 2.) / ZTrain.shape[0]
	EinNonLinAll.append(EinNonLin)
	print "      Ein non-linear: {n:2.4f}.".format(n=EinNonLin)

	# Classify the test set with non-linear transform and calculate out-of-sample error.
	YTestLrcNonLin = lrcNonLin.classify(ZTest, WNonLin)
	EoutNonLin = 1 - numpy.sum(((YTestLrcNonLin * YTest) + 1) / 2.) / XTest.shape[0]
	EoutNonLinAll.append(EoutNonLin)
	print "      Eout non-linear: {n:2.4f}.".format(n=EoutNonLin)


	for x, y, ysvm in zip(XTest, YTest, YTestLrcNonLin):
		if y == ysvm and y == 1:
			plt.plot(x[0], x[1], 'r.')
		elif y != ysvm and y == 1:
			plt.plot(x[0], x[1], 'ro', markerfacecolor='None')
		elif y == ysvm and y == -1:
			plt.plot(x[0], x[1], 'b.')
		elif y != ysvm and y == -1:
			plt.plot(x[0], x[1], 'bo', markerfacecolor='None')
	plt.show()


print ""
if  EinNonLinAll[0] < EinNonLinAll[1] and EoutNonLinAll[0] > EoutNonLinAll[1]:
	print "   Solution a."

if EinNonLinAll[0] == EinNonLinAll[1]:
	print "   Solution b."

if EoutNonLinAll[0] == EoutNonLinAll[1]:
	print "   Solution c."

if EinNonLinAll[0] < EinNonLinAll[1] and EoutNonLinAll[0] < EoutNonLinAll[1]:
	print "   Solution d."

if EinNonLinAll[0] > EinNonLinAll[1] and EoutNonLinAll[0] > EoutNonLinAll[1]:
	print "   Solution d."
