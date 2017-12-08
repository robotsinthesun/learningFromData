#!/usr/bin/python
#coding:utf8

import sys
import numpy
from matplotlib import pyplot as plt
from sklearn.svm import SVC

# For using the support vector classifier from scikitlearn with kernels, refer to this documentation:
# http://scikit-learn.org/dev/modules/svm.html#svm-kernels


# Import the data. Data format is digit, intensity, symmetry.
dataTrain = numpy.fromfile('./digitFeaturesTrain', dtype=float, sep=' ').reshape(-1,3)
dataTest = numpy.fromfile('./digitFeaturesTest', dtype=float, sep=' ').reshape(-1,3)

# Extract training data for 1 vs 5.
picks = numpy.any([dataTrain[:,0]==5, dataTrain[:,0]==1], axis=0)
XTrain1vs5 = dataTrain[picks,1:]
YTrain1vs5 = (dataTrain[picks,0] == 1) * 2 - 1

# Extract test data for 1 vs 5.
picks = numpy.any([dataTest[:,0]==5, dataTest[:,0]==1], axis=0)
XTest1vs5 = dataTest[picks,1:]
YTest1vs5 = (dataTest[picks,0] == 1) * 2 - 1

EinAll = []
EoutAll = []
Cs = [0.01, 1.0, 100, 10000, 1000000]
for C in Cs:
	print "   Testing C {n:g}".format(n=C)

	# Run the SVM with polynomial kernel.
	svc = SVC(C=C, kernel='rbf', gamma=1)
	svc.fit(XTrain1vs5, YTrain1vs5)

	# Calculate Ein.
	YTrain1vs5Svm = svc.predict(XTrain1vs5)
	Ein = 1 - numpy.sum(((YTrain1vs5Svm * YTrain1vs5) + 1) / 2.) / XTrain1vs5.shape[0]
	EinAll.append(Ein)
	print "         Ein: {n:2.10f}.".format(n=Ein)

	# Calculate Eout.
	YTest1vs5Svm = svc.predict(XTest1vs5)
	Eout = 1 - numpy.sum(((YTest1vs5Svm * YTest1vs5) + 1) / 2.) / XTest1vs5.shape[0]
	EoutAll.append(Eout)
	print "         Eout: {n:2.10f}.".format(n=Eout)


	for x, y, ysvm in zip(XTrain1vs5, YTrain1vs5, YTrain1vs5Svm):
		if y == ysvm and y == 1:
			plt.plot(x[0], x[1], 'rx')
		elif y != ysvm and y == 1:
			plt.plot(x[0], x[1], 'r+', markerfacecolor='None')
		elif y == ysvm and y == -1:
			plt.plot(x[0], x[1], 'bx')
		elif y != ysvm and y == -1:
			plt.plot(x[0], x[1], 'b+', markerfacecolor='None')

	for x, y, ysvm in zip(XTest1vs5, YTest1vs5, YTest1vs5Svm):
		if y == ysvm and y == 1:
			plt.plot(x[0], x[1], 'r.')
		elif y != ysvm and y == 1:
			plt.plot(x[0], x[1], 'ro', markerfacecolor='None')
		elif y == ysvm and y == -1:
			plt.plot(x[0], x[1], 'b.')
		elif y != ysvm and y == -1:
			plt.plot(x[0], x[1], 'bo', markerfacecolor='None')
	plt.show()


EinAll = numpy.array(EinAll).T
EoutAll = numpy.array(EoutAll).T

print "Lowest Ein found for C = {n:2.2f}.".format(n = Cs[numpy.argsort(numpy.array(EinAll))[0]])
print "Lowest Eout found for C = {n:2.2f}.".format(n = Cs[numpy.argsort(numpy.array(EoutAll))[0]])

plt.semilogx(Cs, EinAll, 'r-', label="$E_{in}$")
plt.semilogx(Cs, EoutAll, 'b-', label="$E_{out}$")
plt.gca().set_xlabel('C')
plt.gca().set_ylabel('$E_{in}$, $E_{out}$')
plt.legend()
plt.show()
