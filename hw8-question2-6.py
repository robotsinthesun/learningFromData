#!/usr/bin/python
#coding:utf8

import numpy
from matplotlib import pyplot as plt
from sklearn.svm import SVC

# For using the support vector classifier from scikitlearn, refer to this documentation:
# http://scikit-learn.org/dev/modules/svm.html#svm-kernels

# Import the data. Data format is digit, intensity, symmetry.
dataTrain = numpy.fromfile('./digitFeaturesTrain', dtype=float, sep=' ').reshape(-1,3)
dataTest = numpy.fromfile('./digitFeaturesTest', dtype=float, sep=' ').reshape(-1,3)

print "Questions 2 to 4."

EinAll = []
nSupportsAll = []
for digit in range(10):
	print "   Testing digit {n:g}".format(n=digit)

	# Get features.
	XTrain = dataTrain[:,1:]
	# Generate binary label for the current digit.
	YTrain = (dataTrain[:,0] == digit) * 2 - 1
	print "      Occurrences: {n:g}.".format(n=numpy.sum(YTrain+1)/2.)

	# Run the SVM with polynomial kernel.
	Q = 2
	C = 0.01
	svc = SVC(C=C, kernel='poly', degree=Q, coef0=1, gamma=1)
	svc.fit(XTrain, YTrain)
	nSupportsAll.append(svc.support_vectors_.shape[0])
	print "      Number of support vectors:  {n:g}.".format(n=nSupportsAll[-1])

	# Calculate Ein.
	YTrainSvm = svc.predict(XTrain)
	Ein = 1 - numpy.sum(((YTrainSvm * YTrain) + 1) / 2.) / XTrain.shape[0]
	EinAll.append(Ein)
	print "      Ein: {n:2.2f}.".format(n=Ein)


	'''
	if digit == 1:
		counterOccurance = 0
		counterClassified = 0
		for x, y, ysvm in zip(XTrain, YTrain, YTrainSvm):
			if y == ysvm and y == 1:
				counterOccurance += 1
				counterClassified +=1
				plt.plot(x[0], x[1], 'r.')
			elif y != ysvm and y == 1:
				counterOccurance += 1
				plt.plot(x[0], x[1], 'ro', markerfacecolor='None')
			elif y == ysvm and y == -1:
				plt.plot(x[0], x[1], 'b.')
			elif y != ysvm and y == -1:
				plt.plot(x[0], x[1], 'bo', markerfacecolor='None')
		print "   Found {n:g} of {n2:g} occurances.".format(n=counterClassified, n2=counterOccurance)
		plt.show()
	'''


# Question 2:
choices = [0,2,4,6,8]
choice2 = choices[numpy.argsort(numpy.array([EinAll[i] for i in choices]))[-1]]
print ""
print "   Question 2: highest Ein for digit {n:g}.".format(n=choice2)

# Question 3:
choices = [1,3,5,7,9]
choice3 = choices[numpy.argsort(numpy.array([EinAll[i] for i in choices]))[0]]
print ""
print "   Question 3: lowest Ein for digit {n:g}.".format(n=choice3)


# Question 3:
nSupportsDifference = abs(nSupportsAll[choice2] - nSupportsAll[choice3])
print ""
print "   Question 4: difference in number of support vectors: {n:g}.".format(n=nSupportsDifference)



# Ouestion 5:
print ""
print "Question 5 and 6."
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
nSupportsAll = []
Cs = [0.0001, 0.001, 0.01, 0.1, 1.]
for C in Cs:
	print "   Testing C {n:g}".format(n=C)

	# Run the SVM with polynomial kernel.
	Qs = [2, 5]
	EinQAll = []
	EoutQAll = []
	nSupportsQAll = []
	for Q in Qs:
		print "      Q: " + str(Q)
		svc = SVC(C=C, kernel='poly', degree=Q, coef0=1, gamma=1)
		svc.fit(XTrain1vs5, YTrain1vs5)
		nSupportsQAll.append(svc.support_vectors_.shape[0])
		print "         Number of support vectors:  {n:g}.".format(n=nSupportsQAll[-1])

		# Calculate Ein.
		YTrain1vs5Svm = svc.predict(XTrain1vs5)
		Ein = 1 - numpy.sum(((YTrain1vs5Svm * YTrain1vs5) + 1) / 2.) / XTrain1vs5.shape[0]
		EinQAll.append(Ein)
		print "         Ein: {n:2.10f}.".format(n=Ein)

		# Calculate Eout.
		YTest1vs5Svm = svc.predict(XTest1vs5)
		Eout = 1 - numpy.sum(((YTest1vs5Svm * YTest1vs5) + 1) / 2.) / XTest1vs5.shape[0]
		EoutQAll.append(Eout)
		print "         Eout: {n:2.10f}.".format(n=Eout)

		'''
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
		'''
	nSupportsAll.append(nSupportsQAll)
	EinAll.append(EinQAll)
	EoutAll.append(EoutQAll)

EinAll = numpy.array(EinAll).T
EoutAll = numpy.array(EoutAll).T
nSupportsAll = numpy.array(nSupportsAll).T

# Evaluation for question 5. Leave out all results for C = 0.0001.
print ""
print "   Question 5: "
print "      Number of support vectors decreases with C: " + str(numpy.all(numpy.diff(nSupportsAll[0,1:]) < 0))
print "      Number of support vectors increases with C: " + str(numpy.all(numpy.diff(nSupportsAll[0,1:]) > 0))
print "      Eout decreases with C: " + str(numpy.all(numpy.diff(EoutAll[0,1:]) < 0))
print "      Lowest Ein at maximum C: " + str(Cs[1+numpy.argmin(EinAll[0,1:])] == 1)

# Evaulation for question 6.
print ""
print "   Question 6: "
print "      Ein higher for Q = " + str(Qs[1]) + " with C = " + str(Cs[0]) + ": " + str(EinAll[1,0] > EinAll[0,0])
print "      Number of support vectors lower for Q = " + str(Qs[1]) + " with C = " + str(Cs[1]) + ": " + str(nSupportsAll[1,1] < nSupportsAll[0,1])
print "      Ein higher for Q = " + str(Qs[1]) + " with C = " + str(Cs[2]) + ": " + str(EinAll[1,2] > EinAll[0,2])
print "      Eout lower for Q = " + str(Qs[1]) + " with C = " + str(Cs[4]) + ": " + str(EoutAll[1,4] < EoutAll[0,4])


fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
Qs = [2,5]
ls = ['-', '--']
for i in range(2):
	ax1.semilogx(Cs, EinAll[i], 'r'+ls[i], label="Q = " + str(Qs[i]) + ", $E_{in}$")
	ax1.semilogx(Cs, EoutAll[i], 'b'+ls[i], label="Q = " + str(Qs[i]) + ", $E_{out}$")
	ax1.set_xlabel('C')
	ax1.set_ylabel('$E_{in}$, $E_{out}$')
	ax2.semilogx(Cs, nSupportsAll[i], 'g'+ls[i], label='$Q = {n:g}, \#$ supports'.format(n=Qs[i]))
	ax2.set_ylabel('Number of supports')
	ax2.tick_params('y', colors='g')
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1+h2, l1+l2, loc=1)
fig.tight_layout()
plt.show()
