#!/usr/bin/python
#coding:utf8

import sys
import numpy
from matplotlib import pyplot as plt
from sklearn.svm import SVC
import learningAlgorithms

plot = False

nPoints = 100
nPointsTest = 100
nRuns = 1000
colors = ['b', 'r']


def targetFunction(X):
	return numpy.sign(X[:,1] - X[:,0] + 0.25*numpy.sin(numpy.pi*X[:,0]))

svmNonSepCounter = 0
failCounter = 0

EinSvmAll = []
EoutSvmAll = []
EinRbfAll = []
EoutRbfAll = []



run = 0
while run < nRuns:

	print "Run {n:g} of {n2:g}.".format(n=run+1, n2=nRuns)
	svmNonSep = False


	# Generate data set.
	X = numpy.random.random((nPoints, 2)) * 2 - 1
	Y = targetFunction(X)
	XTest = numpy.random.random((nPoints, 2)) * 2 - 1
	YTest = targetFunction(XTest)



	# Create and train a support vector classifier with RBF kernel and gamma = 1.5.
	# For using the support vector classifier from scikitlearn, refer to this documentation:
	# http://scikit-learn.org/dev/modules/svm.html#svm-kernels
	# Set parameters.
	gamma = 1.5
	svc = SVC(C=10e10, kernel='rbf', gamma=gamma)
	svc.fit(X, Y)

	# Classify the points.
	YSvm = svc.predict(X)
	YTestSvm = svc.predict(XTest)

	# Calc in-sample error for RBF kernel.
	Ein = 1 - numpy.sum(((YSvm * Y) + 1) / 2.) / X.shape[0]
	print "   SVM Ein: {n:2.2f}.".format(n=Ein)

	# If svm wasn't able to separate, break the current iteration.
	if Ein != 0:
		svmNonSepCounter += 1
		svmNonSep = True
		failCounter += 1
		print "   Run aborted!"
		continue

	# Get out-of-sample errors.
	EoutSvm = 1 - numpy.sum(((YTestSvm * YTest) + 1) / 2.) / XTest.shape[0]
	print "   Eout SVM: {n:2.4f}.".format(n=EoutSvm)




	# Create and train the RBF K-means thingy.
	# Do this for K = 9 and 12 and gamma = 1.5 and 2.
	EinKs = []
	EoutKs = []
	failFlag = False
	for K in [9, 12]:
		EinGammas = []
		EoutGammas = []
		for gamma in [1.5, 2.0]:

			rbf = learningAlgorithms.rbf()
			success = rbf.learn(X, Y, K, gamma)
			# If KMeans did not succeed because one or more clusters where empty, break the current iteration.
			if not success:
				failFlag = True
				break

			# Classify the points.
			YRbf = rbf.classify(X, gamma)
			YTestRbf = rbf.classify(XTest, gamma)


			# Calc in-sample error for KMeans.
			EinRbf = 1 - numpy.sum(((YRbf * Y) + 1) / 2.) / X.shape[0]
			EinGammas.append(EinRbf)
			print "   Ein KMeans: {n:2.2f}.".format(n=Ein)


			# Calculate Eout.
			EoutRbf = 1 - numpy.sum(((YTestRbf* YTest) + 1) / 2.) / XTest.shape[0]
			EoutGammas.append(EoutRbf)
			print "   Eout KMeans: {n:2.4f}.".format(n=EoutRbf)


			if plot:
				# Set up plots for svm and rbf.
				fig, (ax0, ax1) = plt.subplots(1, 2)
				ax0.set_aspect(1)
				ax0.set_title('SVM with RBF kernel, $\gamma = 1.5$')
				ax1.set_aspect(1)
				ax1.set_title('RBF K-means, $\gamma = {g:1.1f}$, $K = {k:g}$'.format(g=gamma, k=K))
				ax0.set_xticks([-1, -0.5, 0, .5, 1])
				ax0.set_yticks([-1, -0.5, 0, .5, 1])
				ax1.set_xticks([-1, -0.5, 0, .5, 1])
				ax1.set_yticks([-1, -0.5, 0, .5, 1])
				plt.tight_layout()

				# Plot the target function and classifiers.
				# Set up point grid.
				xGrid = numpy.linspace(-1.1, 1.1, 1000)
				yGrid = numpy.linspace(-1.1, 1.1, 1000)
				YGrid, XGrid = numpy.meshgrid(yGrid, xGrid)
				xyGrid = numpy.vstack([XGrid.ravel(), YGrid.ravel()]).T
				# Evaluate target function and plot.
				P = targetFunction(xyGrid).reshape(XGrid.shape)
				ax0.contour(XGrid, YGrid, P, colors=[[0.97, 0.97, 0.97]], levels=[0], linewidths=[2])
				ax1.contour(XGrid, YGrid, P, colors=[[0.97, 0.97, 0.97]], levels=[0], linewidths=[2])
				# Evaluate SVM classifier function and plot.
				P = svc.decision_function(xyGrid).reshape(XGrid.shape)
				ax0.contour(XGrid, YGrid, P, colors=['b', 'g', 'r'], levels=[-1, 0, 1], linewidths=[0.2, 0.5, 0.2])
				# Evaluate RBF classifier function and plot.
				P = rbf.classify(xyGrid, gamma).reshape(XGrid.shape)
				ax1.contour(XGrid, YGrid, P, colors=['g'], levels=[0], linewidths=[0.5])



				# Plot SVM classified points and support vectors.
				ax0.scatter(X[:,0], X[:,1], s=7, c=[colors[int(i+1/2.)] for i in YSvm], zorder=2)
				# Mark support vectors.
				ax0.scatter(svc.support_vectors_[:,0], svc.support_vectors_[:,1], marker='o', facecolor='None', s=50, edgecolor=[colors[int(i+1/2.)] for i in YSvm[svc.support_]], zorder=2)


				# Plot RBF classified points and cluster centers.
				# Centers.
				for k in range(K):
					# Plot cluster centers.
					#ax1.scatter(rbf.KCenters[k,0], rbf.KCenters[k,1], marker='${n:g}$'.format(n=k), color='g')
					ax1.scatter(rbf.KCenters[k,0], rbf.KCenters[k,1], color='g', zorder=2)
					# Plot cluster lines.
					for i in rbf.indicesKToPoints[k]:
						ax1.plot([rbf.KCenters[k,0], X[i,0]], [rbf.KCenters[k,1], X[i,1]], 'gray', lw=0.2)
				# Classified points.
				ax1.scatter(X[:,0], X[:,1], s=7, c=[colors[int(i+1/2.)] for i in YRbf], zorder=2)
				# Plot points as point indices.
				#for x in range(X.shape[0]):
				#	plt.scatter(X[x,0], X[x,1], marker='${n:g}$'.format(n=x), c=[colors[int(i+1/2.)] for i in YSvm])


				plt.show()

		if failFlag:
			break
		EinKs.append(EinGammas)
		EoutKs.append(EoutGammas)




	if failFlag:
		failCounter += 1
		print "   Run aborted!"
		continue


	EinSvmAll.append(Ein)
	EoutSvmAll.append(EoutSvm)
	EinRbfAll.append(EinKs)
	EoutRbfAll.append(EoutKs)

	run += 1


EinSvmAll = numpy.array(EinSvmAll)
EoutSvmAll = numpy.array(EoutSvmAll)
EinRbfAll = numpy.array(EinRbfAll)
EoutRbfAll = numpy.array(EoutRbfAll)

print ""
print "{n:g} valid runs, {t:g} total runs.".format(n=nRuns, t=nRuns + failCounter)


print ""
print "Question 13:"
print "   Number of non-seprable data sets in {t:g} total runs: {n:g} ({n2:2.2f}%).".format(t=nRuns+failCounter, n=svmNonSepCounter, n2=svmNonSepCounter/float(nRuns+failCounter)*100)



print ""
print "Question 14:"
# Get KMeans Eouts for gamma = 1.5 and K = 9.
# 1st axis: run, 2nd axis: K in [9, 12], 3rd axis: gamma in [1.5, 2].
EoutKMeans = EoutRbfAll[:,0,0]
nSvmWinsK9 = EoutKMeans[EoutSvmAll < EoutKMeans].shape[0]
print "   Svm wins over KMeans in {n:3.2f} of all valid runs.".format(n=nSvmWinsK9 / float(nRuns) * 100)



print ""
print "Question 15:"
# Get KMeans Eouts for gamma = 1.5 and K = 9.
# 1st axis: run, 2nd axis: K in [9, 12], 3rd axis: gamma in [1.5, 2].
EoutKMeans = EoutRbfAll[:,1,0]
nSvmWinsK12 = EoutKMeans[EoutSvmAll < EoutKMeans].shape[0]
print "   Svm wins over KMeans in {n:3.2f} of all valid runs.".format(n=nSvmWinsK12 / float(nRuns) * 100)



print ""
print "Question 16:"
# Count the events for each choice.
EinK9 = EinRbfAll[:,0,0]
EinK12 = EinRbfAll[:,1,0]
EoutK9 = EoutRbfAll[:,0,0]
EoutK12 = EoutRbfAll[:,1,0]
nAWins = EinK9[numpy.logical_and(EinK12 < EinK9 , EoutK12 > EoutK9)].shape[0]	# Ein goes down, Eout goes up.
nBWins = EinK9[numpy.logical_and(EinK12 > EinK9 , EoutK12 < EoutK9)].shape[0]	# Ein goes up, Eout goes down.
nCWins = EinK9[numpy.logical_and(EinK12 > EinK9 , EoutK12 > EoutK9)].shape[0]	# Ein goes up, Eout goes up.
nDWins = EinK9[numpy.logical_and(EinK12 < EinK9 , EoutK12 < EoutK9)].shape[0]	# Ein goes down, Eout goes down.
nEWins = EinK9[numpy.logical_and(EinK12 == EinK9 , EoutK12 == EoutK9)].shape[0]	# Ein and Eout stay the same.
# Get the winner.
winIndex = numpy.argsort(numpy.array([nAWins, nBWins, nCWins, nDWins, nEWins]))[-1]
print "   Wins per choice:"
print "      a: {n:3.2f}%.".format(n=[nAWins, nBWins, nCWins, nDWins, nEWins][0] / float(nRuns) * 100)
print "      b: {n:3.2f}%.".format(n=[nAWins, nBWins, nCWins, nDWins, nEWins][1] / float(nRuns) * 100)
print "      c: {n:3.2f}%.".format(n=[nAWins, nBWins, nCWins, nDWins, nEWins][2] / float(nRuns) * 100)
print "      d: {n:3.2f}%.".format(n=[nAWins, nBWins, nCWins, nDWins, nEWins][3] / float(nRuns) * 100)
print "      e: {n:3.2f}%.".format(n=[nAWins, nBWins, nCWins, nDWins, nEWins][4] / float(nRuns) * 100)
print "   When going from K = 9 to K = 12, choice {c:s} wins in {n:3.2f}% of the runs.".format(c=['a', 'b', 'c', 'd', 'e'][winIndex], n=[nAWins, nBWins, nCWins, nDWins, nEWins][winIndex] / float(nRuns) * 100)



print ""
print "Question 17:"
# Count the events for each choice.
EinG15 = EinRbfAll[:,0,0]
EinG20 = EinRbfAll[:,0,1]
EoutG15 = EoutRbfAll[:,0,0]
EoutG20 = EoutRbfAll[:,0,1]
nAWins = EinG15[numpy.logical_and(EinG20 < EinG15 , EoutG20 > EoutG15)].shape[0]	# Ein goes down, Eout goes up.
nBWins = EinG15[numpy.logical_and(EinG20 > EinG15 , EoutG20 < EoutG15)].shape[0]	# Ein goes up, Eout goes down.
nCWins = EinG15[numpy.logical_and(EinG20 > EinG15 , EoutG20 > EoutG15)].shape[0]	# Ein goes up, Eout goes up.
nDWins = EinG15[numpy.logical_and(EinG20 < EinG15 , EoutG20 < EoutG15)].shape[0]	# Ein goes down, Eout goes down.
nEWins = EinG15[numpy.logical_and(EinG20 == EinG15 , EoutG20 == EoutG15)].shape[0]	# Ein and Eout stay the same.
# Get the winner.
winIndex = numpy.argsort(numpy.array([nAWins, nBWins, nCWins, nDWins, nEWins]))[-1]
print "   Wins per choice:"
print "      a: {n:3.2f}%.".format(n=[nAWins, nBWins, nCWins, nDWins, nEWins][0] / float(nRuns) * 100)
print "      b: {n:3.2f}%.".format(n=[nAWins, nBWins, nCWins, nDWins, nEWins][1] / float(nRuns) * 100)
print "      c: {n:3.2f}%.".format(n=[nAWins, nBWins, nCWins, nDWins, nEWins][2] / float(nRuns) * 100)
print "      d: {n:3.2f}%.".format(n=[nAWins, nBWins, nCWins, nDWins, nEWins][3] / float(nRuns) * 100)
print "      e: {n:3.2f}%.".format(n=[nAWins, nBWins, nCWins, nDWins, nEWins][4] / float(nRuns) * 100)
print "  When going from gamma = 1.5 to gamma = 2.0, choice {c:s} wins in {n:3.2f}% of the runs.".format(c=['a', 'b', 'c', 'd', 'e'][winIndex], n=[nAWins, nBWins, nCWins, nDWins, nEWins][winIndex] / float(nRuns) * 100)


print ""
print "Question 18:"
EinK9G15 = EinRbfAll[:,0,0]
nEin0 = EinK9G15[EinK9G15==0].shape[0]
print "      Ein is 0 for K = 9 and gamma = 1.5 in {n:3.2f}% of the runs.".format(n=(nEin0 / float(nRuns)) * 100)
#EoutSvm = numpy.average(numpy.array(EoutSvmAll))
#EoutRbf = numpy.average(numpy.array(EoutRbfAll))
#print "   Eout Svm: {es:2.4f}, Eout Rbf: {er:2.4f}. "
