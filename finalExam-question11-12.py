#!/usr/bin/python
#coding:utf8

import sys
import numpy
from matplotlib import pyplot as plt
from sklearn.svm import SVC


X = numpy.array([	[1,		0	],
					[0,		1	],
					[0,		-1	],
					[-1,	0	],
					[0,		2	],
					[0,		-2	],
					[-2,	0	]	])

Y = numpy.array([	-1,
					-1,
					-1,
					+1,
					+1,
					+1,
					+1	])


# Question 11. *****************************************************************
Z = numpy.vstack([numpy.power(X[:,1], 2)-2*X[:,0]-1   ,   numpy.power(X[:,0], 2)-2*X[:,1]+1]).T

fig, (ax1, ax0) = plt.subplots(1, 2)

colors = ['b', 'r']
# Plot X points.
ax0.scatter(X[:,0], X[:,1], s=7, c=[colors[int(i+1/2.)] for i in Y])
ax0.set_xlim(-2.2, 2.2)
ax0.set_ylim(-2.2, 2.2)
ax0.set_xlabel('$x_1$')
ax0.set_ylabel('$x_2$')
ax0.set_aspect(1)
ax0.set_title('$\mathcal{X}$')
# Plot Z points.
# Mark the classifier line and the margins.
ax1.plot([0.5, 0.5], [-6, 6], 'k-', lw=0.5)
ax1.plot([0, 0], [-6, 6], 'gray', lw=0.1)
ax1.plot([1, 1], [-6, 6], 'gray', lw=0.1)
# Plot points.
ax1.scatter(Z[:,0], Z[:,1], s=7, c=[colors[int(i+1/2.)] for i in Y])
ax1.set_xlim(-5.2, 5.2)
ax1.set_ylim(-5.2, 5.2)
ax1.set_xlabel('$z_1$')
ax1.set_ylabel('$z_2$')
ax1.set_aspect(1)
ax1.set_title('$\mathcal{Z}$')
# Mark the support vectors that were derived by looking at the plot.
ax1.scatter(Z[1:4,0], Z[1:4,1], marker='o', facecolor='None', s=50, edgecolor=[colors[int(i+1/2.)] for i in Y][1:4])
ax1.set_xticks([-5., -2.5, 0, 2.5, 5.0])
ax1.set_yticks([-5., -2.5, 0, 2.5, 5.0])
plt.tight_layout()



# Question 12. *****************************************************************
# Fit a hard margin SVC.
# The polynomial kernel is defined by (gamma(xTx') + r)^2 where gamma = r = 1. In sklearn, r = coef0
svc = SVC(C=10e10, kernel='poly', degree=2, coef0=1, gamma=1)
svc.fit(X, Y)
print "Question 12:"
print "   Number of support vectors: {n:g}.".format(n=svc.support_vectors_.shape[0])

# Plot the classifier and margins according to this resource: https://jakevdp.github.io/PythonDataScienceHandbook/05.07-support-vector-machines.html
xGrid = numpy.linspace(ax0.get_xlim()[0], ax0.get_xlim()[1], 30)
yGrid = numpy.linspace(ax0.get_ylim()[0], ax0.get_ylim()[1], 30)
YGrid, XGrid = numpy.meshgrid(yGrid, xGrid)
xyGrid = numpy.vstack([XGrid.ravel(), YGrid.ravel()]).T
P = svc.decision_function(xyGrid).reshape(XGrid.shape)
# Plot the support vectors.
ax0.scatter(svc.support_vectors_[:, 0], svc.support_vectors_[:, 1], marker='o', facecolor='None', s=50, edgecolor=[colors[int(i+1/2.)] for i in Y[svc.support_]]);

# plot decision boundary and margins
ax0.contour(XGrid, YGrid, P, colors=['gray', 'k', 'gray'], levels=[-1, 0, 1], linewidths=[0.2, 0.5, 0.2])

plt.show()
