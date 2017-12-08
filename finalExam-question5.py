#!/usr/bin/python
#coding:utf8

import numpy as np
from matplotlib import pyplot as plt



C = 0.01

# Set up Tikhonov matrix.
T = np.array([[0.3,0],[0,0.15]])
evalReg = lambda x: x.dot(T).dot(T).dot(x)

# Set up point grid of wLin vectors.
nPoints = 250
xGrid = np.linspace(-1.1, 1.1, nPoints)
yGrid = np.linspace(-1.1, 1.1, nPoints)
YGrid, XGrid = np.meshgrid(yGrid, xGrid)
xyGrid = np.vstack([XGrid.ravel(), YGrid.ravel()]).T
# Evaluate target function and plot.
P = []
for p in range(nPoints**2):
	P.append(evalReg(xyGrid[p,:]))
P = np.array(P)
P = P.reshape(XGrid.shape)
print XGrid
print P
plt.contour(XGrid, YGrid, P, colors=[[0.5, 0.5, 0.5]], levels=[C], linewidths=[1])






'''
for run in range(100):
	#wLin = np.array([2,3])
	wLin = np.random.random(2)*10-5
	T = np.array([[0.3,0],[0,0.15]])

	C = 3.

	reg = wLin.dot(T).dot(T).dot(wLin)

	# Plot wlin.
	plt.plot(wLin[0],wLin[1], 'g.')

	# Plot wreg.
	if reg < C:
		plt.plot(wLin.dot(T)[0], wLin.dot(T)[1], 'b.')
	else:
		plt.plot(wLin.dot(T)[0], wLin.dot(T)[1], 'r.')
'''
# Plot circle for reg.
#plt.plot(np.sin(np.linspace(-np.pi,np.pi,100))*reg, np.cos(np.linspace(-np.pi,np.pi,100))*reg, 'purple')

# Plot circle for C.
#plt.plot(0, 0, 'k.')
#plt.plot(np.sin(np.linspace(-np.pi,np.pi,100))*C, np.cos(np.linspace(-np.pi,np.pi,100))*C)


plt.gca().set_aspect(1)
plt.show()
