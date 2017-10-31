#!/usr/bin/python
#coding: utf8

import numpy
from matplotlib import cm # colormaps
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt

# Create some data. This is for visualization only.
# Create uv grid.
uSurf, vSurf = numpy.meshgrid(numpy.arange(-.5, 1, 0.01), numpy.arange(-1, 1, 0.01))
# Evaluate error function on grid points.
eSurf = numpy.power(uSurf*numpy.exp(vSurf) - 2*vSurf*numpy.exp(-uSurf),2)


# Define E and gradE.
def E(uv):
	u = uv[0]
	v = uv[1]
	return pow(u*numpy.exp(v) - 2*v*numpy.exp(-u),2)

def gradE(uv):
	u = uv[0]
	v = uv[1]
	return numpy.array(	[	2 * (u*numpy.exp(v) - 2*v*numpy.exp(-u)) * (numpy.exp(v) + 2*v*numpy.exp(-u) ), 2 * (u*numpy.exp(v) - 2*v*numpy.exp(-u)) * (u*numpy.exp(v) - 2*numpy.exp(-u) ) ])

# Question 5. ******************************************************************
# Set up gradient descent.
uv = numpy.array((1.,1.))
eta = 0.1
nIterations = 20
terminationResidual = 10e-14
history5 = [[1.,1.,E(uv)]]
# Iterate.
for i in range(nIterations):
	# Get gradient, scale by learning rate and add to current coords.
	# Negative because we want to descent.
	uv -= (gradE(uv) * eta)
	# Record for plot.
	history5.append([uv[0], uv[1], E(uv)])
	# Terminate if necessary.
	if E(uv) < terminationResidual:
		break

print "Question 5: Reached E = {n:.2E} after {m:g} iterations.".format(n=E(uv), m=i+1)

# Question 6. ******************************************************************
print "Question 6: Final (u,v): ({n1:2.3f}, {n2:2.3f}).".format(n1 = uv[0], n2=uv[1])


# Question 7. ******************************************************************
# Set up gradient descent.
uv = numpy.array((1.,1.))
eta = 0.1
nIterations = 15
history7 = [[1.,1.,E(uv)]]
# Iterate.
for i in range(nIterations):
	# Get gradient and scale by learning rate.
	# Use only the u component.
	gradU = gradE(uv)[0] * eta
	# Move into gradU direction.
	# Negative because we want to descent.
	uv -= numpy.array([gradU, 0])
	# Record for plot.
	history7.append([uv[0], uv[1], E(uv)])
	# Again, get gradient and scale by learning rate.
	# Use only the v component.
	gradV = gradE(uv)[1] * eta
	# Move into gradV direction.
	# Negative because we want to descent.
	uv -= numpy.array([0, gradU])
	# Record for plot.
	history7.append([uv[0], uv[1], E(uv)])


print "Question 7: Reached E = {n:.2E} after {m:g} iterations.".format(n=E(uv), m=i+1)


# Plot surface.
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(uSurf, vSurf, eSurf, cmap=cm.autumn, alpha=.5 ,antialiased=False, linewidth=0)#, rstride=1, cstride=1)
# Plot all iterations question 5.
history5 = numpy.array(history5)
ax.plot(history5[:,0], history5[:,1], history5[:,2], 'w.-', linewidth=0.5)
ax.plot([history5[0,0]], [history5[0,1]], [history5[0,2]], 'r*')
ax.plot([history5[-1,0]], [history5[-1,1]], [history5[-1,2]], 'wv')
# Plot all iterations question 7.
(history7) = numpy.array(history7)
ax.plot(history7[:,0], history7[:,1], history7[:,2], 'g.-', linewidth=0.5)
ax.plot([history5[0,0]], [history5[0,1]], [history5[0,2]], 'r*')
ax.plot([history7[-1,0]], [history7[-1,1]], [history7[-1,2]], 'gv')
ax.set_zlim(0,5)
ax.set_xticks([-.5, -.25, 0, .25, .5, .75, 1])
ax.set_yticks([-1, -.75, -.5, -.25, 0, .25, .5, .75, 1])
plt.show()
