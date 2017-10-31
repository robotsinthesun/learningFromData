#!/usr/bin/python
#coding: utf8

import numpy
from matplotlib import pyplot as plt


# All possible combinations of w·x and y.
# First two should return zero error, second two should return an error.
combinations = numpy.array([[-1, -1],
							[+1, +1],
							[-1, +1],
							[+1, -1]])

def errorA(wx, y):
	return numpy.exp(-y * wx)

def errorB(wx, y):
	return -y * wx

def errorC(wx, y):
	return numpy.power(y - wx, 2)

def errorD(wx, y):
	return numpy.log(1 + numpy.exp(-y * wx))

def errorE(wx, y):
	return -numpy.min([0, y * wx])

wxTest = numpy.linspace(-1.5, 1.5, 200)

# Plot the error functions for a range of w·x from -1.5 to 1.5 and for y = +1.
for errorFcn, name, color in zip([errorA, errorB, errorC, errorD, errorE], ["A","B","C","D","E"], ["r","b","g","k","c"]):
	print "Testing error method {e:s}.".format(e=name)
	errorYpos = [errorFcn(wx, 1) for wx in wxTest]
	plt.plot(wxTest, errorYpos, '-', c=color, label=name)


plt.gca().set_ylabel("e")
plt.gca().set_xlabel("$\mathbf{w}^T\cdot\mathbf{x}$")
plt.gca().set_title("Error functions for y = 1")
plt.grid()
plt.legend()
plt.show()

