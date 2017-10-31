#!/usr/bin/python
#coding: utf8

import numpy
from matplotlib import pyplot as plt

choicesN = [10., 25., 100., 500., 1000.]
sigma = 0.1
d = 8

for N in choicesN:
	print numpy.power(sigma, 2) * (1. - ((d+1.)/N))
