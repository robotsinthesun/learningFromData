#!/usr/bin/python
import numpy

def combination(n,k):
	return numpy.math.factorial(n) / (numpy.math.factorial(k) * numpy.math.factorial(n-k))

N = 10

print "Maximum for growth function with N = {n:g}: {m:g}.".format(n=N, m=pow(2,N))

# 1:
result = N + 1
print "i: {num:2.2f}.".format(num=result)

if result <= pow(2,N):
	print "   Pass"
else:
	print "   Fail"


# 2:
result = 1 + N + combination(N, 2)
print "ii: {num:2.2f}.".format(num=result)
if result <= pow(2,N):
	print "   Pass"
else:
	print "   Fail"


# 3:
upperLimit = int(numpy.floor(numpy.sqrt(N)))
result = numpy.array([combination(N,i+1) for i in range(upperLimit)]).sum()
print "iii: {num:2.2f}.".format(num=result)
if result <= pow(2,N):
	print "   Pass"
else:
	print "   Fail"


# 4:
result = pow(2, numpy.floor(N/2))
print "iv: {num:2.2f}.".format(num=result)
if result <= pow(2,N):
	print "   Pass"
else:
	print "   Fail"


# 5:
result = pow(2, N)
print "v: {num:2.2f}.".format(num=result)
if result <= pow(2,N):
	print "   Pass"
else:
	print "   Fail"
