#!/usr/bin/python

# Define input space.
Xx = [[bool(int(digit)) for digit in list(bin(i)[2:].zfill(3))]   for i in range(8)  ]

# Define sample data.
Dx = Xx[0:5]
Dy = [False, True, True, False, True]


# Define out of sample data.
Ox = Xx[5:]

# Define possible output for out of sample data.
# Three bools --> 8 possibilities.
Oy = [[bool(int(digit)) for digit in list(bin(i)[2:].zfill(3))]   for i in range(8)  ]

# Evaluate the outcome of the hypotheses:
#	f(a b c) = 1
#	f(a b c) = 0
#	f(a b c) = a xor b xor c
#	f(a b c) = not (a xor b xor c)
def hypothesis(input, i):
	if i == 0:
		return True
	elif i == 1:
		return False
	elif i == 2:
		return input[0] ^ input[1] ^ input[2]
	elif i == 3:
		return not (input[0] ^ input[1] ^ input[2])

# For each hypothesis, we count the number of matches for the inputs with all 8 possible outputs.
# Loop through hypotheses.
result = []
for g in range(4):
	print "Evaluating hypothesis {num:g}. ******************".format(num=g)
	# Reset score.
	# Holds number of triple matches, double matches and single matches.
	hypothesisScore = [0,0,0]
	# Loop through possible outputs.
	for y in Oy:
		print "   Current output set: {num:s}".format(num=y)
		matchCounter = 0
		# Loop through out of sample data.
		for ix in range(len(Ox)):

			if y[ix] == hypothesis(Ox[ix], g):
				print "      Current sample: {x:s} --> {y:s}. Match!".format(x=str(Ox[ix]), y=str(y[ix]))
				matchCounter += 1
			else:
				print "      Current sample: {x:s} --> {y:s}.".format(x=str(Ox[ix]), y=str(y[ix]))
		# Add to score.
		if matchCounter == 3:
			hypothesisScore[0] += 1
		elif matchCounter == 2:
			hypothesisScore[1] += 1
		elif matchCounter == 1:
			hypothesisScore[2] += 1

	# Sum up the scores with weights. 3 for triple match, 2 for double match and 1 for single match.
	hypothesisScore = hypothesisScore[0]*3 + hypothesisScore[1]*2 + hypothesisScore[2]*1
	result.append(hypothesisScore)

print result
