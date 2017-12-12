import sys
import numpy
import gzip, cPickle
import types

"""
Given a data .pkl file, try to identify its content
"""

if len(sys.argv) < 2:
	print "Usage: %s <pkl-file>+"%sys.argv[0]
	exit(1)

for i in range(1, len(sys.argv)):
	sfn = sys.argv[i]
	print "\n===  %s  ====="%sfn
	try:
		o = cPickle.load(gzip.open(sfn, "rb"))
	except IOError:
		o = cPickle.load(open(sfn, "rb"))

	print len(o)
	try:
		lX, lY = o
		print "CONTAINS ([X0, ... ,Xn], [Y0, ... ,Yn])"
		print "   n = %d"%len(lX)
		assert len(lX) == len(lY)
	except ValueError:
		lO = o
		#is-it a lY or a lX ??
		try:
			NF, E, EF = lO[0]
			lX, lY = lO  , None
			print "CONTAINS [X0, ... ,Xn]"
			print "   n = %d"%len(lX)
		except:
			lX, lY = None, lO
			print "CONTAINS [Y0, ... ,Yn]"
			print "   n = %d"%len(lY)

	if lX:
		NF = numpy.vstack([NF for NF, E, EF in lX])	
		print "Consolidated node:            (N_node x N_feature) = ", NF.shape
		N = numpy.vstack([E for NF, E, EF in lX])	
		print "Consolidated number of edges: (N_edge x 2)         = ", N.shape
		EF = numpy.vstack([EF for NF, E, EF in lX])	
		print "Consolidated node:            (N_edge x N_feature) = ", EF.shape

	if lY:
		print "Total number of Ys", numpy.hstack(lY).shape
		if lX:
			for X, Y in zip(lX, lY):
				assert X[0].shape[0] == Y.shape[0], "Some Xs and Ys have INCONSISTENT shapes: %s and %s"%(X[0].shape[0], Y.shape[0])
			print "Xs and Ys have consistent shapes"	
