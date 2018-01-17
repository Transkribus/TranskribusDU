# -*- coding: utf-8 -*-
#
# A simple chronometer
#
# JL Meunier, May 2004
#
# Copyright Xerox 2004
#

from __future__ import absolute_import
from __future__ import  print_function
from __future__ import unicode_literals

import time

class Chrono:
	"""
	A utility class to compute elapsed time.
	Typical usage:
	c = Chrono().on()
	...
	print "elapsed time: %.1fs" % c.off()
	"""

	def __init__(self, nbDigit=1):
		self.t0 = None
		self.nbDigit = nbDigit

	def on(self):
		self.t0 = time.time()
		return self

	def get(self):
		t = time.time()
		return round(t - self.t0, self.nbDigit)        
	
	def off(self):
		t = time.time()
		return round(t - self.t0, self.nbDigit)

#--- obsolete interface

ltChrono = []
#Start a chronometer, you can start several of them
def chronoOn(name=None):
    global ltChrono
    c = Chrono().on()
    ltChrono.append((c,name))

#stop the last started chronometer and returns its value in second
def chronoOff(expected_name=None):
    global ltChrono
    c,name = ltChrono.pop()
    assert name == expected_name, "INTERNAL ERROR: chronoOn and chronoOff calls not properly nested"
    return c.off()


#----------   SELF-TEST    --------------
if __name__ == "__main__":

    print ("Selft-test")
    chronoOn()
    time.sleep(1)
    v = chronoOff()
    print (v==1, v)

    chronoOn()
    time.sleep(1)

    chronoOn()
    time.sleep(2.2)
    v = chronoOff()
    print ( v==2.2, v)
    
    c = Chrono().on()
    print("elapsed time: %.1fs" % c.off())

    
#    v = chronoOff(2)
#    print abs(round(v-3)) < 0.5, v
    
