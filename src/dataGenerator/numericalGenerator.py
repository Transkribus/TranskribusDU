# -*- coding: utf-8 -*-
"""


    textGenerator.py

    create (generate) numerical annotated data 
     H. Déjean
    

    copyright Xerox 2017
    READ project 

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
    
    
    Developed  for the EU project READ. The READ project has received funding 
    from the European Union's Horizon 2020 research and innovation programme 
    under grant agreement No 674943.
"""
from __future__ import unicode_literals
import random
from generator import Generator

"""
     see http://www.southampton.ac.uk/~fangohr/blog/physical-quantities-numerical-value-with-units-in-python.html
     for physical meqsures 
     Ek = 0.5 * m * v **2
     Out[41]: 96450.617 m^2*kg/s^2
     
     wget https://bitbucket.org/birkenfeld/ipython-physics/raw/default/physics.py
     
"""
class numericalGenerator(Generator):
    """
        generic numerical Generator
    """
    def __init__(self,mean,sd):
        Generator.__init__(self)
        self._name='num'
        self._mean = mean
        self._std= sd
        
        self._generation = None
        self._value = None
                             
        self.realization= ['D', # digits : 101
                            'N' # text (hundred one)
                            ]
    
    def getUple(self): return (self._mean,self._std)
    def setUple(self,ms):
        m,s = ms 
        self._mean = m
        self._std = s

    def exportAnnotatedData(self,lLabels):
        lLabels.append(self.getLabel())
        self._GT = [(self._generation,lLabels[:])]    
        return self._GT     
    
    def generate(self):
        self._generation = random.gauss(self._mean,self._std)
        return self
    def GTtokenize(self):return unicode(self._generation)
    def serialize(self):
        return unicode(self._generation)

class positiveNumericalGenerator(numericalGenerator):
    def generate(self):
        self._generation = random.gauss(self._mean,self._std)
        while self._generation < 0:  
            self._generation = random.gauss(self._mean,self._std)            
        return self

class positiveIntegerGenerator(numericalGenerator):
    def generate(self):
        self._generation = int(round(random.gauss(self._mean,self._std)))
        while self._generation < 0:  
            self._generation = int(round(random.gauss(self._mean,self._std)))         
        return self
    
class integerGenerator(numericalGenerator):
    def generate(self):
        self._generation = int(round(random.gauss(self._mean,self._std)))
    
if __name__ == "__main__":
    for  i in range(10):
        numGen= integerGenerator(1000,1000)
        numGen.generate()
        print numGen.exportAnnotatedData([])
    