# -*- coding: utf-8 -*-
"""


    textGenerator.py

    create (generate) numerical annotated data 
     H. Déjean
    

    copyright Xerox 2017
    READ project 


    
    
    Developed  for the EU project READ. The READ project has received funding 
    from the European Union's Horizon 2020 research and innovation programme 
    under grant agreement No 674943.
"""




import random

from .generator import Generator

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
    def GTtokenize(self):return self._generation
    def serialize(self):
        return str(self._generation)

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
  
  
def test_integerGenerator():
    for  i in range(10):
        numGen= integerGenerator(1000,1000)
        numGen.generate()
        print(numGen.exportAnnotatedData([]))
          
if __name__ == "__main__":
    for  i in range(10):
        numGen= integerGenerator(1000,1000)
        numGen.generate()
        print(numGen.exportAnnotatedData([]))
    