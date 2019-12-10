# -*- coding: utf-8 -*-
"""


    textNoiseGenerator.py

    create (generate) random char 
     H. Déjean
    

    copyright Xerox 2017
    READ project 


    
    
    Developed  for the EU project READ. The READ project has received funding 
    from the European Union's Horizon 2020 research and innovation programme 
    under grant agreement No 674943.
"""




import random
import string
from .textGenerator import textGenerator

class textRandomGenerator(textGenerator):
    """
        generic numerical Generator
    """
    def __init__(self,length,sd):
        textGenerator.__init__(self,lang=None)
        self._name='txtnoise'
        self._length= length
        self._std = sd
        self._generation = None
        self._value = None
                             
    
    def exportAnnotatedData(self,lLabels):
        lLabels.append(self.getName())
        self._GT = [(self._generation,lLabels[:])]    
        return self._GT 
    def generate(self):
        self._generation=""
        for i in range(int(round(random.gauss(self._length,self._std)))):
            self._generation += random.choice(string.ascii_letters+ string.punctuation)

        return self
    
    def serialize(self):
        return self._generation
    
    
    def noiseSplit(self):
        textGenerator.noiseSplit(self)
        
class textletterRandomGenerator(textRandomGenerator):
    def generate(self):
        self._generation=""
        for i in range(int(round(random.gauss(self._length,self._std)))):
            self._generation += random.choice(string.ascii_letters)
        return self

if __name__ == "__main__":
    for i in range(10):
        nGen= textRandomGenerator(10,10)
        nGen.generate()
        print(nGen.serialize())