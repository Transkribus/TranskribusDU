# -*- coding: utf-8 -*-
"""


    Generator.py

    create (generate) annotated data 
     H. Déjean
    

    copyright Xerox 2017
    READ project 

    
    
    Developed  for the EU project READ. The READ project has received funding 
    from the European Union's Horizon 2020 research and innovation programme 
    under grant agreement No 674943.
"""
from __future__ import absolute_import
from __future__ import  print_function
from __future__ import unicode_literals

from dataGenerator.generator import Generator 

try:basestring
except NameError:basestring = str

class listGenerator(Generator):
    """
           a generator  for list
    """
    def __init__(self,config,configKey,objGen,nbMaxGen):
        Generator.__init__(self,config,configKey)
        self.configObj=configKey
        self.myObjectGen = objGen
        self.nbMax = nbMaxGen
        
    def getValuedNb(self): return self.nbMax._generation
    
    
    def instantiate(self):
        self._instance = []
        self.nbMax.generate()
        for i in range(self.nbMax._generation):
            try: 
                o = self.myObjectGen(self.getConfig(),self.configObj)
                o.instantiate()
            except TypeError:  
                o = self.myObjectGen(*self.getConfig(),self.configObj)
                o.instantiate()
            
            o.setNumber(i)
            self._instance.append(o)
        return self
    
    def exportAnnotatedData(self,foo):

        self._GT=[]
        for obj in self._generation:
            if type(obj._generation) == basestring:
                self._GT.append((obj._generation,[obj.getLabel()]))
            elif type(obj) == int:
                self._GT.append((obj._generation,[obj.getLabel()]))
            else:
                self._GT.append((obj.exportAnnotatedData([]),obj))
        
        return self._GT    
    
if __name__ == "__main__":
    
    from dataGenerator.numericalGenerator import integerGenerator
    integerGenerator(10,0)
    lG = listGenerator((5,4),integerGenerator, integerGenerator(10,0))
    lG.instantiate()
    lG.generate()
    print(lG._generation)