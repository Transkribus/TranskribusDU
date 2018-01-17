# -*- coding: utf-8 -*-
"""


    Generator.py

    create (generate) annotated data 
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
from __future__ import absolute_import
from __future__ import  print_function
from __future__ import unicode_literals

from .generator import Generator 


class listGenerator(Generator):
    """
           a generator  for list
    """
    def __init__(self,objGen,nbMaxGen,*objParam):
        Generator.__init__(self)
        self.myObjectGen = objGen
        self.objParams = objParam
        self.nbMax = nbMaxGen
        
    def getValuedNb(self): return self.nbMax._generation
    
    
    def instantiate(self):
        self._instance = []
        self.nbMax.generate()
        for i in range(self.nbMax._generation):
            o = self.myObjectGen(*self.objParams).instantiate()
            o.setNumber(i)
#             print "xx",o,o.getNumber()
            self._instance.append(o)
#             print "linst:",self.getLabel(), self.nbMax._generation, o
#         print self.getLabel(), self._instance
        return self
    
#     def generate(self):
#         ss
#         self._generation  = []
#         for obj in self._instance:
#             print "lsitgen", obj.getLabel()
#             obj.generate()
#             self._generation.append(obj)
#         return self
    
    def exportAnnotatedData(self,foo):

        self._GT=[]
        for obj in self._generation:
            if type(obj._generation) == unicode:
                self._GT.append((obj._generation,[obj.getLabel()]))
            elif type(obj) == int:
                self._GT.append((obj._generation,[obj.getLabel()]))
            else:
                self._GT.append((obj.exportAnnotatedData([]),obj))
        
        return self._GT    
    
if __name__ == "__main__":
    
    from .numericalGenerator import integerGenerator
    
    lG =listGenerator(integerGenerator,10,100,5)
    lG.generate()
    print(lG._generation)