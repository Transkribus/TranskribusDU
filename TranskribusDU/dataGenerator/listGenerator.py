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
            self._instance.append(o)
        return self
    
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
    
    lG =listGenerator(integerGenerator,integerGenerator(10,0),5,4)
    lG.instantiate()
    lG.generate()
    print(lG._generation)