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

import random
from generator import Generator

class numericalGenerator(Generator):
    """
        generic numerical Generator
    """
    def __init__(self,mean,sd):
        Generator.__init__(self)
        self._name='num'
        self._mean = mean
        self._std= sd
        
        self._value = None
                             
        self.realization= ['D', # digits : 101
                            'N' # text (hundred one)
                            ]
    
    def generate(self):
        self._value = random.uniform(self._mean,self._std)
        return (self,self._value)
    
    
if __name__ == "__main__":
    numGen= numericalGenerator(10,1)
    print numGen.generate()