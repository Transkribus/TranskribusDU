# -*- coding: utf-8 -*-
"""


    textGenerator.py

    create (generate) textual annotated data 
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

import os
import cPickle
import csv


from generator import Generator 


class textGenerator(Generator):
    def __init__(self,lang):
        import locale
        self.lang = lang
        locale.setlocale(locale.LC_TIME, self.lang)        
        Generator.__init__(self)


    def getSeparator(self):
        """
            separator between token
        """
        return " "
    
    def generate(self):
        self._generation = self.getRandomElt(self._value)
        return self
    
    def exportAnnotatedData(self):
        ## export (generated value, label) for terminal 
        self._GT  = []
        
        if type(self._generation) == unicode:
            self._GT.append((self._generation,self.getName()))
        elif type(self._generation) == int:
            self._GT.append((self._generation,self.getName()))         
        else:
            for i,obj  in enumerate(self._generation):
                if type(obj) == unicode:
                    self._GT.append((obj._generation,self.getName()))
                elif type(obj) == int:
                    self._GT.append((obj._generation,self.getName()))                       
                else:
                    self._GT.extend(obj.exportAnnotatedData())
        return self._GT 
            
    def serialize(self):
        self._serialization  = ""
        
        #terminal
        if type(self._generation) == unicode:
            self._serialization +=  self._generation
        elif type(self._generation) == int:
            self._serialization += "%d"%self._generation            
        else:
            for i,obj  in enumerate(self._generation):
                if type(obj) == unicode:
                    self._serialization +=  obj
                elif type(obj) == int:
                    self._serialization +=  "%d"%self._generation                         
                else:
                    if i == 0:
                        self._serialization +=  obj.serialize()
                    else:
                        self._serialization += self.getSeparator() + obj.serialize()                        
    #             self._serialization += self.getSeparator() + obj.serialize()
        return self._serialization    
    

if __name__ == '__main__':
    print "toto"
    cmp.textGenerator()
    cmp.createCommandLineParser()
    dParams, args = cmp.parseCommandLine()
    
    #Now we are back to the normal programmatic mode, we set the componenet parameters
    cmp.setParams(dParams)
    #This component is quite special since it does not take one XML as input but rather a series of files.
    #doc = cmp.loadDom()
    doc = cmp.run()    