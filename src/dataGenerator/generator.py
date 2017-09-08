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
from __future__ import unicode_literals

import random
import cPickle
import gzip

class Generator(object):
    def __init__(self):
        
        # structure of the object: list of Generator
        self._structure = None

        self._generation = None
        self._serialization = None
        self._value = None        

        self._lresources = None
        
        # nice label for ML (getName is too long)
        self._label = None

        # contains GT version  (X,Y)
        self._GT= None
    def __str__(self): return self.getName()
    def __repr__(self): return self.getName()
    
        
    def getLabel(self): return self._label
    def setLabel(self,l): self._label= l 
    
    def getName(self): return self.__class__.__name__
    
    #   getGeneratedValue()
    def getRandomElt(self,mylist):
        return mylist[random.randint(0,len(mylist)-1)]
    

    def loadResources(self,lfilenames):
        """
            Open and read resource files
            
            take just (Value,freq)
        """
        self._lresources =[]
        for filename in lfilenames:
            res= cPickle.load(gzip.open(filename,'r'))
            self._lresources.extend(res)
        return self._lresources
              
    def getValue(self):
        """
            return value 
        """
        return self._value

    def serialize(self):
        """
            create the final format
                string for textGen
                DSXML for DSdocument,...
            
        """
        raise Exception, 'must be instantiated'
    
    def exportAnnotatedData(self):
        """
            generate annotated data for self
        """
        raise Exception, 'must be instantiated'
    
    def generate(self):
        """
            return object : value, annotation
        """
        self._generation  = []
        structproba = self.getRandomElt(self._structure)
        struct, proba = structproba[:-1], structproba[-1]
        for obj, number,proba in struct:
            for i in range(number):
                generateProb = 1.0 * random.uniform(1,100)
                if generateProb < proba:
                    self._generation.append(obj.generate())
                else:
                    pass
        return self
        

