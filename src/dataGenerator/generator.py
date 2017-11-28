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

import numpy as np

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
        
        
        # weighted resources:
        self.totalW = None
        self._prob = None
        self._flatlr= None
        self._lweightedIndex = None
        self.isWeighted = False
        
    def __str__(self): return self.getName()
    def __repr__(self): return self.getName()
    
        
    def getLabel(self): return self._label
    def setLabel(self,l): self._label= l 
    
    def getName(self): return self.__class__.__name__
    
    #   getGeneratedValue()
    def getRandomElt(self,mylist):
        if self.isWeighted:
            ret= self.getWeightedRandomElt(mylist)
            return self.getWeightedRandomElt(mylist)
        else:
            ii =random.randint(0,len(mylist)-1)
            return mylist[ii]
    
    def getWeightedRandomElt(self,myList):
        """
            weight the drawing with element weight (frequency)
            
            too slows to draw each time a value:
                - > generate man values and pop when needed!
        """
        # need to generate again if pop empty
        try:
            ind= self._lweightedIndex.pop()
        except IndexError:
            self._lweightedIndex  = list(np.random.choice(self._flatlr,100000,p=self._prob))
            ind= self._lweightedIndex.pop()
        return unicode(ind)
    
        ret = np.random.choice(self._flatlr,1,p=self._prob)[0]  
        if type(ret) ==  np.unicode_:
            ret = unicode(ret)
        return ret        
       
       
        
    def loadResourcesFromList(self,lLists):
        """
            Open and read resource files
            take just (Value,freq)
        """
        self._lresources =[]
        for mylist in lLists:
            self._lresources.extend(mylist)
        if self.totalW is None:
            self.totalW = 1.0 * sum(map(lambda (_,y):y, self._lresources))
        if self.totalW != len(self._lresources):
            self.isWeighted = True
        if self._prob is None:
            self._prob = map(lambda (_,y):y / self.totalW,self._lresources)           
        if self._flatlr is None:
            self._flatlr = map(lambda (x,_):x,self._lresources)
        # generate many (100000) at one ! otherwise too slow
        self._lweightedIndex  = list(np.random.choice(self._flatlr,100000,p=self._prob))

        return self._lresources        

    def loadResources(self,lfilenames):
        """
            Open and read resource files
            
            take just (Value,freq)
        """
        self._lresources =[]
        for filename in lfilenames:
            res = cPickle.load(gzip.open(filename,'r'))
            self._lresources.extend(res)
        if self.totalW is None:
            self.totalW = 1.0 * sum(map(lambda (_,y):y, self._lresources))
        if self.totalW != len(self._lresources):
            self.isWeighted = True
        if self._prob is None:
            self._prob = map(lambda (_,y):y / self.totalW,self._lresources)           
        if self._flatlr is None:
            self._flatlr = map(lambda (x,_):x,self._lresources)
        # generate many (100000) at one ! otherwise too slow
        self._lweightedIndex  = list(np.random.choice(self._flatlr,100000,p=self._prob))

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
                    obj.generate()
                    self._generation.append(obj)
                else:
                    pass
        return self
    
if __name__ == "__main__":

    g= Generator()
    g.loadResources(['resources/profession.pkl'])
    g.generate()
    print g
