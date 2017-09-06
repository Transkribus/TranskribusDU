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

import random
import cPickle


class Generator(object):
    def __init__(self):
        
        self._name = self.__class__.__name__
        # structure of the object: list of Generator
        self._structure = None

        self._generation =None
        self._value = None        
        self._lresources = None

    def __str__(self): return self._name
    def __repr__(self): return self._name
    
    #   getGeneratedValue()
    def getRandomElt(self,mylist):
        return mylist[random.randint(0,len(mylist)-1)]
    
    def parseElement(self,elt):
        """
            not used 
        """
        if type(elt)  == list:
            # alternative
            # select one element according to weight
            #  [ (a,pb), (b,pb) ]
            #  return a 
            pass
        elif type(elt) == tuple:
            # sequence
            # return the sequence
            pass
        elif type(elt) == Generator:
            return elt
            pass
        elif type(elt) == str:
            pass

    def loadResources(self,lfilenames):
        """
            Open and read resource files
            
            take just (Value,freq)
        """
        self._lresources =[]
        for filename in lfilenames:
            lre,ln = cPickle.load(open(filename,'r'))
            self._lresources.extend(ln)
        return self._lresources
              
    def getValue(self):
        """
            return value 
        """
        return self._value
    
    def getAnnotation(self):
        """
            return annotation
        """
        raise 'must be instanciated'

    def serialize(self):
        """
            create the final format
                string for textGen
                DSXML for DSdocument,...
            
        """
    def exportAnnotatedData(self):
        """
            generate annotated data for self
        """
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
        return (self,self._generation)
        

class textGenerator(Generator):
    def __init__(self,lang):
        import locale
        self.lang = lang
        locale.setlocale(locale.LC_TIME, self.lang)        
        Generator.__init__(self)

    def layout(self):
        """
            take text and add:
                CR  (can also split a token with X = Y)

            paramters: width and justification?  
                justification: random: l, r, c, any
                CR : 
             
        """
    def addNoise(self):
        """
            change content (ATR error)
            
            return new value?   self._noisyValue
            store the correct one somewhere
        """
        
