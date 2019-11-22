
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

import random
import numpy as np
import json
import pickle
import gzip

class Generator(object):
    ID=0
    
    lClassesToBeLearnt=[]
    def __init__(self,config=None,configKey=None):

        self.config=config    
        self._configKey = configKey
        # structure of the object: list of Generators with alterniatives (all possible structures)
        self._structure = None

        # instance: one selected structure among _structure
        self._instance  = None
        # concrete instance
        self._generation = None
        # 
        self._serialization = None

#         # for text only?
        self._value = None        
#         self._lresources = None
#         self.isWeighted = False

        # contains GT version  (X,Y)
        self._GT= None
        
        self.ID  = Generator.ID
        Generator.ID+=1


        self.noiseType = 0 
        self.noiseLevel = 0
        
        # nice label for ML (getName is too long)
        # default name
        self._label = self.getName()
    
        self._parent =None    
        # position in a list (Generated by listGen)
        self._number= None
        
        # weighted resources:
        self.totalW = None
        self._prob = None
        self._flatlr= None
        self._lweightedIndex = None
        self.isWeighted = False
        
    def __str__(self): return self.getName()
    def __repr__(self): return self.getName()
    
    def setConfig(self,c): self.config = c
    def getConfig(self): return self.config
    
    def setConfigKey(self,k): self._configKey = k
    def getConfigKey(self): return self._configKey
    
    def getMyConfig(self):
        try: return self.getConfig()[self._configKey]
        except KeyError: return None    
    
    def getLabel(self): return self._label
    def setLabel(self,l): self._label= l 
    
    def setClassesToBeLearnt(self,l): 
        self.lClassesToBeLearnt = l
        if self._structure is not None:
            for x in [x[0] for se in self._structure for x in se[:-1]]:
                x.setClassesToBeLearnt(l)

    
    def setNoiseType(self,t): self.noiseType = t
    def getNoiseType(self): return self.noiseType 
    def setNoiseLevel(self,t): self.noiseLevel = t
    def getNoiseLevel(self): return self.noiseLevel    
#     def getName(self): return "%s_%d"%(self.__class__.__name__ ,self.ID)
    def getName(self): return self.__class__.__name__
    
    def getParent(self): return self._parent
    def setParent(self,p):self._parent = p
    # when generated by listGenerator
    def setNumber(self,n): self._number = n
    def getNumber(self): return self._number
    
    def getInstances(self,iclass):
        lRet = []
        if type(self._generation) != list:
            return []
        for obj in self._generation:
            if isinstance(obj,iclass):
                lRet.append(obj)
            lRet.extend(obj.getInstances(iclass))
        return lRet   
        
    
    def loadResourcesFromList(self,lLists,iMax=100000):
        """
            Open and read resource files
            take just (Value,freq)
        """
        self._lresources =[]
        for mylist in lLists:
            self._lresources.extend(mylist)
        if self.totalW is None:
            self.totalW = 1.0 * sum(list(map(lambda xy:xy[1], self._lresources)))
        if self.totalW != len(self._lresources):
            self.isWeighted = True
        if self._prob is None:
            self._prob = list(map(lambda  xy:xy[1] / self.totalW,self._lresources))           
        if self._flatlr is None:
            self._flatlr = list(map(lambda  xy:xy[0],self._lresources))
        # generate many (100000) at one ! otherwise too slow
        self._lweightedIndex  = list(np.random.choice(self._flatlr,iMax,p=self._prob))
#         print(self.isWeighted,self.totalW,self._lweightedIndex )
        return self._lresources        

    def loadResources(self,lfilenames):
        """
            Open and read resource files
            
            take just (Value,freq)
        """
        self._lresources =[]
        for filename in lfilenames:
            res = pickle.load(gzip.open(filename,'r'))
            self._lresources.extend(res)
        if self.totalW is None:
            self.totalW = 1.0 * sum(list(map(lambda  xy:xy[1], self._lresources)))
        if self.totalW != len(self._lresources):
            self.isWeighted = True
        if self._prob is None:
            self._prob = list(map(lambda  xy:xy[1] / self.totalW,self._lresources))           
        if self._flatlr is None:
            self._flatlr = list(map(lambda  xy:xy[0],self._lresources))
        # generate many (100000) at one ! otherwise too slow
        self._lweightedIndex  = list(np.random.choice(self._flatlr,100000,p=self._prob))

        return self._lresources
              
    def getValue(self):
        """
            return value 
        """
        return self._value    
    
    #   getGeneratedValue()
    def getRandomElt(self,mylist):
        if self.isWeighted:
            # for textOnly?? 
            return self.getWeightedRandomElt(mylist)
        else:
            ii =random.randint(0,len(mylist)-1)
            return mylist[ii]
        
    def getWeightedRandomElt(self,myList):
        """
            weight the drawing with element weight (frequency)
            
            too slows to draw each time a value:
                - > generate many values and pop when needed!
        """
        # need to generate again if pop empty
        try:
            ind= self._lweightedIndex.pop()
        except IndexError:
            self._lweightedIndex  = list(np.random.choice(self._flatlr,100000,p=self._prob))
            ind= self._lweightedIndex.pop()
        return ind
        
#         ret = np.random.choice(self._flatlr,1,p=self._prob)[0]  
#         if type(ret) ==  np.unicode_:
#             ret = ret
#         return ret     

    def reportMe(self):
        """
            return the instanciated form
        """
        return self._instance
        
    def serialize(self):
        """
            create the final format
                string for textGen
                DSXML for DSdocument,...
            
        """
        raise Exception('must be instantiated')
    
#     def exportAnnotatedData(self,foo=[]):
#         """
#             generate annotated data for self
#             build a full version of self._generation: integration of the subparts (subobjects)
# 
#         """
#         raise Exception( 'must be instantiated',self)
    def exportAnnotatedData(self,foo):
        """
            build a full version of generation: integration of the subparts (subtree)
            
            what are the GT annotation for document?  
             
        """
        ## export (generated value, label) for terminal 

        self._GT=[]
        for obj in self._generation:
            self._GT.append((obj.exportAnnotatedData([]),obj))
        
        return self._GT    
    
    def instantiate(self):
        """
            select using proba stuff the final realisation in terms of structure: no generation
        """
        if self._structure is None:
            self._instance = (self,)
        else:
            self._instance  = []
            structproba = self.getRandomElt(self._structure)
            print (structproba)
            struct, proba = structproba[:-1], structproba[-1]
            # terminal textual stuff is not tuple but unicode: the generateProb need to be more efficient
            if type(struct) in [ tuple,list] :
                for obj, _,proba in struct:
                    if obj is not None:
                        generateProb = 1.0 * random.uniform(1,100)
                        if generateProb < proba:
                            self._instance.append(obj.instantiate())
        assert  self._instance != [], (self,struct,proba)
        return self        
        
    
    def generate(self):
        """
            return object
        """
        self._generation  = []
        for obj in self._instance:
            obj.generate()
            self._generation.append(obj)
        return self    
    
    
    def noiseErase(self):
        """
            element is not generated
        """
        raise Exception('must be instantiated')
    
    def noiseMerge(self):
        """
            if self has several _structural/generated elements
            test if there is a merge operation (draw)
                weight: add as parameter of self (in _structural)
            if merge: select two consecutive elements and merge them: 
            annotation: add L1_L2 as label?
            
            How to select elements?: (merge of lines from several columns)
                what is 'physically' near may not be near form a structural view point
        """
        raise Exception('must be instantiated')
    
    def noiseSplit(self):
        """
            test if there is a split  operation (draw)  on self
            if split: select two consecutive elements and merge them: 
            annotation: add Split1, SplitN in label?
        """
        raise Exception('must be instantiated')
    
    def saveconfig(self,config,filename):
        """
            json dump of the config
        """
        try: 
            f = open(filename,"wb",encoding='utf-8')
            json.dump(f,config,indent=True)
        except IOError:print('not possible to open %s.'%(filename))
        
    def loadconfig(self,filename):
        try: 
            f = open(filename,"rb",encoding='utf-8')
            return json.load(f)
        except IOError:print('not possible to open %s.'%(filename))
        
    
if __name__ == "__main__":

    g= Generator()
    print(g)
