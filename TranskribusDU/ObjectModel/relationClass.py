# -*- coding: utf-8 -*-
"""

    object class 
    
    Hervé Déjean
    cpy Naver Labs Europe 2019
    
    a class for (binary) relation 

"""
from __future__ import absolute_import
from __future__ import  print_function
from __future__ import unicode_literals

class  relationClass():
    """
        (binary) relation class
    """
    
    id = 0
    def __init__(self):
        self._name = None
        self.src =None
        self.tgt = None
        
        self.srcId = None
        self.srcTgtId = None
        # characteristics
        self._lAttributes = {}

    
    def __str__(self): return self.getName()
    def __repr__(self): return "%s"%self.getName()
    
    def getName(self): return self._name
    def setName(self,n): self._name  = n
    def getObjects(self): 
        return self._lObjects

    def getSrcTgtById(self,idsrc,idtgt):
        """
            
        """    
    def setSource(self,s): self.src = s
    def getSource(self): return self.src
    def setTarget(self,t): self.tgt = t
    def getTarget(self): return self.tgt
        
    def setSourceId(self,s): self.srcId = s
    def getSourceId(self): return self.srcId
    def setTargetId(self,t): self.tgtId = t
    def getTargetId(self): return self.tgtId
    
    def addAttribute(self,name,value): 
        self._lAttributes[name] = value
        
    def hasAttribute(self,name):
        try:
            self._lAttributes[name]
            return True
        except KeyError: 
            return False
    
    def getAttribute(self,name): 
        try: return self._lAttributes[name]
        except KeyError: return None


    def display(self,level=0):
        margin = " " * level
        print (margin,self.getName())

        for obj in self.getObjects():
            obj.display(level+1)
            
    
