# -*- coding: utf-8 -*-
"""

    object class 
    
    Hervé Déjean
    cpy Xerox 2009
    
    a class for object
"""
from __future__ import absolute_import
from __future__ import  print_function
from __future__ import unicode_literals


class sequenceAPI():
    
    def __init__(self):
        
        # how to create features
        self._featureFunction = None
        # threshold value for equality
        self._featureFunctionTH = None
        # which objetc/subobject to use
        self._subObjects=None
#         self.objectClass = None

        # which features to use
        self._lFeatureList= None
        
        # list of features 
        self._lBasicFeatures = None
        
        # set of canonical features
        self._canonicalFeatures = None
        
        self._subObjects=None
    def __hash__(self):
        try:
            return hash(str(list([x.getID() for x in self.getSetofFeatures()])))
        except: return hash('None')
    
    def __repr__(self):
#         return str(len(self._node.getContent())) + " " + self._node.getContent().encode('utf-8').strip()[:20]
        try:
            return len(self._node.getContent()) , " [" + self._node.name+self._node.getContent()[:20]+']'
        except AttributeError:
            # no node: use features?
            return str(self.getSetofFeatures())



    def resetFeatures(self):
        ## assume structure define elsewhere
        self.setStructures([])
        self._lBasicFeatures = None           
        
    
    def setFeatureFunction(self,foo,TH=5,lFeatureList = None,myLevel=None):
        """
            select featureFunction that have to be used
            
        """
        self._featureFunction = foo
        self._featureFunctionTH=TH    
        self._lFeatureList=lFeatureList
        self._subObjects = myLevel

    def computeSetofFeatures(self):
        self._featureFunction(self._featureFunctionTH,self._lFeatureList,self._subObjects)
        
    def setSequenceOfFeatures(self,l):
        self._lBasicFeatures = l
        
    def addFeature(self,f):
        if self._lBasicFeatures is not None: 
            if f not in self.getSetofFeatures():
                self._lBasicFeatures.append(f)
        else:
            self._lBasicFeatures = [f]
            
    def getSetofFeatures(self):
        return self._lBasicFeatures

    def getCanonicalFeatures(self): return self._canonicalFeatures
    def addCanonicalFeatures(self,f ):
        if self._canonicalFeatures is None:
            self._canonicalFeatures=[]
        if f not in self._canonicalFeatures:
            self._canonicalFeatures.append(f)
            
        
        
