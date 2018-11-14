# -*- coding: utf-8 -*-
"""

    Vertical Zones template class: virutal template representinf a 'canonical' layout having vertical zones
        Basic example: a n-columns page
    
    Hervé Déjean
    cpy Xerox 2016
    
    READ project
    
"""
from __future__ import absolute_import
from __future__ import  print_function
from __future__ import unicode_literals

import numpy as np
from scipy.optimize import linear_sum_assignment

from spm.frechet import frechetDist
from .templateClass import templateClass

class verticalZonestemplateClass(templateClass):
    """
        a class  for vertical layout  (like columns) 
    """
    gID = 1
    def __init__(self):
        
        templateClass.__init__(self)

        self.ID = verticalZonestemplateClass.gID 
        verticalZonestemplateClass.gID += 1
        self._TH = 20
        self.bMirrored= False
        # list of X1 positions for the cts
        self.lX= []
        
        #list of the zone width
#         self.lWidth= []
        
        #for mirrored template
        self.brother=None
    
    def __str__(self): return 'VZTemp:'+ str(self.pattern)
    def __repr__(self): return 'VZTemp:'+ str(self.pattern)

    def __hash__(self):
        return hash(self.ID)
        
    def setPattern(self,p): 
        """
            an itemset
        """
        
        self.pattern = p
        self.bMirrored = len(self.pattern) == 2
        
#         lSubWith=[]
#         prev = 0
        for item in self.pattern:
            self.addXCut(item)
#             w= item.getValue() - prev
#             if item.getValue() not in self.lWidth:
#                 lSubWith.append(w)
#             prev= item.getValue()
#         self.lWidth.append(lSubWith)
    
    def setTH(self,t): self._TH = t
    def getTH(self): return self._TH
    
    def setXCuts(self,lx): self.lX = lx
    
    def addXCut(self,x): 
        if x not in self.lX:
            self.lX.append(x)
            self.lX.sort()    
    
    def getXCuts(self): return self.lX
    
    
    def isRegularGrid(self):
        """
            is the template a regular grid (N columns= page justification)
        """
        
        return False
        
    
    def findBestMatch2(self,lRegCuts,lCuts):
        """
            best match using hungarian
            add a threshold!
        """
        cost_matrix=np.zeros((len(lRegCuts),len(lCuts)),dtype=float)
        
        for a,refx in enumerate(lRegCuts):
            for b,x in enumerate(lCuts):
                dist = refx.getDistance(x)
                cost_matrix[a,b]=dist
            
        r1,r2 = linear_sum_assignment(cost_matrix)
        
        ltobeDel=[]
        for a,i in enumerate(r2):
            #if cost is too high: cut the assignment?
            print (a,i,r1,r2,lRegCuts[a],lCuts[i], cost_matrix[a,i])
            if cost_matrix[a,i] > 100:
                ltobeDel.append(a)
        r2 = np.delete(r2,ltobeDel)
        r1 = np.delete(r1,ltobeDel)
#         print ('\t',r1,r2,ltobeDel,lRegCuts,lCuts)
        # score Fréchet distance etween two mapped sequences    
        return r1,r2,None
    
             
    def computeScore(self,p,q):
        d =frechetDist(list(map(lambda x:(x.getValue(),0),p)),list(map(lambda x:(x.getValue(),0),q)))
#         print (d,list(map(lambda x:(x.getValue(),0),p)),list(map(lambda x:(x.getValue(),0),q)))
        if d == 0:
            return 1
        return 1/(frechetDist(list(map(lambda x:(x.getValue(),0),p)),list(map(lambda x:(x.getValue(),0),q))))
                  
    def registration(self,anobject):
        """
            'register': match  the model to an object
            can only a terminal template 
        """
        lobjectFeatures = anobject.lFeatureForParsing
#         lobjectFeatures = anobject._fullFeaturesx
        print (anobject, lobjectFeatures, self)
        # empty object
        if lobjectFeatures == []:
            return None,None,-1
        
#         print self.getPattern(), lobjectFeatures
        try:  self.getPattern().sort(key=lambda x:x.getValue())
        except: pass ## P3 < to be defined for featureObject
#         print self.getPattern(), anobject, lobjectFeatures
        foundReg,bestReg, _ = self.findBestMatch2(self.getPattern(), lobjectFeatures)

#         bestReg, _ = self.findBestMatch(self.getPattern(), lobjectFeatures)
#         print bestReg, curScore
        if bestReg != []:
            lFinres = list(zip([(lobjectFeatures[i]) for i in bestReg], ([self.getPattern()[i] for i in foundReg])))
#             print (lFinres)
#             score1 = self.computeScore(len(self.getPattern()), lFinres, [],lobjectFeatures)
#             print (bestReg, self.getPattern(),[(self.getPattern()[i]) for i in bestReg]) 
#             score1 = self.computeScore([(self.getPattern()[i]) for i in foundReg], lobjectFeatures)
            score1 = self.computeScore(self.getPattern(), lobjectFeatures)

            return lFinres,None,score1
        else:
            return None,None,-1    
    
    
    
    