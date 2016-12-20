# -*- coding: utf-8 -*-
"""

    double page template class: [left, right] double page representation 
    
    Hervé Déjean
    cpy Xerox 2016
    
    READ project
    
"""
from templateClass import templateClass
from verticalZonesTemplateClass import verticalZonestemplateClass

class doublePageTemplateClass(templateClass):
#     def __hash__(self):
#         return sequenceAPI.__hash__(self)
    def __init__(self):
        templateClass.__init__(self)
        
        # will contained other templates for each page
        self.leftPage = None 
        self.rightPage = None        

        # getSetofFeatures used for using this template
        self.myInitialFeatures = None
    
    def __str__(self):
        return "doublepage:[%s] [%s]" %(self.leftPage,self.rightPage)
    def __repr__(self):
        return "doublepage:[%s] [%s]" %(self.leftPage,self.rightPage)
    
    
    def fromPattern(self,pattern):
        
        assert len(pattern)==2
        
        self.setPattern(pattern)
        self.leftPage = verticalZonestemplateClass()
        self.leftPage.setType('leftpage')
        self.leftPage.setPattern(pattern[0])
        self.leftPage.setParent(self)
        
        self.rightPage = verticalZonestemplateClass()
        self.rightPage.setType('rightpage')
        self.rightPage.setPattern(pattern[1])
        self.rightPage.setParent(self)
    
    
    def setInitialFeatures(self,seqofF): self.myInitialFeatures = seqofF
    
        
    def findTemplatePartFromPattern(self,pattern):
        if self.leftPage.getPattern() == pattern:
            return self.leftPage
        if self.rightPage.getPattern() == pattern:
            return self.rightPage
        return None
        
    def registration(self,object):
        ## need two objects ???
        ## here send also the objects features used for the registration????
        raise "SOFTWARE ERROR: your component must define a testRun method"

        
      