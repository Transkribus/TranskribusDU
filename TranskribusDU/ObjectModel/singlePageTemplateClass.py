# -*- coding: utf-8 -*-
"""

    single page template class 
    
    Hervé Déjean
    cpy Xerox 2016
    
    READ project
    
"""
from templateClass import templateClass
class singlePageTemplateClass(templateClass):
#     def __hash__(self):
#         return sequenceAPI.__hash__(self)
    def __init__(self):
        templateClass.__init__(self)
        self.headerZone=None
        self.footerZone=None
        self.mainZone=None
        
        # getSetofFeatures used for using this template
        self.myInitialFeatures = None
        
    def __str__(self):
        return 'singPage:main=%s'%(self.mainZone)

    def getPattern(self): return [self.mainZone.getPattern()]
    def setInitialFeatures(self,seqofF): self.myInitialFeatures = seqofF
    def findTemplatePartFromPattern(self,p): return  self.mainZone
    def registration(self,object):
        ## here send also the objects features used for the registration????
        raise "SOFTWARE ERROR: your component must define a testRun method"

        
      