# -*- coding: utf-8 -*-
"""

    single page template class 
    
    Herv� D�jean
    cpy Xerox 2016
    
    READ project
    
"""
from templateClass import templateClass
class singlePageTemplateClass(templateClass):
#     def __hash__(self):
#         return sequenceAPI.__hash__(self)
    def __init__(self):
        
        self.headerZone=None
        self.footerZone=None
        self.mainZone=None
        
        # getSetofFeatures used for using this template
        self.myInitialFeatures = None
        
    def setInitialFeatures(self,seqofF): self.myInitialFeatures = seqofF
    
    def registration(self,object):
        ## here send also the objects features used for the registration????
        raise "SOFTWARE ERROR: your component must define a testRun method"

        
      