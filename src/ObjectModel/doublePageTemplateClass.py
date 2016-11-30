# -*- coding: utf-8 -*-
"""

    double page template class: [left, right] double page representation 
    
    Hervé Déjean
    cpy Xerox 2016
    
    READ project
    
"""
from templateClass import templateClass
class doublePageTemplateClass(templateClass):
#     def __hash__(self):
#         return sequenceAPI.__hash__(self)
    def __init__(self):
        
        
        # will contained other templates for each page
        self.leftPage = None 
        self.rightPAge = None        

        # getSetofFeatures used for using this template
        self.myInitialFeatures = None
        
    def setInitialFeatures(self,seqofF): self.myInitialFeatures = seqofF
    
    def registration(self,object):
        ## need two objects ???
        ## here send also the objects features used for the registration????
        raise "SOFTWARE ERROR: your component must define a testRun method"

        
      