# -*- coding: utf-8 -*-
"""

    Vertical Zones template class: virutal template representinf a 'canonical' layout having vertical zones
        Basic example: a n-columns page
    
    Herv� D�jean
    cpy Xerox 2016
    
    READ project
    
"""
from templateClass import templateClass
class verticalZonestemplateClass(templateClass):
    ## integrate the structure of  "n-grams" 
    ## 
#     def __hash__(self):
#         return sequenceAPI.__hash__(self)
    def __init__(self):
        
        
        # list of X1 positions for the cts
        self.lX=[]
        
        #list of the zone width
        self.lWidth=[]
        
        # getSetofFeatures used for using this template
        self.myInitialFeatures = None
        
    def setInitialFeatures(self,seqofF): self.myInitialFeatures = seqofF
    
    def setXCuts(self,lx): self.lX = lx
    def addXCuts(self,x): 
        if x not in self.lX:
            self.lX.append(x)
            self.lX.sort()    
    def getXCuts(self): return self.lX
    
    def registration(self,object):
        ## here send also the objects features used for the registration????
        raise "SOFTWARE ERROR: your component must define a testRun method"

        
      