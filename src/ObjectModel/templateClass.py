# -*- coding: utf-8 -*-
"""

    template class: virutal template representinf a 'canonical' document object
        Basic example: a table table
        Has to be associated to a 'registration' step againt a page to be 'realized' for this page
    
    Template can be used in order to define a reading order among its sub-structure 


    Idea: associate vy default most common templates to a document:
        single-page, double-page
        single column, double column

    Hervé Déjean
    cpy Xerox 2016
    
    READ project
    
    
"""
from sequenceAPI import sequenceAPI
from objectClass import objectClass

class templateClass(objectClass,sequenceAPI):
    """
        a template can include other templates   (verticalZones contains gridlines, table contains rows and columns templates)
        
            stored in objectClass.getObjects()
            
    """
    
#     def __hash__(self):
#         return sequenceAPI.__hash__(self)
    
    def __init__(self):
        objectClass.__init__(self)
        sequenceAPI.__init__(self)
        
        # type of template: vertical zones,...: or use self._name ??? 
        self._templateType = None
        
        
    def setType(self,t): self._templateType = t
    def getType(self): return self._templateType
    def registration(self,object):
        raise "SOFTWARE ERROR: your component must define a testRun method"

    def tagDom(self,dom):
        raise "SOFTWARE ERROR: your component must define a testRun method"        
    