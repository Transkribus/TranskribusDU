# -*- coding: utf-8 -*-
"""
    document class 
    Hervé Déjean
    cpy Xerox 2013

    a (abstract) class for document
"""

from objectClass import objectClass 

class  documentObject(objectClass):
    """ 
        Representation of a document (a tree)
    """
    
    def __init__(self):
        objectClass.__init__(self)
        self._type = None
        #root?
        self._lObjects = []
        self._name='document'
        
    def getObjects(self):
        return self._lObjects
    
    def getNamedObjects(self,objectName):
        """
            objectName : instanceName  OR string
            
            here hierarchical ?
        """    
        lList = []
        for elt in self.getObjects():
            try:
                if isinstance(elt,objectName):
                    lList.append(elt)
            except:
                if elt.getName() == objectName:
                    lList.append(elt)
                
            
        return lList
