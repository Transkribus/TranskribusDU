# -*- coding: utf-8 -*-

"""
    document class 
    Hervé Déjean
    cpy Xerox 2013

    a class for document
"""
from __future__ import absolute_import
from __future__ import  print_function
from __future__ import unicode_literals

from .documentClass import documentObject 
from .XMLObjectClass import XMLObjectClass

class  XMLDocument(documentObject):
    """ 
        Representation of a document"
        XML format  (same for html->hierarchical stuff html is a subclass of xml ?  (specific loaddom with fuzzy parsing))
    """
    
    def __init__(self,domDoc=None):
        documentObject.__init__(self)
        self._type = "XML"
        self._dom = domDoc
        self._rootObject = None
#         self._name = domDoc.name
        
    
    
    def getRootObject(self):return self._rootObject
    
    ### strategies for generating ordered sequences from a XML document
    def setDom(self,docdom):
        self._dom  = docdom
    def getDom(self): return self._dom
    
    def generateSiblingFlatStructureforObject(self,node,objectName):
        """
            collect flat list of objectName elements (stop eve if node has also objectName elements)
        """
        if node.name == objectName:
            return [node]
        lNodes  = []     
        lsubNodes = []
        for child in node.getObjects():
            lsubNodes =  self.generateFlatStructureforObject(child,objectName)
            if lsubNodes:
                # append?
                lNodes.extend(lsubNodes)
        return lNodes
    
    def treeTraversal(self,node,func,iMin=1):
        """
            apply func
            func must have internal structure to store information
        """
        
        lNodes  = []     
        for elt in node.getObjects():
            self.treeTraversal(elt, func)
            lNodes.append(elt)
        if lNodes:
            if len(lNodes) > iMin:
                func(lNodes)    
    
    def loadFromDom(self,docDom=None):
        """
            For each node, create an object with the tag name. features?
              
        """
        if docDom:
            self.setDom(docDom)
        if self.getDom():
            self._rootObject = XMLObjectClass()
            self._rootObject.fromDom(self.getDom().getroot())
            #self._lObjects = self._rootObject.getObjects()
        else:
            return -1
        
 
    def getTerminalNamedObjects(self,objectName): 
        try:
            if isinstance(self,objectName):
                if self.getObjects()==[]:
                    return [self]
        except TypeError:
#             print self.getName(),self.getObjects()
            if self.getName() == objectName:
                if self.getObjects()==[]:
                    return [self]
            
        lList = []        
        for elt in self.getObjects():
            try:
                if isinstance(elt,objectName):
                    subList= elt.getObjects()
                    if subList == []:
                        lList.append(elt)
                    else: 
                        lList.extend(elt.getAllNamedObjects(objectName))
                else:
                    
                    lList.extend(elt.getAllNamedObjects(objectName))
            except TypeError:
                if elt.getName() == objectName:
                    subList= elt.getObjects()
                    if subList == []:
                        lList.append(elt)
                    else: 
                        lList.extend(elt.getAllNamedObjects(objectName))
                else:
                    lList.extend(elt.getAllNamedObjects(objectName))
                    
        return lList       

    def getAllNamedObjects(self,objectName):
        lList =[]
        try:
            if isinstance(self,objectName):
                lList.append(self)
        except TypeError:
            if self.getName() == objectName:
                lList.append(self)
            
        for elt in self.getObjects():
            lList.extend(elt.getAllNamedObjects(objectName))
                    
        return lList
                

    def display(self,lvl=0):
        print ('Document: ',self.getName())
        self.getRootObject().display(lvl+1)

