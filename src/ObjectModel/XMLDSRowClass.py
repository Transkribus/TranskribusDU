# -*- coding: utf-8 -*-
"""

    XMLDS ROW
    Hervé Déjean
    cpy Xerox 2017
    
    a class for table row from a XMLDocument

"""

from XMLDSObjectClass import XMLDSObjectClass
from config import ds_xml_def as ds_xml

class  XMLDSTABLEROWClass(XMLDSObjectClass):
    """
        LINE class
    """
    name = ds_xml.sROW
    def __init__(self,index,domNode = None):
        XMLDSObjectClass.__init__(self)
        XMLDSObjectClass.id += 1
        self._domNode = domNode
        self.tagName = 'ROW'
        self._index= index
        self._lcells=[]
        self.setName(XMLDSTABLEROWClass.name)
    
    def __repr__(self):
        return "%s %s"%(self.getName(),self.getIndex())  
    def __str__(self):
        return "%s %s"%(self.getName(),self.getIndex())          
        
    def getID(self): return self.getIndex()
    def getIndex(self): return self._index
    def setIndex(self,i): self._index = i
    
    def getCells(self): return self._lcells
    def addCell(self,c): 
        if c not in self.getCells():
            self._lcells.append(c)            
            self.addObject(c)
#             if c.getNode() is not None and self.getNode() is not None:
#                 c.getNode().unlinkNode()
#                 self.getNode().addChild(c.getNode())

    
    ########## TAGGING ##############
    def addField(self,tag):
        [cell.addField(tag) for cell in self.getCells()]


        
