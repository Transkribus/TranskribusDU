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
    
    def getIndex(self): return self._index
    def setIndex(self,i): self._index = i
    
    def getCells(self): return self._lcells
    def addCell(self,c): 
        if c not in self.getCells():
            self._lcells.append(c)            
            self.addObject(c)
            if c.getNode() is not None and self.getNode() is not None:
                c.getNode().unlinkNode()
                self.getNode().addChild(c.getNode())

    
    ########## TAGGING ##############
    def addField(self,tag):
        [cell.addField(tag) for cell in self.getCells()]



#     ## possible
#     def fromDom(self,domNode):
#         """
#             only contains TEXT?
#         """
#         self.setName(domNode.name)
#         self.setNode(domNode)
#         # get properties
#         prop = domNode.properties
#         while prop:
#             self.addAttribute(prop.name,prop.getContent())
#             # add attributes
#             prop = prop.next
#         self.setIndex(int(self.getAttribute('irow')),int(self.getAttribute('icol')))
#             
#         ctxt = domNode.doc.xpathNewContext()
#         ctxt.setContextNode(domNode)
#         ldomElts = ctxt.xpathEval('./%s'%(ds_xml.sCELL))
#         ctxt.xpathFreeContext()
#         for elt in ldomElts:
#             myObject= XMLDSTEXTClass(elt)
#             self.addObject(myObject)
#             myObject.setPage(self.getPage())
#             myObject.fromDom(elt)
        
         
        
