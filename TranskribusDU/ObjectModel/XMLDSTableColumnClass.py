# -*- coding: utf-8 -*-
"""

    XMLDS COLUMN
    Hervé Déjean
    cpy Xerox 2017
    
    a class for table column from a XMLDocument

    READ project 


    
    
    Developed  for the EU project READ. The READ project has received funding 
    from the European Union's Horizon 2020 research and innovation programme 
    under grant agreement No 674943.
"""
from __future__ import absolute_import
from __future__ import  print_function
from __future__ import unicode_literals

from .XMLDSObjectClass import XMLDSObjectClass
from config import ds_xml_def as ds_xml

class  XMLDSTABLECOLUMNClass(XMLDSObjectClass):
    """
        Column class
    """
    name = ds_xml.sCOL_Elt
    def __init__(self,index=None,domNode = None):
        XMLDSObjectClass.__init__(self)
        XMLDSObjectClass.id += 1
        self._domNode = domNode
        self._index= index
        self._lcells=[]
        self.tagName= 'COL'
        self.setName(XMLDSTABLECOLUMNClass.name)
    
    def __repr__(self):
        return "%s %s"%(self.getName(),self.getIndex())        
    def __str__(self):
        return "%s %s"%(self.getName(),self.getIndex())       
    
    def getIndex(self): return self._index
    def setIndex(self,i): self._index = i
    
    def delCell(self,cell):
        try:self._lcells.remove(cell)
        except:pass
    def getCells(self): return self._lcells
    def addCell(self,c): 
        if c not in self.getCells():
            self._lcells.append(c)
            self.addObject(c)                

    ########## LABELLING ##############
    def addField(self,field):
        [cell.addField(field) for cell in self.getCells()]





     
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
        
         
        
