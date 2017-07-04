# -*- coding: utf-8 -*-
"""

    XMLDS TABLE
    Hervé Déjean
    cpy Xerox 2013
    
    a class for table from a XMLDocument

"""

from XMLDSObjectClass import XMLDSObjectClass
from XMLDSCELLClass import XMLDSTABLECELLClass
from config import ds_xml_def as ds_xml

class  XMLDSTABLEClass(XMLDSObjectClass):
    """
        LINE class
    """
    name = ds_xml.sLINE_Elt
    def __init__(self,domNode = None):
        XMLDSObjectClass.__init__(self)
        XMLDSObjectClass.id += 1
        self._domNode = domNode
    
    
    def addCell(self,cell):
        """
            add a  cell
            update row and col data structure 
        """
        self.addObject(cell)
        
    def fromDom(self,domNode):
        """
            only contains TEXT?
        """
        self.setName(domNode.name)
        self.setNode(domNode)
        # get properties
        prop = domNode.properties
        while prop:
            self.addAttribute(prop.name,prop.getContent())
            # add attributes
            prop = prop.next
            
            
        ctxt = domNode.doc.xpathNewContext()
        ctxt.setContextNode(domNode)
        ldomElts = ctxt.xpathEval('./%s'%(ds_xml.sCELL))
        ctxt.xpathFreeContext()
        for elt in ldomElts:
            myObject= XMLDSTABLECELLClass(elt)
            self.addCell(myObject)
            myObject.setPage(self.getPage())
            myObject.fromDom(elt)        
        
         
        
