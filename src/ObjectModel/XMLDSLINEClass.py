# -*- coding: latin-1 -*-
"""

    XML object class 
    
    Herv� D�jean
    cpy Xerox 2009
    
    a class for object from a XMLDocument

"""

from XMLDSObjectClass import XMLDSObjectClass
from XMLDSTEXTClass import XMLDSTEXTClass
from config import ds_xml_def as ds_xml

class  XMLDSLINEClass(XMLDSObjectClass):
    """
        LINE class
    """
    name = ds_xml.sXY_LINE_Elt
    def __init__(self,domNode = None):
        XMLDSObjectClass.__init__(self)
        XMLDSObjectClass.id += 1
        self._domNode = domNode
    
    def fromDom(self,domNode):
        """
            only contains TEXT?
        """
        self._name = domNode.name
        self.setNode(domNode)
        # get properties
        prop = domNode.properties
        while prop:
            self.addAttribute(prop.name,prop.getContent())
            # add attributes
            prop = prop.next
            
            
        ctxt = domNode.doc.dom.xpathNewContext()
        ctxt.setContextNode(domNode)
        ldomElts = ctxt.xpathEval('./%s'%(ds_xml.sTEXT))
        ctxt.xpathFreeContext()
        for elt in ldomElts:
            myObject= XMLDSTEXTClass(elt)
            self.addObject(myObject)
            myObject.setPage(self.getPage())
            myObject.fromDom(elt)        
            print myObject.getContent()    
        
         
        
