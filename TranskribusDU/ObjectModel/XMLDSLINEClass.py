# -*- coding: utf-8 -*-
"""

    XMLDS line object class 
    
    Hervé Déjean
    cpy Xerox 2009
    
    a class for line from a XMLDocument

"""
from __future__ import absolute_import
from __future__ import  print_function
from __future__ import unicode_literals

from .XMLDSObjectClass import XMLDSObjectClass
from .XMLDSTEXTClass import XMLDSTEXTClass

from config import ds_xml_def as ds_xml

class  XMLDSLINEClass(XMLDSObjectClass):
    """
        LINE class
    """
    name = ds_xml.sLINE_Elt
    def __init__(self,domNode = None):
        XMLDSObjectClass.__init__(self)
        XMLDSObjectClass.id += 1
        self._domNode = domNode
    
    def fromDom(self,domNode):
        """
            only contains TEXT?
        """
        self.setName(domNode.tag)
        self.setNode(domNode)
        
        # get properties
        for prop in domNode.keys():
            self.addAttribute(prop,domNode.get(prop))
            
#         ctxt = domNode.doc.xpathNewContext()
#         ctxt.setContextNode(domNode)
#         ldomElts = ctxt.xpathEval('./%s'%(ds_xml.sTEXT))
#         ctxt.xpathFreeContext()
        ldomElts = domNode.findall('./%s'%(ds_xml.sTEXT))
        for elt in ldomElts:
            myObject= XMLDSTEXTClass(elt)
            self.addObject(myObject)
            myObject.setPage(self.getPage())
            myObject.fromDom(elt)        
        
         
        
