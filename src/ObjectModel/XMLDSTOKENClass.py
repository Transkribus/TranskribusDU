# -*- coding: latin-1 -*-
"""

    XML object class 
    
    Hervé Déjean
    cpy Xerox 2009
    
    a class for object from a XMLDocument

"""

from XMLDSObjectClass import XMLDSObjectClass
from config import ds_xml_def as ds_xml

class  XMLDSTOKENClass(XMLDSObjectClass):
    """
        LINE class
    """
    def __init__(self,domNode = None):
        XMLDSObjectClass.__init__(self)
        XMLDSObjectClass.id += 1
        self._domNode = domNode
    
    def fromDom(self,domNode):
        """
            only contains TEXT?
        """
        
        # must be PAGE        
        self.setName(domNode.name)
        self.setNode(domNode)
        # get properties
        # all?
        prop = domNode.properties
        while prop:
            self.addAttribute(prop.name,prop.getContent())
            # add attributes
            prop = prop.next
        self.addContent(domNode.getContent().decode("UTF-8"))
        
        self.addAttribute('x2', float(self.getAttribute('x'))+self.getWidth())
        self.addAttribute('y2',float(self.getAttribute('y'))+self.getHeight() )                
            
        
