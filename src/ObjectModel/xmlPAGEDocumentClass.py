# -*- coding: utf-8 -*-
"""
    PAGE Document class 
    Hervé Déjean
    cpy Xerox 2016

    a class for PAGE document
    
    Developed  for the EU project READ. The READ project has received funding 
    from the European Union’s Horizon 2020 research and innovation programme 
    under grant agreement No 674943.
    
"""

import sys, os.path

sys.path.append(os.path.dirname(os.path.abspath(sys.argv[0])))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0]))))

from xmlDocumentClass import XMLDocument
from XMLObjectClass import XMLObjectClass
from XMLDSPageClass import XMLDSPageClass

from config import ds_xml_def as ds_xml

class  XMLPAGEDocument(XMLDocument):
    """ 
        Representation of a XML document using PAGE schema: 
        http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15' 
    """
    
    def __init__(self,domDoc=None):
        self._type = "XMLPAGEDocument"
        self._dom = domDoc
        self._rootObject = None
        XMLDocument.__init__(self)        
    
    
    def loadFromFolder(self,pathname):
        """
            For each node, create an object with the tag name. features?
        
            assume specbook/division/PAGE
        """
        if docDom:
            self.setDom(docDom)
        if self.getDom():
            self._rootObject = XMLObjectClass()
            self.loadPages(self.getDom().getRootElement())
        else:
            return -1
        

    def display(self,lvl=0):
        print 'Document: ',self.getName()
        self.getRootObject().display(lvl+1)

