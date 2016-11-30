# -*- coding: utf-8 -*-
"""
    PAGE Document class 
    Hervé Déjean
    cpy Xerox 2016

    a class for vertical page-zone document
    
    Developed  for the EU project READ. The READ project has received funding 
    from the European Union’s Horizon 2020 research and innovation programme 
    under grant agreement No 674943.
    
"""

import sys, os.path

sys.path.append(os.path.dirname(os.path.abspath(sys.argv[0])))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0]))))

from XMLObjectClass import XMLObjectClass
from XMLDSPageClass import XMLDSPageClass

from config import ds_xml_def as ds_xml

class  XMLVERTICALPAGEZONE(XMLDSObjectClass):
    """ 
    """
    
    def __init__(self,domDoc=None):
        XMLDSObjectClass.__init__(self)
        XMLDSObjectClass.id += 1        
        
        self._domNode = domNode
        
        self._parent= None  (XMLDSPageClass)  # XMLDSPageClass must have a link to a list of vertical zones  (and model?) 
        
        self._nextVZone = None # next vertical zone in the page
        self._prevVZone = None # prev vertical zone in the page
        
        # x, y, w, h, x2, : already defined
        
        
        

