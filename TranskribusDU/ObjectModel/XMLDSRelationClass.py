# -*- coding: utf-8 -*-
"""

    object class 
    
    Hervé Déjean
    cpy Naver Labs Europe 2019
    
    a class for (binary) relation 

"""
from __future__ import absolute_import
from __future__ import  print_function
from __future__ import unicode_literals
from .relationClass import relationClass

class  XMLDSRelationClass(relationClass):
    """
        (binary) relation class
    """
    
    id = 0
    def __init__(self):
        super(relationClass,self).__init__()
        self._name = None
        self.src =None
        self.tgt = None
        # characteristics
        self._lAttributes = {}

    

    def fromDom(self,node):
        """
            <EDGE src="line_1500973670363_2513" tgt="line_1500973672753_2514" type="HorizontalEdge" w="0.999994" points="122.88,111.60 23.28,111.36"/> 
        """
        self.setSourceId(node.get('src'))
        self.setTargetId(node.get('tgt'))
        for prop in node.keys():
            if prop not in ['src','tgt']:
                self.addAttribute(prop,node.get(prop))