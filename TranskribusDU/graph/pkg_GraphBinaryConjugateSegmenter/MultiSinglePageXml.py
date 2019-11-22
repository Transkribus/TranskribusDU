# -*- coding: utf-8 -*-

"""
    Multi single PageXml graph in conjugate mode

    Copyright NAVER(C) 2019 JL. Meunier
"""

from .GraphBinaryConjugateSegmenter_DOM         import GraphBinaryConjugateSegmenter_DOM
from graph.Graph_Multi_SinglePageXml            import Graph_MultiSinglePageXml


class MultiSinglePageXml(
          GraphBinaryConjugateSegmenter_DOM
        , Graph_MultiSinglePageXml):
    """
    Multi single PageXml graph in conjugate mode
    """
    # --- NODE TYPES and LABELS
    _lNodeType       = []       #the list of node types for this class of graph
    _bMultitype      = False    # equivalent to len(_lNodeType) > 1
    _dLabelByCls     = None     #dictionary across node types
    _dClsByLabel     = None     #dictionary across node types
    _nbLabelTot      = 0        #total number of labels
    
    def __init__(self):
        super(MultiSinglePageXml, self).__init__()

