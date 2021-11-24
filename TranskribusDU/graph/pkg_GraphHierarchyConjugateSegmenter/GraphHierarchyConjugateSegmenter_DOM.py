# -*- coding: utf-8 -*-

"""
    Train, test, predict steps for a graph-based model using a binary conjugate 
    (two classes on the primal edges)

    Structured machine learning, currently using graph-CRF or Edge Convolution Network

    Copyright NAVER(C) 2019 JL. Meunier


    
    
    Developed  for the EU project READ. The READ project has received funding 
    from the European Union�s Horizon 2020 research and innovation programme 
    under grant agreement No 674943.
    
"""
from graph.Graph_DOM import Graph_DOM
from .GraphHierarchyConjugateSegmenter import GraphHierarchyConjugateSegmenter

from .I_GraphHierarchyConjugateClusterer_DOM import I_GraphHierarchyConjugateClusterer_DOM


class GraphHierarchyConjugateSegmenter_DOM(I_GraphHierarchyConjugateClusterer_DOM, GraphHierarchyConjugateSegmenter, Graph_DOM):
    # --- NODE TYPES and LABELS
    _lNodeType       = []       #the list of node types for this class of graph
    _bMultitype      = False    # equivalent to len(_lNodeType) > 1
    _dLabelByCls     = None     #dictionary across node types
    _dClsByLabel     = None     #dictionary across node types
    _nbLabelTot      = 0        #total number of labels
    
    def __init__(self, lNode = [], lEdge = [], sOuputXmlAttribute=None):
        sOuputXmlAttribute
        I_GraphHierarchyConjugateClusterer_DOM.__init__(self)
        GraphHierarchyConjugateSegmenter.__init__(self)
        Graph_DOM.__init__(self, lNode, lEdge)
    
