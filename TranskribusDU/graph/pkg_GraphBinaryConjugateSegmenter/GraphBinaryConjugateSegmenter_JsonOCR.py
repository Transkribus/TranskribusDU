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
from common.trace import traceln

from graph.Graph_JsonOCR import Graph_JsonOCR
from .GraphBinaryConjugateSegmenter import GraphBinaryConjugateSegmenter 


class GraphBinaryConjugateSegmenter_jsonOCR(GraphBinaryConjugateSegmenter, Graph_JsonOCR):
    # --- NODE TYPES and LABELS
    _lNodeType       = []       #the list of node types for this class of graph
    _bMultitype      = False    # equivalent to len(_lNodeType) > 1
    _dLabelByCls     = None     #dictionary across node types
    _dClsByLabel     = None     #dictionary across node types
    _nbLabelTot      = 0        #total number of labels
    
    bWARN_TODO_addClusterToDoc = True
    bWARN_TODO_addEdgeToDoc    = True
    
    def __init__(self, lNode = [], lEdge = [], sOuputXmlAttribute=None):
        GraphBinaryConjugateSegmenter.__init__(self, sOuputXmlAttribute=sOuputXmlAttribute)
        Graph_JsonOCR.__init__(self, lNode, lEdge)
   
    def addClusterToDoc(self, dCluster, sAlgo=None):
        """
        JSON OCR version
        """
        # TODO
        if self.bWARN_TODO_addClusterToDoc: 
            traceln(self, "  addClusterToDoc  ", "not implemented")
            self.bWARN_TODO_addClusterToDoc = False
        return

        
    def addEdgeToDoc(self):
        """
        JSON OCR version
        """
        # TODO
        if self.bWARN_TODO_addEdgeToDoc: 
            traceln(self, "  addEdgeToDoc  ", "not implemented")
            self.bWARN_TODO_addEdgeToDoc = False
        return        
