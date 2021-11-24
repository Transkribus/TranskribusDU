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

from graph.GraphConjugateSegmenter import GraphConjugateSegmenter
from graph.pkg_GraphBinaryConjugateSegmenter.I_GraphBinaryConjugateClusterer import I_GraphBinaryConjugateClusterer


class GraphBinaryConjugateSegmenter(I_GraphBinaryConjugateClusterer, GraphConjugateSegmenter):
    """
    What is independent from the input format (XML or JSON, currently)
    """
    _balancedWeights = False   #  Uniform does same or better, in general

    lEdgeLabel  = ["continue", "break"]
    nbEdgeLabel = 2

    
    def __init__(self, sOuputXmlAttribute=None):
        """
        a CRF model, with a name and a folder where it will be stored or retrieved from
        
        the cluster index of each object with be store in an Xml attribute of given name
        """
        I_GraphBinaryConjugateClusterer.__init__(self, sOuputXmlAttribute)
        GraphConjugateSegmenter.__init__(self)
        

    def parseDocLabels(self):
        """
        Parse the label of the graph from the dataset, and set the node label
        return the set of observed class (set of integers in N+)
        
        Here, no check at all, because we just need to see if two labels are the same or not.
        """
        setSeensLabels = set()
        for nd in self.lNode:
            nodeType = nd.type 
            sLabel = nodeType.parseDocNodeLabel(nd)
            try:
                cls = self._dClsByLabel[sLabel]  #Here, if a node is not labelled, and no default label is set, then KeyError!!!
            except KeyError:
                cls = len(self._dClsByLabel)
                self._dClsByLabel[sLabel] = cls
            nd.cls = cls
            setSeensLabels.add(cls)
        return setSeensLabels    
    
    def computeEdgeLabels(self):
        """
        Given the loaded graph with labeled nodes, compute the edge labels.
        
        This results in each edge having a .cls attribute.
    
        return the set of observed class (set of integers in N+)
        """
        setSeensLabels = set()
        for edge in self.lEdge:
            edge.cls = 0 if (edge.A.cls == edge.B.cls) else 1
            setSeensLabels.add(edge.cls)
        return setSeensLabels    