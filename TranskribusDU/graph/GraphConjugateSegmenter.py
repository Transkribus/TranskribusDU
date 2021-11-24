# -*- coding: utf-8 -*-

"""
    Conjugate Graph Class used for Segmentation

    Copyright NAVER(C) 2019 JL. Meunier
"""


from common.trace import traceln

from graph.GraphConjugate import GraphConjugate


class GraphConjugateSegmenter(GraphConjugate):
    
    bConjugate = True
    
    lEdgeLabel  = None # for binary segmenter: ["continue", "break"]
    nbEdgeLabel = None # for binary segmenter: 2


    def __init__(self, lNode = [], lEdge = []):
        super(GraphConjugateSegmenter, self).__init__(lNode, lEdge)
    
    @classmethod
    def _setlEdgeLabel(cls, lEdgeLabel):
        cls.lEdgeLabel = lEdgeLabel
        cls.nbEdgeLabel = len(lEdgeLabel)

    @classmethod
    def getEdgeLabels(cls): return cls.lEdgeLabel
        
    # --- Clusters -------------------------------------------------------
    def addClusterToDoc(self, lCluster, sAlgo=None):
        """
        Do whatever is required with the ClusterList object
        """
        raise Exception("To be specialised!")
    
    def setDocLabels(self, Y,TH=0.5):
        """
        Set the labels of the graph nodes from the Y matrix
        """
        GraphConjugate.setDocLabels(self, Y)
        self.lCluster.append(self.form_cluster(Y,TH))
        
        # just the last one just generated 
        self.addClusterToDoc(self.lCluster[-1])
        traceln(" %d cluster(s) found" % (len(self.lCluster[-1])))
