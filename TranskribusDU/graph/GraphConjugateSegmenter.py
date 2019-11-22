# -*- coding: utf-8 -*-

"""
    Conjugate Graph Class used for Segmentation

    Copyright NAVER(C) 2019 JL. Meunier
"""


from common.trace import traceln

from graph.GraphConjugate import GraphConjugate


class GraphConjugateSegmenter(GraphConjugate):
    
    bConjugate = True
    
    def __init__(self, lNode = [], lEdge = []):
        super(GraphConjugateSegmenter, self).__init__(lNode, lEdge)
    
    # --- Clusters -------------------------------------------------------
    def addClusterToDoc(self, lCluster):
        """
        Do whatever is required with the ClusterList object
        """
        raise Exception("To be specialised!")
    
    def setDocLabels(self, Y):
        """
        Set the labels of the graph nodes from the Y matrix
        Is this used anywhere???
        """
        GraphConjugate.setDocLabels(self, Y)
        self.lCluster = self.form_cluster(Y)
        
        self.addClusterToDoc(self.lCluster)
        traceln(" %d cluster(s) found" % (len(self.lCluster)))
