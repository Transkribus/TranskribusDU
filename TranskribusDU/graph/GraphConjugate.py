# -*- coding: utf-8 -*-

"""
    Conjugate Graph Class

    Copyright NAVER(C) 2019 JL. Meunier
"""
import numpy as np

from graph.Graph import Graph, GraphException


class GraphConjugateException(GraphException): pass


class GraphConjugate(Graph):
    
    bConjugate = True
    
    def __init__(self, lNode = [], lEdge = []):
        super(GraphConjugate, self).__init__(lNode, lEdge)
    
    @classmethod
    def getLabelNameList(cls):
        """
        We work on edges!
        """
        return cls.getEdgeLabelNameList()

    def setDocLabels(self, Y,TH=0.5):
        """
        Set the labels of the graph nodes from the Y matrix
        Is this used anywhere???
        """
        # lEdgeLabel = self.getEdgeLabelNameList()
        for i,edge in enumerate(self.lEdge):
            edge.cls = Y[i]   # might be a Numpy vector! (argmax not yet done)
        return

    # --- Numpy matrices --------------------------------------------------------
    def getX(self, node_transformer, edge_transformer):
        """
        make 1 node-feature matrix     (or list of matrices for multitype graphs)
         and 1 edge-feature matrix     (or list of matrices for multitype graphs)
         and 1 edge matrix             (or list of matrices for multitype graphs)
         for the graph
        return a triplet
        
        return X for the graph
        """
        X = Graph.getX(self, node_transformer, edge_transformer)
        X = self.convert_X_to_LineDual(X)
        return X

    def getY(self):
        """
        WARNING, in multitype graphs, the order of the Ys is bad
        """
        Y = np.fromiter( (e.cls for e in self.lEdge), dtype=np.int, count=len(self.lEdge))
        return Y

    # --- Conjugate --------------------------------------------------------
    def convert_X_to_LineDual(self, X):
        """
        Convert to a dual graph
        Animesh 2018
        Revisited by JL April 2019
        
        NOTE: isolated nodes are not reflected in the dual.
        Should we add a reflexive edge to have the node in the dual??
        """
        if self._bMultitype:
            raise "Not yet implemented: conjugate of multitype graph"
        (nf, edge, ef) = X
        if nf.shape[0] == 0: raise GraphConjugateException("Graph without node: no edge in conjugate graph!")
        nb_edge = edge.shape[0]
        if nb_edge == 0: raise GraphConjugateException("Graph without edge: empty conjugate graph!")
        
        all_edges = []      # all edges created so far

        nf_dual     = ef    # edges become nodes
        edge_dual   = []
        ef_dual     = []            
        
        for i in range(nb_edge):
            edgei_from_idx, edgei_to_idx = edge[i]
            
            edge_from  = set([edgei_from_idx, edgei_to_idx])
            for j in range(i+1, nb_edge):    
                edge_to = set([edge[j][0], edge[j][1]])
                edge_candidate = edge_from.symmetric_difference(edge_to)
                # we should get 4, 2 or 0 primal nodes
                if len(edge_candidate) == 2 and edge_candidate not in all_edges:
                    # edge_to and edge_from share 1 primal node => create dual edge! 
                    all_edges.append(edge_candidate)
                    [shared_node_idx] = edge_from.intersection(edge_to)
                    shared_node_nf = nf[shared_node_idx]
                    ef_dual.append(shared_node_nf)
                    edge_dual.append([i, j])

        if len(edge_dual) == 0: raise GraphConjugateException("Conjugate graph without edge!")
        nf_dual     = np.array(nf_dual)
        edge_dual   = np.array(edge_dual)
        ef_dual     = np.array(ef_dual)

        assert (edge_dual.shape[0] == ef_dual.shape[0])

        return (nf_dual, edge_dual, ef_dual)
