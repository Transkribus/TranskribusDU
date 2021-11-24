# -*- coding: utf-8 -*-

"""
    Interface: Clustering function for Conjugate graph

    Structured machine learning, currently using graph-CRF or Edge Convolution Network

    Copyright NAVER(C) 2020 JL. Meunier


"""
import numpy as np

from graph.pkg_GraphBinaryConjugateSegmenter.I_GraphBinaryConjugateClusterer import I_GraphBinaryConjugateClusterer

from util.cluster_node import HierarchicalHypothesisClusterer

class I_GraphHierarchyConjugateClusterer(I_GraphBinaryConjugateClusterer):
    """
    What is independent from the input format (XML or JSON, currently)
    """
   
    def __init__(self, sOuputXmlAttribute=None):
        """
        a CRF model, with a name and a folder where it will be stored or retrieved from
        
        the cluster index of each object with be store in an Xml attribute of given name
        """
        super(I_GraphHierarchyConjugateClusterer, self).__init__(sOuputXmlAttribute=sOuputXmlAttribute)
    
    # ---------------------------------------------------------------------------------------
    def form_cluster_hier(self, Y_proba, N):
        """
        Do a connected component algo
        Return a list of clusters per level, starting by top level 0 to N-1
        so this is a list of length N, of lists of clusters, each cluster being a set of node index in self.lNode
        """
        llCluster = []
        
        # deepest level
        Y_proba_binary = self.getY_proba_at_level(Y_proba, N-1)
        lCluster = self.form_cluster(Y_proba_binary)
        llCluster.insert(0, list(lCluster))

        # moving towards top-level        
        for i in range(N-2, -1, -1):
            Y_proba_binary = self.getY_proba_at_level(Y_proba, i)
            dEdges = self.getEdges(self.lEdge, Y_proba_binary)
            lCluster = self.clusterPlus(lCluster, dEdges)
            llCluster.insert(0, list(lCluster))
            
        return llCluster   

    def getY_proba_at_level(self, Y, lvl):
        """
        convert a Y_level into a Y of type continue/break
        """
        nEdge, nLbl = Y.shape
        assert lvl < nLbl-1, "%d levels, so level %d is inconsistent" % (nLbl-1, lvl)
        
        Ylvl = np.zeros(shape=(nEdge, 2))  # make it a continue/break probability
        sptr = lvl+2
        Ylvl[:,0] = Y[:,0] + Y[:,sptr:].sum(axis=1)
        Ylvl[:,1] = Y[:,1:sptr].sum(axis=1)
        return Ylvl

    # ---------------------------------------------------------------------------------------
    # December 2020 new methods

    def form_cluster_hierarchical_hypothesis(self, Y_proba, N, nBest=5):
        """
        run 5 hypothesis in parallel, for a all levels
        """
        nn, _ne, aAB = self._form_cluster_hypothesis_prepare_data(Y_proba, N)
        
        doer = HierarchicalHypothesisClusterer(nn, N, None, aAB, Y_proba, nBest=nBest)
        
        ltScorellCluster = doer.clusterNBest()
        
        return ltScorellCluster[0][1]     # take first solution, return the list per level of clusters